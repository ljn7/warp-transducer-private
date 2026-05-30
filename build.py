"""
build.py - One-command build and install for warp-transducer + Python binding.

Usage:
    python build.py               # auto-detect CUDA, build everything, install
    python build.py --no-gpu      # force CPU-only build
    python build.py --no-install  # build C library only, do not pip install
    python build.py --build-dir path/to/dir  # custom build directory

What this script does:
    1. Checks required tools (cmake, a C++ compiler)
    2. Detects CUDA (unless --no-gpu)
    3. Builds the C library (warprnnt.dll / libwarprnnt.so)
    4. Runs "pip install ." in the pytorch_binding directory

Tested on:
    - Windows 10/11 with MSVC (Visual Studio Build Tools) + CUDA
    - Linux with GCC + CUDA
    - Linux with GCC, CPU-only
    - macOS with Clang, CPU-only
"""

import argparse
import os
import platform
import shutil
import subprocess
import sys

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------
IS_WINDOWS = platform.system() == "Windows"
IS_MACOS   = platform.system() == "Darwin"

def run(cmd, cwd=None, env=None, description=""):
    """Run a command, print it, and raise on failure."""
    label = description or " ".join(str(c) for c in cmd)
    print()
    print(">>> " + label)
    print("    " + " ".join(str(c) for c in cmd))
    result = subprocess.run(cmd, cwd=cwd, env=env)
    if result.returncode != 0:
        print()
        print("FAILED (exit code {}): {}".format(result.returncode, label))
        sys.exit(result.returncode)

def find_tool(name, extra_paths=None):
    """Return the full path to a tool, or None if not found."""
    found = shutil.which(name)
    if found:
        return found
    for p in (extra_paths or []):
        candidate = os.path.join(p, name)
        if os.path.isfile(candidate):
            return candidate
        candidate = os.path.join(p, name + ".exe")
        if os.path.isfile(candidate):
            return candidate
    return None

def detect_cuda():
    """Return the CUDA root directory, or None if not found."""
    # Already set by user
    for var in ("CUDA_HOME", "CUDA_PATH"):
        val = os.environ.get(var)
        if val and os.path.isdir(val):
            return val

    # Common installation paths
    candidates = []
    if IS_WINDOWS:
        cuda_base = r"C:\Program Files\NVIDIA GPU Computing Toolkit\CUDA"
        if os.path.isdir(cuda_base):
            versions = sorted(os.listdir(cuda_base), reverse=True)
            candidates += [os.path.join(cuda_base, v) for v in versions]
    else:
        candidates += [
            "/usr/local/cuda",
            "/usr/cuda",
        ]
        # Also check /usr/local/cuda-XX.Y
        import glob
        candidates += sorted(glob.glob("/usr/local/cuda-*"), reverse=True)

    for path in candidates:
        nvcc = os.path.join(path, "bin", "nvcc" + (".exe" if IS_WINDOWS else ""))
        if os.path.isfile(nvcc):
            return path

    # Last resort: is nvcc anywhere in PATH?
    nvcc = shutil.which("nvcc")
    if nvcc:
        # nvcc lives at <cuda_root>/bin/nvcc
        return os.path.dirname(os.path.dirname(nvcc))

    return None

def detect_ninja():
    """Find Ninja. On Windows it ships with VS Build Tools."""
    found = find_tool("ninja")
    if found:
        return found

    if IS_WINDOWS:
        # Common Visual Studio / Build Tools locations
        import glob
        patterns = [
            r"C:\Program Files\Microsoft Visual Studio\*\*\Common7\IDE\CommonExtensions\Microsoft\CMake\Ninja\ninja.exe",
            r"C:\Program Files (x86)\Microsoft Visual Studio\*\*\Common7\IDE\CommonExtensions\Microsoft\CMake\Ninja\ninja.exe",
        ]
        for pattern in patterns:
            matches = sorted(glob.glob(pattern), reverse=True)
            if matches:
                return matches[0]
    return None

# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
def main():
    parser = argparse.ArgumentParser(
        description="Build warp-transducer C library and optionally install the PyTorch binding."
    )
    parser.add_argument("--no-gpu", action="store_true",
                        help="Force CPU-only build even if CUDA is detected.")
    parser.add_argument("--no-install", action="store_true",
                        help="Build the C library only; skip 'pip install'.")
    parser.add_argument("--build-dir", default="build",
                        help="Directory to use for the cmake build (default: build).")
    parser.add_argument("--generator", default=None,
                        help="CMake generator override (e.g. 'Ninja', 'Unix Makefiles').")
    args = parser.parse_args()

    repo_root = os.path.dirname(os.path.abspath(__file__))
    build_dir = os.path.join(repo_root, args.build_dir)
    binding_dir = os.path.join(repo_root, "pytorch_binding")

    print()
    print("=" * 60)
    print(" warp-transducer build script")
    print("=" * 60)
    print(" Platform:    {}".format(platform.system()))
    print(" Python:      {}".format(sys.version.split()[0]))
    print(" Repo root:   {}".format(repo_root))
    print(" Build dir:   {}".format(build_dir))

    # -------------------------------------------------------------------
    # Check cmake
    # -------------------------------------------------------------------
    cmake = find_tool("cmake")
    if not cmake:
        print()
        print("ERROR: cmake not found in PATH.")
        print("Install cmake from https://cmake.org/download/ and re-run.")
        sys.exit(1)

    cmake_version = subprocess.check_output([cmake, "--version"], text=True).split()[2]
    print(" CMake:       {}".format(cmake_version))

    # -------------------------------------------------------------------
    # CUDA detection
    # -------------------------------------------------------------------
    cuda_root = None
    if not args.no_gpu:
        cuda_root = detect_cuda()

    if cuda_root:
        print(" CUDA root:   {}".format(cuda_root))
        print(" GPU build:   YES")
    else:
        if not args.no_gpu:
            print(" CUDA root:   not found  (building CPU-only)")
        else:
            print(" GPU build:   NO (--no-gpu)")

    # -------------------------------------------------------------------
    # Choose a CMake generator
    # -------------------------------------------------------------------
    generator = args.generator
    if generator is None:
        ninja = detect_ninja()
        if ninja:
            generator = "Ninja"
            # Make sure ninja is in PATH for cmake to find it
            ninja_dir = os.path.dirname(ninja)
            if ninja_dir not in os.environ.get("PATH", ""):
                os.environ["PATH"] = ninja_dir + os.pathsep + os.environ.get("PATH", "")
            print(" Generator:   Ninja ({})".format(ninja))
        elif IS_WINDOWS:
            # Fall back to NMake if ninja not found
            generator = "NMake Makefiles"
            print(" Generator:   NMake Makefiles (Ninja not found)")
        else:
            generator = "Unix Makefiles"
            print(" Generator:   Unix Makefiles")

    print("=" * 60)

    # -------------------------------------------------------------------
    # Step 1: cmake configure
    # -------------------------------------------------------------------
    os.makedirs(build_dir, exist_ok=True)

    cmake_configure = [
        cmake,
        repo_root,
        "-G", generator,
        "-DCMAKE_BUILD_TYPE=Release",
    ]

    if args.no_gpu or cuda_root is None:
        cmake_configure += ["-DWITH_GPU=OFF"]
    else:
        # Help cmake find CUDA if it is not in a standard place
        cuda_bin = os.path.join(cuda_root, "bin")
        if cuda_bin not in os.environ.get("PATH", ""):
            os.environ["PATH"] = cuda_bin + os.pathsep + os.environ.get("PATH", "")

    run(cmake_configure, cwd=build_dir, description="CMake configure")

    # -------------------------------------------------------------------
    # Step 2: cmake build
    # -------------------------------------------------------------------
    cmake_build = [cmake, "--build", ".", "--config", "Release"]
    if not IS_WINDOWS:
        import multiprocessing
        cmake_build += ["--", "-j{}".format(multiprocessing.cpu_count())]

    run(cmake_build, cwd=build_dir, description="Build C library")

    # -------------------------------------------------------------------
    # Step 3: verify the library was built
    # -------------------------------------------------------------------
    if IS_WINDOWS:
        lib_name = "warprnnt.dll"
    elif IS_MACOS:
        lib_name = "libwarprnnt.dylib"
    else:
        lib_name = "libwarprnnt.so"

    lib_path = os.path.join(build_dir, lib_name)
    if not os.path.exists(lib_path):
        print()
        print("ERROR: Build finished but {} was not found in:".format(lib_name))
        print("  {}".format(build_dir))
        print("Check the build output above for errors.")
        sys.exit(1)

    print()
    print("C library built successfully: {}".format(lib_path))

    # -------------------------------------------------------------------
    # Step 4: pip install the PyTorch binding
    # -------------------------------------------------------------------
    if args.no_install:
        print()
        print("Skipping pip install (--no-install was set).")
        print("To install later, from pytorch_binding/ run:")
        print("  WARP_RNNT_PATH={} pip install .".format(build_dir))
        return

    pip_env = os.environ.copy()
    pip_env["WARP_RNNT_PATH"] = build_dir

    # Tell the binding where CUDA is if we found it
    if cuda_root:
        pip_env.setdefault("CUDA_HOME", cuda_root)
        pip_env.setdefault("CUDA_PATH", cuda_root)

    run(
        [sys.executable, "-m", "pip", "install", "."],
        cwd=binding_dir,
        env=pip_env,
        description="pip install pytorch binding",
    )

    print()
    print("=" * 60)
    print(" Installation complete.")
    print()
    print(" To verify, open Python and run:")
    print("   from warprnnt_pytorch import RNNTLoss")
    print("   loss = RNNTLoss()")
    print("   print('OK')")
    print("=" * 60)

if __name__ == "__main__":
    main()
