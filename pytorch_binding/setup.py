"""
setup.py for warprnnt_pytorch.

Supports Linux, macOS, and Windows.

BUILD BEHAVIOUR
---------------
1. If WARP_RNNT_PATH is set, the pre-built library at that path is used.
2. If the library already exists at the default ../build path, it is used.
3. Otherwise, cmake is invoked automatically to build the C library first,
   then the Python extension is compiled and linked against it.

This means the following all work without any manual cmake step:

    pip install .                         # from inside pytorch_binding/
    pip install git+https://github.com/…  # installs from git clone
    python build.py                       # top-level automation script

PYPI NOTE
---------
A source distribution uploaded to PyPI (pip install warprnnt-pytorch) will
also work automatically IF the user has cmake and a C++ compiler installed.
For truly zero-dependency installs, binary wheels are needed (see README).
"""

import multiprocessing
import os
import platform
import shutil
import subprocess
import sys

from packaging.version import Version
from setuptools import find_packages, setup
from setuptools.command.build_ext import build_ext as _build_ext

import torch
from torch.utils.cpp_extension import BuildExtension, CppExtension

# ---------------------------------------------------------------------------
# Platform detection
# ---------------------------------------------------------------------------
IS_WINDOWS = platform.system() == "Windows"
IS_MACOS   = platform.system() == "Darwin"

# ---------------------------------------------------------------------------
# Library names by platform
#
#   Linux:   libwarprnnt.so    (shared object, also used as import lib)
#   macOS:   libwarprnnt.dylib
#   Windows: warprnnt.dll      (the DLL loaded at runtime)
#            warprnnt.lib      (the import library used by the linker)
# ---------------------------------------------------------------------------
if IS_WINDOWS:
    _runtime_lib = "warprnnt.dll"
    _link_lib    = "warprnnt.lib"
elif IS_MACOS:
    _runtime_lib = "libwarprnnt.dylib"
    _link_lib    = _runtime_lib
else:
    _runtime_lib = "libwarprnnt.so"
    _link_lib    = _runtime_lib

# ---------------------------------------------------------------------------
# Locate the C library
#
# Search order:
#   1. WARP_RNNT_PATH environment variable (user override)
#   2. ../build   (default: the cmake build directory relative to this file)
# ---------------------------------------------------------------------------
_this_dir  = os.path.dirname(os.path.abspath(__file__))
_repo_root = os.path.dirname(_this_dir)          # one level up from pytorch_binding/
_default_build = os.path.join(_repo_root, "build")

warp_rnnt_path = os.environ.get("WARP_RNNT_PATH", _default_build)
warp_rnnt_path = os.path.realpath(warp_rnnt_path)

_link_lib_path    = os.path.join(warp_rnnt_path, _link_lib)
_runtime_lib_path = os.path.join(warp_rnnt_path, _runtime_lib)

# ---------------------------------------------------------------------------
# Auto-build the C library with cmake if it is not already present
# ---------------------------------------------------------------------------
def _find_ninja():
    """Find ninja executable. On Windows it ships inside VS Build Tools."""
    found = shutil.which("ninja")
    if found:
        return found
    if IS_WINDOWS:
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


def _cmake_build_c_library():
    """
    Run cmake to build the warprnnt C library.

    Called automatically when the library is not already present.
    Requires cmake to be in PATH (or findable by the OS).
    """
    cmake = shutil.which("cmake")
    if not cmake:
        print()
        print("ERROR: cmake not found in PATH.")
        print("Install cmake from https://cmake.org/download/ and re-run.")
        print("Or build the library manually and set WARP_RNNT_PATH.")
        sys.exit(1)

    # The C library's CMakeLists.txt is in the repo root (one level up)
    cmake_src = _repo_root
    if not os.path.exists(os.path.join(cmake_src, "CMakeLists.txt")):
        # Might be an sdist where only the pytorch_binding contents are present
        print()
        print("ERROR: CMakeLists.txt not found at:")
        print("  {}".format(cmake_src))
        print()
        print("This usually means you are installing from a PyPI source distribution")
        print("that does not include the C library source.")
        print()
        print("Options:")
        print("  1. Clone the full repo and run: pip install pytorch_binding/")
        print("  2. Set WARP_RNNT_PATH to a directory with the pre-built library.")
        print("  3. Use a binary wheel when one is available for your platform.")
        sys.exit(1)

    os.makedirs(warp_rnnt_path, exist_ok=True)

    # Choose cmake generator
    cmake_args = [cmake, cmake_src, "-DCMAKE_BUILD_TYPE=Release"]
    ninja = _find_ninja()
    if ninja:
        cmake_args += ["-G", "Ninja"]
        ninja_dir = os.path.dirname(ninja)
        if ninja_dir not in os.environ.get("PATH", ""):
            os.environ["PATH"] = ninja_dir + os.pathsep + os.environ.get("PATH", "")
    elif IS_WINDOWS:
        cmake_args += ["-G", "NMake Makefiles"]

    print()
    print("Building warprnnt C library...")
    print("  Source: {}".format(cmake_src))
    print("  Output: {}".format(warp_rnnt_path))
    print()

    try:
        subprocess.check_call(cmake_args, cwd=warp_rnnt_path)
        build_cmd = [cmake, "--build", ".", "--config", "Release"]
        if not IS_WINDOWS:
            build_cmd += ["--", "-j{}".format(multiprocessing.cpu_count())]
        subprocess.check_call(build_cmd, cwd=warp_rnnt_path)
    except subprocess.CalledProcessError as e:
        print()
        print("ERROR: cmake build failed (exit {}).".format(e.returncode))
        print("Check the output above for compiler errors.")
        sys.exit(e.returncode)

    if not os.path.exists(_link_lib_path):
        print()
        print("ERROR: cmake succeeded but {} was not found.".format(_link_lib))
        sys.exit(1)

    print()
    print("C library built: {}".format(_link_lib_path))


# Run auto-build if the library is missing
if not os.path.exists(_link_lib_path):
    _cmake_build_c_library()

# ---------------------------------------------------------------------------
# CUDA / GPU support
#
# torch.cuda.is_available() checks the installed GPU driver at runtime.
# CUDA_HOME / CUDA_PATH are set by the CUDA toolkit installer.
# We check all three so we cover all common setups.
# ---------------------------------------------------------------------------
_has_cuda = (
    torch.cuda.is_available()
    or "CUDA_HOME" in os.environ
    or "CUDA_PATH" in os.environ
)

if _has_cuda:
    enable_gpu = True
else:
    print("CUDA not detected. Building CPU-only PyTorch binding.")
    enable_gpu = False

# ---------------------------------------------------------------------------
# Compiler flags (platform-specific)
# ---------------------------------------------------------------------------
_torch_version = Version(torch.__version__.split("+")[0])

if IS_WINDOWS:
    # MSVC syntax
    if _torch_version >= Version("2.1.0"):
        _std_flag = "/std:c++17"
    elif _torch_version >= Version("1.5.0"):
        _std_flag = "/std:c++14"
    else:
        _std_flag = "/std:c++11"
    extra_compile_args = [_std_flag]
    if enable_gpu:
        extra_compile_args += ["/DWARPRNNT_ENABLE_GPU"]
else:
    # GCC / Clang syntax
    if _torch_version >= Version("2.1.0"):
        _std_flag = "-std=c++17"
    elif _torch_version >= Version("1.5.0"):
        _std_flag = "-std=c++14"
    else:
        _std_flag = "-std=c++11"
    extra_compile_args = ["-fPIC", _std_flag]
    if enable_gpu:
        extra_compile_args += ["-DWARPRNNT_ENABLE_GPU"]

# ---------------------------------------------------------------------------
# Linker flags (platform-specific)
#
# Linux/macOS: embed the library search path so import works without
#              LD_LIBRARY_PATH changes.
# Windows:     no rpath concept; we copy the DLL into the package instead.
# ---------------------------------------------------------------------------
if IS_WINDOWS:
    extra_link_args = []
elif IS_MACOS:
    extra_link_args = [
        "-Wl,-rpath," + warp_rnnt_path,
        "-Wl,-rpath,@loader_path",
    ]
else:
    extra_link_args = ["-Wl,-rpath," + warp_rnnt_path]

# ---------------------------------------------------------------------------
# Custom build_ext
#
# On Windows: copies warprnnt.dll into the Python package directory so that
# os.add_dll_directory() in __init__.py can find it at import time.
# ---------------------------------------------------------------------------
class WarpRNNTBuildExt(_build_ext):
    def run(self):
        super().run()
        if IS_WINDOWS and os.path.exists(_runtime_lib_path):
            pyd_dir = os.path.join(self.build_lib, "warprnnt_pytorch")
            os.makedirs(pyd_dir, exist_ok=True)
            dst = os.path.join(pyd_dir, _runtime_lib)
            print("Bundling {} into package...".format(_runtime_lib))
            shutil.copy2(_runtime_lib_path, dst)

# ---------------------------------------------------------------------------
# Package data: include the bundled DLL on Windows
# ---------------------------------------------------------------------------
_package_data = {}
if IS_WINDOWS:
    _package_data["warprnnt_pytorch"] = ["*.dll"]

# ---------------------------------------------------------------------------
# Setup
# ---------------------------------------------------------------------------
setup(
    name="warprnnt_pytorch",
    version="0.2.0",
    description="PyTorch wrapper for RNN-Transducer (RNNT loss)",
    long_description=open(os.path.join(_this_dir, "README.md")).read(),
    long_description_content_type="text/markdown",
    url="https://github.com/HawkAaron/warp-transducer",
    author="Mingkun Huang",
    author_email="mingkunhuang95@gmail.com",
    packages=find_packages(),
    package_data=_package_data,
    install_requires=["packaging"],
    python_requires=">=3.7",
    ext_modules=[
        CppExtension(
            name="warprnnt_pytorch.warp_rnnt",
            sources=["src/binding.cpp"],
            include_dirs=[os.path.realpath("../include")],
            library_dirs=[warp_rnnt_path],
            libraries=["warprnnt"],
            extra_link_args=extra_link_args,
            extra_compile_args=extra_compile_args,
        ),
    ],
    cmdclass={
        "build_ext": WarpRNNTBuildExt,
    },
)
