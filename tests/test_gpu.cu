#include <cmath>
#include <random>
#include <tuple>
#include <vector>

#include <iostream>

#include <rnnt.h>

#include "test.h"

template<typename T>
void vector_to_gpu(T*& gpu_space, std::vector<T>& vec, cudaStream_t& stream) {
    cudaMalloc(&gpu_space, vec.size() * sizeof(T));
    cudaMemcpyAsync(gpu_space, vec.data(), vec.size() * sizeof(T), cudaMemcpyHostToDevice, stream);
}

bool small_test() {
    const int B = 1;
    const int alphabet_size = 5;
    const int T = 2;
    const int U = 3;

    std::vector<float> acts = {0.1f, 0.6f, 0.1f, 0.1f, 0.1f, 0.1f,
                                0.1f, 0.6f, 0.1f, 0.1f, 0.1f, 0.1f,
                                0.2f, 0.8f, 0.1f, 0.1f, 0.6f, 0.1f,
                                0.1f, 0.1f, 0.1f, 0.1f, 0.2f, 0.1f,
                                0.1f, 0.7f, 0.1f, 0.2f, 0.1f, 0.1f};
    // std::vector<float> log_probs(acts.size());
    // softmax(acts.data(), alphabet_size, B * T * U, log_probs.data(), true);

    float expected_score = 4.495666f;

    std::vector<int> labels = {1, 2};
    std::vector<int> label_lengths = {2};

    std::vector<int> lengths;
    lengths.push_back(T);

    float score;

    rnntOptions options{};
    options.maxT = T;
    options.maxU = U;
    options.loc = RNNT_GPU;
    options.blank_label = 0;
    cudaStream_t stream;
    cudaStreamCreate(&stream);
    options.stream = stream;
    options.num_threads = 1;

    float* acts_gpu;
    vector_to_gpu(acts_gpu, acts, stream);
    int* label_gpu;
    vector_to_gpu(label_gpu, labels, stream);
    int* label_length_gpu;
    vector_to_gpu(label_length_gpu, label_lengths, stream);
    int* input_length_gpu;
    vector_to_gpu(input_length_gpu, lengths, stream);

    size_t gpu_alloc_bytes;
    throw_on_error(get_workspace_size(T, U, B,
                                      true,
                                      &gpu_alloc_bytes),
                   "Error: get_workspace_size in small_test");

    void* rnnt_gpu_workspace;
    cudaMalloc(&rnnt_gpu_workspace, gpu_alloc_bytes);

    throw_on_error(compute_rnnt_loss(acts_gpu,
                                    NULL,
                                    label_gpu, 
                                    label_length_gpu,
                                    input_length_gpu,
                                    alphabet_size,
                                    (int)lengths.size(),
                                    &score,
                                    rnnt_gpu_workspace,
                                    options),
                   "Error: compute_rnnt_loss in small_test");

    cudaFree(rnnt_gpu_workspace);
    cudaFree(acts_gpu);
    cudaFree(label_gpu);
    cudaFree(label_length_gpu);
    cudaFree(input_length_gpu);

    const float eps = 1e-4f;

    const float lb = expected_score - eps;
    const float ub = expected_score + eps;

    return (score > lb && score < ub);
}

bool options_test() {
    const int alphabet_size = 3;
    const int T = 4;
    const int L = 3;
    const int minibatch = 2;

    std::vector<float> acts = {0.065357f, 0.787530f, 0.081592f, 0.529716f, 0.750675f, 0.754135f,
                                0.609764f, 0.868140f, 0.622532f, 0.668522f, 0.858039f, 0.164539f,
                                0.989780f, 0.944298f, 0.603168f, 0.946783f, 0.666203f, 0.286882f,
                                0.094184f, 0.366674f, 0.736168f, 0.166680f, 0.714154f, 0.399400f,
                                0.535982f, 0.291821f, 0.612642f, 0.324241f, 0.800764f, 0.524106f,
                                0.779195f, 0.183314f, 0.113745f, 0.240222f, 0.339470f, 0.134160f,
                                0.505562f, 0.051597f, 0.640290f, 0.430733f, 0.829473f, 0.177467f,
                                0.320700f, 0.042883f, 0.302803f, 0.675178f, 0.569537f, 0.558474f,
                                0.083132f, 0.060165f, 0.107958f, 0.748615f, 0.943918f, 0.486356f,
                                0.418199f, 0.652408f, 0.024243f, 0.134582f, 0.366342f, 0.295830f,
                                0.923670f, 0.689929f, 0.741898f, 0.250005f, 0.603430f, 0.987289f,
                                0.592606f, 0.884672f, 0.543450f, 0.660770f, 0.377128f, 0.358021f};
    // std::vector<float> log_probs(acts.size());
    // softmax(acts.data(), alphabet_size, minibatch * T * L, log_probs.data(), true);

    std::vector<float> expected_grads = {-0.186844f, -0.062555f, 0.249399f, -0.203377f, 0.202399f, 0.000977f,
                                        -0.141016f, 0.079123f, 0.061893f, -0.011552f, -0.081280f, 0.092832f,
                                        -0.154257f, 0.229433f, -0.075176f, -0.246593f, 0.146405f, 0.100188f,
                                        -0.012918f, -0.061593f, 0.074512f, -0.055986f, 0.219831f, -0.163845f,
                                        -0.497627f, 0.209240f, 0.288387f, 0.013605f, -0.030220f, 0.016615f,
                                        0.113925f, 0.062781f, -0.176706f, -0.667078f, 0.367659f, 0.299419f,
                                        -0.356344f, -0.055347f, 0.411691f, -0.096922f, 0.029459f, 0.067463f,
                                        -0.063518f, 0.027654f, 0.035863f, -0.154499f, -0.073942f, 0.228441f,
                                        -0.166790f, -0.000088f, 0.166878f, -0.172370f, 0.105565f, 0.066804f,
                                        0.023875f, -0.118256f, 0.094381f, -0.104707f, -0.108934f, 0.213642f,
                                        -0.369844f, 0.180118f, 0.189726f, 0.025714f, -0.079462f, 0.053748f,
                                        0.122328f, -0.238789f, 0.116460f, -0.598687f, 0.302203f, 0.296484f};

    // Calculate the expected scores analytically
    std::vector<double> expected_scores(2);
    expected_scores[0] = 4.2806528590890736;
    expected_scores[1] = 3.9384369822503591;

    std::vector<int> labels = {1, 2, 1, 1};

    std::vector<int> label_lengths = {2, 2};

    std::vector<int> lengths = {4, 4};

    std::vector<float> grads(acts.size());
    std::vector<float> scores(2);

    rnntOptions options{};
    options.maxT = T;
    options.maxU = L;
    options.loc = RNNT_GPU;
    cudaStream_t stream;
    cudaStreamCreate(&stream);
    options.stream = stream;
    options.num_threads = 1;

    float* acts_gpu;
    vector_to_gpu(acts_gpu, acts, stream);
    float* grads_gpu;
    cudaMalloc(&grads_gpu, grads.size() * sizeof(float));
    int* label_gpu;
    vector_to_gpu(label_gpu, labels, stream);
    int* label_length_gpu;
    vector_to_gpu(label_length_gpu, label_lengths, stream);
    int* input_length_gpu;
    vector_to_gpu(input_length_gpu, lengths, stream);

    size_t gpu_alloc_bytes;
    throw_on_error(get_workspace_size(T, L, minibatch,
                                      true,
                                      &gpu_alloc_bytes),
                   "Error: get_workspace_size in options_test");

    void* rnnt_gpu_workspace;
    cudaMalloc(&rnnt_gpu_workspace, gpu_alloc_bytes);

    throw_on_error(compute_rnnt_loss(acts_gpu,
                                    grads_gpu,
                                    label_gpu,
                                    label_length_gpu,
                                    input_length_gpu,
                                    alphabet_size,
                                    (int)lengths.size(),
                                    scores.data(),
                                    rnnt_gpu_workspace,
                                    options),
                   "Error: compute_rnnt_loss in small_test");

    cudaMemcpyAsync(grads.data(), grads_gpu, grads.size() * sizeof(float), cudaMemcpyDeviceToHost, stream);

    cudaFree(rnnt_gpu_workspace);
    cudaFree(acts_gpu);
    cudaFree(grads_gpu);
    cudaFree(label_gpu);
    cudaFree(label_length_gpu);
    cudaFree(input_length_gpu);

    const double eps = 1e-4;

    bool result = true;
    // activations gradient check
    for (int i = 0; i < grads.size(); i++) {
        const double lb = expected_grads[i] - eps;
        const double ub = expected_grads[i] + eps;
        if (!(grads[i] > lb && grads[i] < ub)) {
            std::cerr << "grad mismatch in options_test"
                      << " expected grad: " << expected_grads[i]
                      << " calculated score: " << grads[i]
                      << " !(" << lb << " < " << grads[i]
                      << " < " << ub << ")" << std::endl;
            result = false;
        }
    }

    for (int i = 0; i < 2; i++) {
        const double lb = expected_scores[i] - eps;
        const double ub = expected_scores[i] + eps;
        if (!(scores[i] > lb && scores[i] < ub)) {
            std::cerr << "score mismatch in options_test"
                      << " expected score: " << expected_scores[i]
                      << " calculated score: " << scores[i]
                      << " !(" << lb << " < " << scores[i]
                      << " < " << ub << ")" << std::endl;
            result = false;
        }
    }
    return result;
}

bool inf_test() {
    const int alphabet_size = 15;
    const int T = 50;
    const int L = 10;
    const int minibatch = 1;

    std::vector<int> labels = genLabels(alphabet_size, L-1);
    labels[0] = 2;
    std::vector<int> label_lengths = {L-1};

    std::vector<float> acts(alphabet_size * T * L * minibatch);
    genActs(acts);

    // std::vector<float> log_probs(acts.size());
    // softmax(acts.data(), alphabet_size, minibatch * T * L, log_probs.data(), true);

    std::vector<int> sizes;
    sizes.push_back(T);

    std::vector<float> grads(acts.size());

    float cost;

    rnntOptions options{};
    options.maxT = T;
    options.maxU = L;
    options.loc = RNNT_GPU;
    cudaStream_t stream;
    cudaStreamCreate(&stream);
    options.stream = stream;
    options.num_threads = 1;

    float* acts_gpu;
    vector_to_gpu(acts_gpu, acts, stream);
    float* grads_gpu;
    cudaMalloc(&grads_gpu, grads.size() * sizeof(float));
    int* label_gpu;
    vector_to_gpu(label_gpu, labels, stream);
    int* label_length_gpu;
    vector_to_gpu(label_length_gpu, label_lengths, stream);
    int* input_length_gpu;
    vector_to_gpu(input_length_gpu, sizes, stream);

    size_t gpu_alloc_bytes;
    throw_on_error(get_workspace_size(T, L, minibatch,
                                      true,
                                      &gpu_alloc_bytes),
                   "Error: get_workspace_size in inf_test");

    void* rnnt_gpu_workspace;
    cudaMalloc(&rnnt_gpu_workspace, gpu_alloc_bytes);

    throw_on_error(compute_rnnt_loss(acts_gpu,
                                    grads_gpu,
                                    label_gpu,
                                    label_length_gpu,
                                    input_length_gpu,
                                    alphabet_size,
                                    (int)sizes.size(),
                                    &cost,
                                    rnnt_gpu_workspace,
                                    options),
                   "Error: compute_rnnt_loss in small_test");

    cudaMemcpyAsync(grads.data(), grads_gpu, grads.size() * sizeof(float), cudaMemcpyDeviceToHost, stream);

    cudaFree(rnnt_gpu_workspace);
    cudaFree(acts_gpu);
    cudaFree(grads_gpu);
    cudaFree(label_gpu);
    cudaFree(label_length_gpu);
    cudaFree(input_length_gpu);

    bool status = true;
    status &= !std::isinf(cost);

    for (int i = 0; i < alphabet_size * L * T * minibatch; ++i)
        status &= !std::isnan(grads[i]);

    return status;
}

void numeric_grad(float* acts, int* flat_labels, int* label_lengths,
                int* sizes, int alphabet_size, int minibatch, 
                void* rnnt_gpu_workspace, rnntOptions& options, std::vector<float>& num_grad) {

    float epsilon = 1e-2f;
    float act;

    for (int i = 0; i < num_grad.size(); ++i) {

        std::vector<float> costsP1(minibatch);
        std::vector<float> costsP2(minibatch);

        cudaMemcpy(&act, &acts[i], sizeof(float), cudaMemcpyDeviceToHost);
        act += epsilon;
        cudaMemcpy(&acts[i], &act, sizeof(float), cudaMemcpyHostToDevice);
        throw_on_error(compute_rnnt_loss(acts,
                                        NULL,
                                        flat_labels, 
                                        label_lengths,
                                        sizes,
                                        alphabet_size,
                                        minibatch,
                                        costsP1.data(),
                                        rnnt_gpu_workspace,
                                        options),
                       "Error: compute_rnnt_loss (1) in grad_check");

        cudaMemcpy(&act, &acts[i], sizeof(float), cudaMemcpyDeviceToHost);
        act -= 2 * epsilon;
        cudaMemcpy(&acts[i], &act, sizeof(float), cudaMemcpyHostToDevice);
        throw_on_error(compute_rnnt_loss(acts,
                                        NULL,
                                        flat_labels, 
                                        label_lengths,
                                        sizes,
                                        alphabet_size,
                                        minibatch,
                                        costsP2.data(),
                                        rnnt_gpu_workspace,
                                        options),
                       "Error: compute_rnnt_loss (2) in grad_check");

        float costP1 = std::accumulate(costsP1.begin(), costsP1.end(), 0.0f);
        float costP2 = std::accumulate(costsP2.begin(), costsP2.end(), 0.0f);

        cudaMemcpy(&act, &acts[i], sizeof(float), cudaMemcpyDeviceToHost);
        act += epsilon;
        cudaMemcpy(&acts[i], &act, sizeof(float), cudaMemcpyHostToDevice);
        num_grad[i] = (costP1 - costP2) / (2 * epsilon);
    }
}

bool grad_check(int T, int L, int alphabet_size,
                  std::vector<float>& acts,
                  const std::vector<std::vector<int>>& labels,
                  std::vector<int>& sizes, float tol) {

    const int minibatch = (int)labels.size();

    std::vector<int> flat_labels;
    std::vector<int> label_lengths;
    for (const auto& l : labels) {
        flat_labels.insert(flat_labels.end(), l.begin(), l.end());
        label_lengths.push_back((int)l.size());
    }

    std::vector<float> costs(minibatch);

    std::vector<float> grads(acts.size());

    rnntOptions options{};
    options.maxT = T;
    options.maxU = L;
    options.loc = RNNT_GPU;
    cudaStream_t stream;
    cudaStreamCreate(&stream);
    options.stream = stream;
    options.num_threads = 1;

    float* acts_gpu;
    vector_to_gpu(acts_gpu, acts, stream);
    float* grads_gpu;
    cudaMalloc(&grads_gpu, grads.size() * sizeof(float));
    int* label_gpu;
    vector_to_gpu(label_gpu, flat_labels, stream);
    int* label_length_gpu;
    vector_to_gpu(label_length_gpu, label_lengths, stream);
    int* input_length_gpu;
    vector_to_gpu(input_length_gpu, sizes, stream);
    options.num_threads = 1;

    size_t gpu_alloc_bytes;
    throw_on_error(get_workspace_size(T, L, (int)sizes.size(),
                                      true,
                                      &gpu_alloc_bytes),
                   "Error: get_workspace_size in grad_check");

    void* rnnt_gpu_workspace;
    cudaMalloc(&rnnt_gpu_workspace, gpu_alloc_bytes);

    throw_on_error(compute_rnnt_loss(acts_gpu,
                                    grads_gpu,
                                    label_gpu,
                                    label_length_gpu,
                                    input_length_gpu,
                                    alphabet_size,
                                    (int)sizes.size(),
                                    costs.data(),
                                    rnnt_gpu_workspace,
                                    options),
                   "Error: compute_rnnt_loss (0) in grad_check");

    float cost = std::accumulate(costs.begin(), costs.end(), 0.0f);

    cudaMemcpyAsync(grads.data(), grads_gpu, grads.size() * sizeof(float), cudaMemcpyDeviceToHost, stream);

    std::vector<float> num_grad(grads.size());

    //perform 2nd order central differencing
    numeric_grad(acts_gpu, label_gpu, label_length_gpu, input_length_gpu,
            alphabet_size, minibatch, rnnt_gpu_workspace, options, num_grad);

    cudaFree(acts_gpu);
    cudaFree(rnnt_gpu_workspace);
    cudaFree(grads_gpu);
    cudaFree(label_gpu);
    cudaFree(label_length_gpu);
    cudaFree(input_length_gpu);

    float diff = rel_diff(grads, num_grad);

    return diff < tol;
}

bool run_tests() {
    std::vector<std::tuple<int, int, int, int, float>> problem_sizes =
       {std::make_tuple(20, 50, 15, 1, 1e-2f),
        std::make_tuple(5, 10, 5, 65, 1e-2f)
       };

    std::mt19937 gen(2);

    bool status = true;
    for (auto problem : problem_sizes) {
        int alphabet_size, T, L, minibatch;
        float tol;
        std::tie(alphabet_size, T, L, minibatch, tol) = problem;

        std::vector<float> acts(alphabet_size * T * L * minibatch);
        genActs(acts);

        std::vector<float> log_probs(acts.size());
        softmax(acts.data(), alphabet_size, minibatch * T * L, log_probs.data(), true);

        std::vector<std::vector<int>> labels;
        std::vector<int> sizes;
        for (int mb = 0; mb < minibatch; ++mb) {
            int actual_length = L - 1;
            labels.push_back(genLabels(alphabet_size, actual_length));
            sizes.push_back(T);
        }

        status &= grad_check(T, L, alphabet_size, acts, labels, sizes, tol);
    }

    return status;
}

int main(void) {
    if (get_warprnnt_version() != 1) {
        std::cerr << "Invalid Warp-transducer version." << std::endl;
        return 1;
    }

    std::cout << "Running gpu tests" << std::endl;

    bool status = true;
    status &= small_test();
    printf("finish small_test %d\n", status);
    status &= options_test();
    printf("finish options_test %d\n", status);
    status &= inf_test();
    printf("finish inf_test %d\n", status);
    status &= run_tests();
    printf("finished %d\n", status);

    if (status) {
        std::cout << "Tests pass" << std::endl;
        return 0;
    } else {
        std::cout << "Some or all tests fail" << std::endl;
        return 1;
    }
}