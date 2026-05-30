#include <cmath>
#include <random>
#include <tuple>
#include <vector>

#include <iostream>

#include <rnnt.h>

#include "test.h"

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
    std::vector<float> log_probs(acts.size());
    softmax(acts.data(), alphabet_size, B * T * U, log_probs.data(), true);

    float expected_score = 4.495666f;

    std::vector<int> labels = {1, 2};
    std::vector<int> label_lengths = {2};

    std::vector<int> lengths;
    lengths.push_back(T);

    float score;

    rnntOptions options{};
    options.maxT = T;
    options.maxU = U;
    options.loc = RNNT_CPU;
    options.batch_first = true;
    options.blank_label = 0;
    options.num_threads = 1;

    size_t cpu_alloc_bytes;
    throw_on_error(get_workspace_size(T, U, B,
                                      false,
                                      &cpu_alloc_bytes),
                   "Error: get_workspace_size in small_test");

    void* rnnt_cpu_workspace = malloc(cpu_alloc_bytes);

    throw_on_error(compute_rnnt_loss(log_probs.data(),
                                    NULL,
                                    labels.data(), 
                                    label_lengths.data(),
                                    lengths.data(),
                                    alphabet_size,
                                    (int)lengths.size(),
                                    &score,
                                    rnnt_cpu_workspace,
                                    options),
                   "Error: compute_rnnt_loss in small_test");

    free(rnnt_cpu_workspace);
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
    std::vector<float> log_probs(acts.size());
    softmax(acts.data(), alphabet_size, minibatch * T * L, log_probs.data(), true);

    std::vector<float> expected_grads = {-0.432226f, -0.567774f, 0.0f, -0.365650f, 0.0f, -0.202123f,
                                        -0.202123f, 0.0f, 0.0f, -0.165217f, -0.267010f, 0.0f,
                                        -0.394365f, 0.0f, -0.238294f, -0.440418f, 0.0f, 0.0f,
                                        -0.052130f, -0.113087f, 0.0f, -0.183138f, 0.0f, -0.324314f,
                                        -0.764732f, 0.0f, 0.0f, 0.0f, -0.052130f, 0.0f,
                                        0.0f, 0.0f, -0.235268f, -1.0f, 0.0f, 0.0f,
                                        -0.716142f, -0.283858f, 0.0f, -0.183829f, -0.100028f, 0.0f,
                                        -0.100028f, 0.0f, 0.0f, -0.411218f, -0.304924f, 0.0f,
                                        -0.329576f, -0.159178f, 0.0f, -0.259206f, 0.0f, 0.0f,
                                        -0.116076f, -0.295142f, 0.0f, -0.286533f, -0.338184f, 0.0f,
                                        -0.597390f, 0.0f, 0.0f, 0.0f, -0.116076f, 0.0f,
                                        0.0f, -0.402610f, 0.0f, -1.0f, 0.0f, 0.0f};
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
    options.loc = RNNT_CPU;
    options.num_threads = 1;
    options.batch_first = true;

    size_t cpu_alloc_bytes;
    throw_on_error(get_workspace_size(T, L, minibatch,
                                      false,
                                      &cpu_alloc_bytes),
                   "Error: get_workspace_size in options_test");

    void* rnnt_cpu_workspace = malloc(cpu_alloc_bytes);

    throw_on_error(compute_rnnt_loss(log_probs.data(),
                                    grads.data(),
                                    labels.data(), 
                                    label_lengths.data(),
                                    lengths.data(),
                                    alphabet_size,
                                    (int)lengths.size(),
                                    scores.data(),
                                    rnnt_cpu_workspace,
                                    options),
                   "Error: compute_rnnt_loss in options_test");

    free(rnnt_cpu_workspace);

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

    std::vector<float> log_probs(acts.size());
    softmax(acts.data(), alphabet_size, minibatch * T * L, log_probs.data(), true);

    std::vector<int> sizes;
    sizes.push_back(T);

    std::vector<float> grads(acts.size());

    float cost;

    rnntOptions options{};
    options.maxT = T;
    options.maxU = L;
    options.loc = RNNT_CPU;
    options.num_threads = 1;
    options.batch_first = true;

    size_t cpu_alloc_bytes;
    throw_on_error(get_workspace_size(T, L, minibatch,
                                      false,
                                      &cpu_alloc_bytes),
                   "Error: get_workspace_size in inf_test");

    void* rnnt_cpu_workspace = malloc(cpu_alloc_bytes);

    throw_on_error(compute_rnnt_loss(acts.data(),
                                    grads.data(),
                                    labels.data(), 
                                    label_lengths.data(),
                                    sizes.data(),
                                    alphabet_size,
                                    (int)sizes.size(),
                                    &cost,
                                    rnnt_cpu_workspace,
                                    options),
                   "Error: compute_rnnt_loss in inf_test");

    free(rnnt_cpu_workspace);

    bool status = true;
    status &= !std::isinf(cost);

    for (int i = 0; i < alphabet_size * L * T * minibatch; ++i)
        status &= !std::isnan(grads[i]);

    return status;
}

void numeric_grad(std::vector<float>& acts, std::vector<int>& flat_labels, std::vector<int>& label_lengths,
                std::vector<int> sizes, int alphabet_size, int minibatch, 
                void* rnnt_cpu_workspace, rnntOptions& options, std::vector<float>& num_grad) {

    float epsilon = 1e-2f;

    for (int i = 0; i < num_grad.size(); ++i) {

        std::vector<float> costsP1(minibatch);
        std::vector<float> costsP2(minibatch);

        acts[i] += epsilon;
        throw_on_error(compute_rnnt_loss(acts.data(),
                                        NULL,
                                        flat_labels.data(), 
                                        label_lengths.data(),
                                        sizes.data(),
                                        alphabet_size,
                                        minibatch,
                                        costsP1.data(),
                                        rnnt_cpu_workspace,
                                        options),
                       "Error: compute_rnnt_loss (1) in grad_check");

        acts[i] -= 2 * epsilon;
        throw_on_error(compute_rnnt_loss(acts.data(),
                                        NULL,
                                        flat_labels.data(), 
                                        label_lengths.data(),
                                        sizes.data(),
                                        alphabet_size,
                                        minibatch,
                                        costsP2.data(),
                                        rnnt_cpu_workspace,
                                        options),
                       "Error: compute_rnnt_loss (2) in grad_check");

        float costP1 = std::accumulate(costsP1.begin(), costsP1.end(), 0.0f);
        float costP2 = std::accumulate(costsP2.begin(), costsP2.end(), 0.0f);

        acts[i] += epsilon;
        num_grad[i] = (costP1 - costP2) / (2 * epsilon);
    }
}

bool grad_check(int T, int L, int alphabet_size,
                  std::vector<float>& acts,
                  const std::vector<std::vector<int>>& labels,
                  const std::vector<int>& sizes, float tol) {

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
    options.loc = RNNT_CPU;
    options.num_threads = 1;
    options.batch_first = true;

    size_t cpu_alloc_bytes;
    throw_on_error(get_workspace_size(T, L, (int)sizes.size(),
                                      false,
                                      &cpu_alloc_bytes),
                   "Error: get_workspace_size in grad_check");

    void* rnnt_cpu_workspace = malloc(cpu_alloc_bytes);

    throw_on_error(compute_rnnt_loss(acts.data(),
                                    grads.data(),
                                    flat_labels.data(), 
                                    label_lengths.data(),
                                    sizes.data(),
                                    alphabet_size,
                                    minibatch,
                                    costs.data(),
                                    rnnt_cpu_workspace,
                                    options),
                   "Error: compute_rnnt_loss (0) in grad_check");

    float cost = std::accumulate(costs.begin(), costs.end(), 0.0f);

    std::vector<float> num_grad(grads.size());

    //perform 2nd order central differencing
    numeric_grad(acts, flat_labels, label_lengths, sizes,
            alphabet_size, minibatch, rnnt_cpu_workspace, options, num_grad);

    free(rnnt_cpu_workspace);

    float diff = rel_diff(grads, num_grad);

    return diff < tol;
}

bool run_tests() {
    std::vector<std::tuple<int, int, int, int, float>> problem_sizes =
       {std::make_tuple(20, 50, 15, 1, 1e-4f),
        std::make_tuple(5, 10, 5, 65, 1e-4f)
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

    std::cout << "Running CPU tests" << std::endl;

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
