// Fused scatter-add aggregation for PI-GNO message passing.
//
// This is the single perf-critical primitive pushed to CUDA (see project README
// for the Python-vs-csrc rationale): out[dst[e], :] += messages[e, :], run every
// forward/backward of every message-passing layer.
//
// forward:  out[n, f] = sum_{e : dst[e]==n} messages[e, f]
// backward: grad_messages[e, f] = grad_out[dst[e], f]
//
// Forward uses atomicAdd over the destination index (handles arbitrary, possibly
// duplicated dst). Backward is a pure gather (no atomics needed). Both kernels are
// parallelized over (edge, feature) for coalesced access on the feature dim.

#include <torch/extension.h>
#include <cuda.h>
#include <cuda_runtime.h>

template <typename scalar_t>
__global__ void scatter_add_forward_kernel(
        const scalar_t* __restrict__ messages,  // [E, F]
        const int64_t*  __restrict__ dst,        // [E]
        scalar_t*       __restrict__ out,        // [N, F]
        int64_t E, int64_t Fdim) {
    int64_t tid = blockIdx.x * blockDim.x + threadIdx.x;
    int64_t total = E * Fdim;
    if (tid >= total) return;
    int64_t e = tid / Fdim;
    int64_t f = tid % Fdim;
    int64_t n = dst[e];
    atomicAdd(&out[n * Fdim + f], messages[tid]);
}

template <typename scalar_t>
__global__ void scatter_add_backward_kernel(
        const scalar_t* __restrict__ grad_out,   // [N, F]
        const int64_t*  __restrict__ dst,         // [E]
        scalar_t*       __restrict__ grad_msg,    // [E, F]
        int64_t E, int64_t Fdim) {
    int64_t tid = blockIdx.x * blockDim.x + threadIdx.x;
    int64_t total = E * Fdim;
    if (tid >= total) return;
    int64_t e = tid / Fdim;
    int64_t f = tid % Fdim;
    int64_t n = dst[e];
    grad_msg[tid] = grad_out[n * Fdim + f];
}

torch::Tensor scatter_add_forward_cuda(torch::Tensor messages,
                                       torch::Tensor dst, int64_t n_nodes) {
    TORCH_CHECK(messages.is_cuda(), "messages must be CUDA");
    TORCH_CHECK(dst.is_cuda(), "dst must be CUDA");
    messages = messages.contiguous();
    dst = dst.contiguous();
    const int64_t E = messages.size(0);
    const int64_t Fdim = messages.size(1);
    auto out = torch::zeros({n_nodes, Fdim}, messages.options());

    const int64_t total = E * Fdim;
    const int threads = 256;
    const int64_t blocks = (total + threads - 1) / threads;

    AT_DISPATCH_FLOATING_TYPES(messages.scalar_type(), "scatter_add_forward", [&] {
        scatter_add_forward_kernel<scalar_t><<<blocks, threads>>>(
            messages.data_ptr<scalar_t>(), dst.data_ptr<int64_t>(),
            out.data_ptr<scalar_t>(), E, Fdim);
    });
    return out;
}

torch::Tensor scatter_add_backward_cuda(torch::Tensor grad_out,
                                        torch::Tensor dst) {
    TORCH_CHECK(grad_out.is_cuda(), "grad_out must be CUDA");
    grad_out = grad_out.contiguous();
    dst = dst.contiguous();
    const int64_t E = dst.size(0);
    const int64_t Fdim = grad_out.size(1);
    auto grad_msg = torch::empty({E, Fdim}, grad_out.options());

    const int64_t total = E * Fdim;
    const int threads = 256;
    const int64_t blocks = (total + threads - 1) / threads;

    AT_DISPATCH_FLOATING_TYPES(grad_out.scalar_type(), "scatter_add_backward", [&] {
        scatter_add_backward_kernel<scalar_t><<<blocks, threads>>>(
            grad_out.data_ptr<scalar_t>(), dst.data_ptr<int64_t>(),
            grad_msg.data_ptr<scalar_t>(), E, Fdim);
    });
    return grad_msg;
}
