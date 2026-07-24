// pybind/torch bindings for the fused scatter-add message-passing kernel.
// Exposed to Python as the `pigno_mp` extension, consumed by src/scatter.py.
//
// Provides a CPU fallback so the extension is importable and correct even on a
// machine without CUDA at build time; src/scatter.py also has a pure-PyTorch
// fallback, so this CPU path is mainly for parity testing.

#include <torch/extension.h>

// CUDA entry points (scatter_cuda.cu)
torch::Tensor scatter_add_forward_cuda(torch::Tensor messages, torch::Tensor dst,
                                       int64_t n_nodes);
torch::Tensor scatter_add_backward_cuda(torch::Tensor grad_out, torch::Tensor dst);

static torch::Tensor scatter_add_forward(torch::Tensor messages, torch::Tensor dst,
                                         int64_t n_nodes) {
    if (messages.is_cuda()) return scatter_add_forward_cuda(messages, dst, n_nodes);
    // CPU reference
    auto out = torch::zeros({n_nodes, messages.size(1)}, messages.options());
    out.index_add_(0, dst, messages);
    return out;
}

static torch::Tensor scatter_add_backward(torch::Tensor grad_out, torch::Tensor dst) {
    if (grad_out.is_cuda()) return scatter_add_backward_cuda(grad_out, dst);
    return grad_out.index_select(0, dst);  // grad_msg[e] = grad_out[dst[e]]
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("scatter_add_forward", &scatter_add_forward,
          "PI-GNO fused scatter-add forward (out[dst[e]] += messages[e])");
    m.def("scatter_add_backward", &scatter_add_backward,
          "PI-GNO fused scatter-add backward (grad_msg[e] = grad_out[dst[e]])");
}
