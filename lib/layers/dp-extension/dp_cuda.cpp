#include <torch/extension.h>

#include <vector>

// CUDA forward declarations

std::vector<torch::Tensor> dp_cuda_forward(
    torch::Tensor unary_h,
    torch::Tensor unary_v,
    torch::Tensor pairwise_h,
    torch::Tensor pairwise_v);

std::vector<torch::Tensor> dp_cuda_backward(
    torch::Tensor grad_marginals_h,
    torch::Tensor grad_marginals_v,
    torch::Tensor argmax_h_left,
    torch::Tensor argmax_h_right,
    torch::Tensor argmax_v_top,
    torch::Tensor argmax_v_bottom);

// C++ interface

#define CHECK_CUDA(x) AT_ASSERTM(x.type().is_cuda(), #x " must be a CUDA tensor")
#define CHECK_CONTIGUOUS(x) AT_ASSERTM(x.is_contiguous(), #x " must be contiguous")
#define CHECK_INPUT(x) CHECK_CUDA(x); CHECK_CONTIGUOUS(x)

std::vector<torch::Tensor> dp_forward(
    torch::Tensor unary_h,
    torch::Tensor unary_v,
    torch::Tensor pairwise_h,
    torch::Tensor pairwise_v,
    float gamma /* just a placeholder */) {
  CHECK_INPUT(unary_h);
  CHECK_INPUT(unary_v);
  CHECK_INPUT(pairwise_h);
  CHECK_INPUT(pairwise_v);

  return dp_cuda_forward(unary_h, unary_v, pairwise_h, pairwise_v);
}

std::vector<torch::Tensor> dp_backward(
    torch::Tensor grad_marginals_h,
    torch::Tensor grad_marginals_v,
    torch::Tensor argmax_h_left,
    torch::Tensor argmax_h_right,
    torch::Tensor argmax_v_top,
    torch::Tensor argmax_v_bottom) {
  CHECK_INPUT(grad_marginals_h);
  CHECK_INPUT(grad_marginals_v);
  CHECK_INPUT(argmax_h_left);
  CHECK_INPUT(argmax_h_right);
  CHECK_INPUT(argmax_v_top);
  CHECK_INPUT(argmax_v_bottom);

  return dp_cuda_backward(
      grad_marginals_h,
      grad_marginals_v,
      argmax_h_left,
      argmax_h_right,
      argmax_v_top,
      argmax_v_bottom);
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
  m.def("forward", &dp_forward, "DP forward (CUDA)");
  m.def("backward", &dp_backward, "DP backward (CUDA)");
}
