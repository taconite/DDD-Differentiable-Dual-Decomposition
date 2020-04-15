#include <torch/extension.h>

#include <cuda.h>
#include <cuda_runtime.h>

#include <vector>

// A thread block contains N_GROUPS * STATE_SIZE threads, which should be less than 1024
#define N_GROUPS 32
#define STATE_SIZE 21

template <typename scalar_t>
__global__ void dp_cuda_forward_kernel(
    const torch::PackedTensorAccessor<scalar_t,4,torch::RestrictPtrTraits,size_t> unary_h,
    const torch::PackedTensorAccessor<scalar_t,4,torch::RestrictPtrTraits,size_t> unary_v,
    const torch::PackedTensorAccessor<scalar_t,5,torch::RestrictPtrTraits,size_t> pairwise_h,
    const torch::PackedTensorAccessor<scalar_t,5,torch::RestrictPtrTraits,size_t> pairwise_v,
    const int width,
    const int height,
    const int state_size,
    torch::PackedTensorAccessor<scalar_t,4,torch::RestrictPtrTraits,size_t> marginals_h_left,
    torch::PackedTensorAccessor<scalar_t,4,torch::RestrictPtrTraits,size_t> marginals_h_right,
    torch::PackedTensorAccessor<scalar_t,4,torch::RestrictPtrTraits,size_t> marginals_v_top,
    torch::PackedTensorAccessor<scalar_t,4,torch::RestrictPtrTraits,size_t> marginals_v_bottom,
    torch::PackedTensorAccessor<int32_t,4,torch::RestrictPtrTraits,size_t> argmax_h_left,
    torch::PackedTensorAccessor<int32_t,4,torch::RestrictPtrTraits,size_t> argmax_h_right,
    torch::PackedTensorAccessor<int32_t,4,torch::RestrictPtrTraits,size_t> argmax_v_top,
    torch::PackedTensorAccessor<int32_t,4,torch::RestrictPtrTraits,size_t> argmax_v_bottom) {
  //batch index
  const int n = blockIdx.y;
  // instance index
  const int inst = blockIdx.x * blockDim.x + threadIdx.x;   // inst id is in [0, width * 2 + height * 2)
  const int k = threadIdx.x;    // k is the index of the selected row/column in current group
  const int s = threadIdx.y;    // s is the index the selected state

  __shared__ scalar_t max_array[N_GROUPS][STATE_SIZE];    // shared memory for max marginals

  int r, c;

  const int len = max(height, width);

  if (inst < height) {
      // case 1: horizontal DP left to right
      r = inst;
      // handle first column
      max_array[k][s] = unary_h[n][s][r][0];
  } else if (inst >= height && inst < height * 2) {
      // case 2: horizontal DP right to left
      r = inst - height;
      // handle last column
      max_array[k][s] = unary_h[n][s][r][width-1];
  } else if (inst >= height * 2 && inst < height * 2 + width) {
      // case 3: vertical DP top to bottom
      c = inst - 2 * height;
      // handle first row
      max_array[k][s] = unary_v[n][s][0][c];
  } else if (inst >= height * 2 + width && inst < (width + height) * 2) {
      // case 4: vertical DP bottom to top
      c = inst - 2 * height - width;
      // handle last row
      max_array[k][s] = unary_v[n][s][height-1][c];
  }


  if (inst < height) {
      scalar_t val = max_array[k][s];
      marginals_h_left[n][s][r][0] = val;
  } else if (inst >= height && inst < height * 2) {
      scalar_t val = max_array[k][s];
      marginals_h_right[n][s][r][width-1] = val;
  } else if (inst >= height * 2 && inst < height * 2 + width) {
      scalar_t val = max_array[k][s];
      marginals_v_top[n][s][0][c] = val;
  } else if (inst >= height * 2 + width && inst < (width + height) * 2) {
      scalar_t val = max_array[k][s];
      marginals_v_bottom[n][s][height-1][c] = val;
  }

  __syncthreads();  // synchronize threads for first/last row/column pixel

  for (int i = 1; i < len; ++i) {
      int max_idx = 0;
      scalar_t max_margin = 0;
      if (inst < height && i < width) {
          max_margin = max_array[k][0] + pairwise_h[n][0][s][r][i-1]; 
          for (int s0 = 1; s0 < state_size; ++s0) {
              scalar_t margin = max_array[k][s0] + pairwise_h[n][s0][s][r][i-1];
              if (margin > max_margin) {
                  max_margin = margin;
                  max_idx = s0;
              }
          }
      } else if (inst >= height && inst < height * 2 && i < width) {
          max_margin = max_array[k][0] + pairwise_h[n][s][0][r][width-1-i]; 
          for (int s1 = 1; s1 < state_size; ++s1) {
              scalar_t margin = max_array[k][s1] + pairwise_h[n][s][s1][r][width-1-i];
              if (margin > max_margin) {
                  max_margin = margin;
                  max_idx = s1;
              }
          }
      } else if (inst >= height * 2 && inst < height * 2 + width && i < height) {
          max_margin = max_array[k][0] + pairwise_v[n][0][s][i-1][c]; 
          for (int s0 = 1; s0 < state_size; ++s0) {
              scalar_t margin = max_array[k][s0] + pairwise_v[n][s0][s][i-1][c];
              if (margin > max_margin) {
                  max_margin = margin;
                  max_idx = s0;
              }
          }
      } else if (inst >= height * 2 + width && inst < (width + height) * 2 && i < height) {
          max_margin = max_array[k][0] + pairwise_v[n][s][0][height-1-i][c]; 
          for (int s1 = 1; s1 < state_size; ++s1) {
              scalar_t margin = max_array[k][s1] + pairwise_v[n][s][s1][height-1-i][c];
              if (margin > max_margin) {
                  max_margin = margin;
                  max_idx = s1;
              }
          }
      }

      __syncthreads();  // synchronize threads for current pixel

      if (inst < height && i < width) {
          max_array[k][s] = unary_h[n][s][r][i] + max_margin;
      } else if (inst >= height && inst < height * 2 && i < width) {
          max_array[k][s] = unary_h[n][s][r][width-1-i] + max_margin;
      } else if (inst >= height * 2 && inst < height * 2 + width && i < height) {
          max_array[k][s] = unary_v[n][s][i][c] + max_margin;
      } else if (inst >= height * 2 + width && inst < (width + height) * 2 && i < height) {
          max_array[k][s] = unary_v[n][s][height-1-i][c] + max_margin;
      }

      __syncthreads();  // synchronize threads for current pixel

      if (inst < height && i < width) {
          scalar_t val = max_array[k][s];
          marginals_h_left[n][s][r][i] = val;
          argmax_h_left[n][s][r][i] = max_idx;
      } else if (inst >= height && inst < height * 2 && i < width) {
          scalar_t val = max_array[k][s];
          marginals_h_right[n][s][r][width-1-i] = val;
          argmax_h_right[n][s][r][width-1-i] = max_idx;
      } else if (inst >= height * 2 && inst < height * 2 + width && i < height) {
          scalar_t val = max_array[k][s];
          marginals_v_top[n][s][i][c] = val;
          argmax_v_top[n][s][i][c] = max_idx;
      } else if (inst >= height * 2 + width && inst < (width + height) * 2 && i < height) {
          scalar_t val = max_array[k][s];
          marginals_v_bottom[n][s][height-1-i][c] = val;
          argmax_v_bottom[n][s][height-1-i][c] = max_idx;
      }
  }
}

std::vector<torch::Tensor> dp_cuda_forward(
    torch::Tensor unary_h,
    torch::Tensor unary_v,
    torch::Tensor pairwise_h,
    torch::Tensor pairwise_v) {

  const auto batch_size = unary_h.size(0);
  const auto state_size = unary_h.size(1);
  const auto height = unary_h.size(2);
  const auto width = unary_h.size(3);

  auto marginals_h_left = torch::zeros_like(unary_h);
  auto marginals_h_right = torch::zeros_like(unary_h);
  auto marginals_v_top = torch::zeros_like(unary_v);
  auto marginals_v_bottom = torch::zeros_like(unary_v);

  auto options = torch::TensorOptions()
    .dtype(torch::kInt32)
    .layout(torch::kStrided)
    .device(torch::kCUDA, unary_h.get_device())
    .requires_grad(false);

  auto argmax_h_left = torch::zeros({batch_size, state_size, height, width}, options);
  auto argmax_h_right = torch::zeros({batch_size, state_size, height, width}, options);
  auto argmax_v_top = torch::zeros({batch_size, state_size, height, width}, options);
  auto argmax_v_bottom = torch::zeros({batch_size, state_size, height, width}, options);

  const dim3 threads(N_GROUPS, STATE_SIZE);
  const dim3 blocks((height * 2 + width * 2 + N_GROUPS - 1) / N_GROUPS, batch_size);

  AT_DISPATCH_FLOATING_TYPES(unary_h.type(), "dp_forward_cuda", ([&] {
    dp_cuda_forward_kernel<scalar_t><<<blocks, threads>>>(
        unary_h.packed_accessor<scalar_t,4,torch::RestrictPtrTraits,size_t>(),
        unary_v.packed_accessor<scalar_t,4,torch::RestrictPtrTraits,size_t>(),
        pairwise_h.packed_accessor<scalar_t,5,torch::RestrictPtrTraits,size_t>(),
        pairwise_v.packed_accessor<scalar_t,5,torch::RestrictPtrTraits,size_t>(),
        width, height, state_size,
        marginals_h_left.packed_accessor<scalar_t,4,torch::RestrictPtrTraits,size_t>(),
        marginals_h_right.packed_accessor<scalar_t,4,torch::RestrictPtrTraits,size_t>(),
        marginals_v_top.packed_accessor<scalar_t,4,torch::RestrictPtrTraits,size_t>(),
        marginals_v_bottom.packed_accessor<scalar_t,4,torch::RestrictPtrTraits,size_t>(),
        argmax_h_left.packed_accessor<int32_t,4,torch::RestrictPtrTraits,size_t>(),
        argmax_h_right.packed_accessor<int32_t,4,torch::RestrictPtrTraits,size_t>(),
        argmax_v_top.packed_accessor<int32_t,4,torch::RestrictPtrTraits,size_t>(),
        argmax_v_bottom.packed_accessor<int32_t,4,torch::RestrictPtrTraits,size_t>());
  }));

  auto marginals_h = marginals_h_left + marginals_h_right - unary_h;
  auto marginals_v = marginals_v_top + marginals_v_bottom - unary_v;

  return {marginals_h, marginals_v, argmax_h_left, argmax_h_right,
          argmax_v_top, argmax_v_bottom};
}

template <typename scalar_t>
__global__ void dp_cuda_backward_kernel(
    const torch::PackedTensorAccessor<scalar_t,4,torch::RestrictPtrTraits,size_t> grad_marginals_h,
    const torch::PackedTensorAccessor<scalar_t,4,torch::RestrictPtrTraits,size_t> grad_marginals_v,
    const torch::PackedTensorAccessor<int32_t,4,torch::RestrictPtrTraits,size_t> argmax_h_left,
    const torch::PackedTensorAccessor<int32_t,4,torch::RestrictPtrTraits,size_t> argmax_h_right,
    const torch::PackedTensorAccessor<int32_t,4,torch::RestrictPtrTraits,size_t> argmax_v_top,
    const torch::PackedTensorAccessor<int32_t,4,torch::RestrictPtrTraits,size_t> argmax_v_bottom,
    const int width,
    const int height,
    const int state_size,
    torch::PackedTensorAccessor<scalar_t,4,torch::RestrictPtrTraits,size_t> d_unary_h_left,
    torch::PackedTensorAccessor<scalar_t,4,torch::RestrictPtrTraits,size_t> d_unary_h_right,
    torch::PackedTensorAccessor<scalar_t,4,torch::RestrictPtrTraits,size_t> d_unary_v_top,
    torch::PackedTensorAccessor<scalar_t,4,torch::RestrictPtrTraits,size_t> d_unary_v_bottom,
    torch::PackedTensorAccessor<scalar_t,5,torch::RestrictPtrTraits,size_t> d_pairwise_h,
    torch::PackedTensorAccessor<scalar_t,5,torch::RestrictPtrTraits,size_t> d_pairwise_v) {
  //batch index
  const int n = blockIdx.y;
  // instance index
  const int inst = blockIdx.x * blockDim.x + threadIdx.x;   // inst id is in [0, width * 2 + height * 2)
  const int k = threadIdx.x;    // k is the index of the selected row/column in current group
  const int s = threadIdx.y;    // s is the index the selected state

  __shared__ scalar_t d_array[N_GROUPS][STATE_SIZE];    // shared memory for accumulated gradients

  int r, c;

  const int len = max(height, width);

  if (inst < height) {
      // case 1: horizontal DP left to right
      r = inst;
      // handle first column
      d_array[k][s] = grad_marginals_h[n][s][r][0];
  } else if (inst >= height && inst < height * 2) {
      // case 2: horizontal DP right to left
      r = inst - height;
      // handle last column
      d_array[k][s] = grad_marginals_h[n][s][r][width-1];
  } else if (inst >= height * 2 && inst < height * 2 + width) {
      // case 3: vertical DP top to bottom
      c = inst - 2 * height;
      // handle first row
      d_array[k][s] = grad_marginals_v[n][s][0][c];
  } else if (inst >= height * 2 + width && inst < (width + height) * 2) {
      // case 4: vertical DP bottom to top
      c = inst - 2 * height - width;
      // handle last row
      d_array[k][s] = grad_marginals_v[n][s][height-1][c];
  }


  if (inst < height) {
      scalar_t val = d_array[k][s];
      d_unary_h_left[n][s][r][0] = val;
  } else if (inst >= height && inst < height * 2) {
      scalar_t val = d_array[k][s];
      d_unary_h_right[n][s][r][width-1] = val;
  } else if (inst >= height * 2 && inst < height * 2 + width) {
      scalar_t val = d_array[k][s];
      d_unary_v_top[n][s][0][c] = val;
  } else if (inst >= height * 2 + width && inst < (width + height) * 2) {
      scalar_t val = d_array[k][s];
      d_unary_v_bottom[n][s][height-1][c] = val;
  }

  __syncthreads();  // synchronize threads for first/last row/column pixel

  for (int i = 1; i < len; ++i) {
      int bt_idx;
      scalar_t grad_u;
      if (inst < height && i < width) {
          grad_u = grad_marginals_h[n][s][r][i];
          for (int s0 = 0; s0 < state_size; ++s0) {
              bt_idx = argmax_h_right[n][s0][r][i-1];
              if (bt_idx == s) {
                  grad_u += d_array[k][s0];
                  atomicAdd((float *)&d_pairwise_h[n][s0][s][r][i-1], (float)d_array[k][s0]);
              }
          }
      } else if (inst >= height && inst < height * 2 && i < width) {
          grad_u = grad_marginals_h[n][s][r][width-1-i];
          for (int s1 = 0; s1 < state_size; ++s1) {
              bt_idx = argmax_h_left[n][s1][r][width-i];
              if (bt_idx == s) {
                  grad_u += d_array[k][s1];
                  atomicAdd((float *)&d_pairwise_h[n][s][s1][r][width-1-i], (float)d_array[k][s1]);
              }
          }
      } else if (inst >= height * 2 && inst < height * 2 + width && i < height) {
          grad_u = grad_marginals_v[n][s][i][c];
          for (int s0 = 0; s0 < state_size; ++s0) {
              bt_idx = argmax_v_bottom[n][s0][i-1][c];
              if (bt_idx == s) {
                  grad_u += d_array[k][s0];
                  atomicAdd((float *)&d_pairwise_v[n][s0][s][i-1][c], (float)d_array[k][s0]);
              }
          }
      } else if (inst >= height * 2 + width && inst < (width + height) * 2 && i < height) {
          grad_u = grad_marginals_v[n][s][height-1-i][c];
          for (int s1 = 0; s1 < state_size; ++s1) {
              bt_idx = argmax_v_top[n][s1][height-i][c];
              if (bt_idx == s) {
                  grad_u += d_array[k][s1];
                  atomicAdd((float *)&d_pairwise_v[n][s][s1][height-1-i][c], (float)d_array[k][s1]);
              }
          }
      }

      __syncthreads();  // synchronize threads for current pixel

      if (inst < height && i < width) {
          d_array[k][s] = grad_u;
      } else if (inst >= height && inst < height * 2 && i < width) {
          d_array[k][s] = grad_u;
      } else if (inst >= height * 2 && inst < height * 2 + width && i < height) {
          d_array[k][s] = grad_u;
      } else if (inst >= height * 2 + width && inst < (width + height) * 2 && i < height) {
          d_array[k][s] = grad_u;
      }
          
      __syncthreads();  // synchronize threads for current pixel

      if (inst < height && i < width) {
          d_unary_h_left[n][s][r][i] = grad_u;
      } else if (inst >= height && inst < height * 2 && i < width) {
          d_unary_h_right[n][s][r][width-1-i] = grad_u;
      } else if (inst >= height * 2 && inst < height * 2 + width && i < height) {
          d_unary_v_top[n][s][i][c] = grad_u;
      } else if (inst >= height * 2 + width && inst < (width + height) * 2 && i < height) {
          d_unary_v_bottom[n][s][height-1-i][c] = grad_u;
      }
  }
}

std::vector<torch::Tensor> dp_cuda_backward(
    torch::Tensor grad_marginals_h,
    torch::Tensor grad_marginals_v,
    torch::Tensor argmax_h_left,
    torch::Tensor argmax_h_right,
    torch::Tensor argmax_v_top,
    torch::Tensor argmax_v_bottom) {

  const auto batch_size = grad_marginals_h.size(0);
  const auto state_size = grad_marginals_h.size(1);
  const auto height = grad_marginals_h.size(2);
  const auto width = grad_marginals_h.size(3);

  auto d_unary_h_left = torch::zeros_like(grad_marginals_h);
  auto d_unary_h_right = torch::zeros_like(grad_marginals_h);
  auto d_unary_v_top = torch::zeros_like(grad_marginals_v);
  auto d_unary_v_bottom = torch::zeros_like(grad_marginals_v);

  auto options = torch::TensorOptions()
    .dtype(torch::kFloat32)
    .layout(torch::kStrided)
    .device(torch::kCUDA, grad_marginals_h.get_device())
    .requires_grad(false);

  auto d_pairwise_h = torch::zeros({batch_size, state_size, state_size, height, width}, options);
  auto d_pairwise_v = torch::zeros({batch_size, state_size, state_size, height, width}, options);

  const dim3 threads(N_GROUPS, STATE_SIZE);
  const dim3 blocks((height * 2 + width * 2 + N_GROUPS - 1) / N_GROUPS, batch_size);

  AT_DISPATCH_FLOATING_TYPES(grad_marginals_h.type(), "dp_backward_cuda", ([&] {
    dp_cuda_backward_kernel<scalar_t><<<blocks, threads>>>(
        grad_marginals_h.packed_accessor<scalar_t,4,torch::RestrictPtrTraits,size_t>(),
        grad_marginals_v.packed_accessor<scalar_t,4,torch::RestrictPtrTraits,size_t>(),
        argmax_h_left.packed_accessor<int32_t,4,torch::RestrictPtrTraits,size_t>(),
        argmax_h_right.packed_accessor<int32_t,4,torch::RestrictPtrTraits,size_t>(),
        argmax_v_top.packed_accessor<int32_t,4,torch::RestrictPtrTraits,size_t>(),
        argmax_v_bottom.packed_accessor<int32_t,4,torch::RestrictPtrTraits,size_t>(),
        width, height, state_size,
        d_unary_h_left.packed_accessor<scalar_t,4,torch::RestrictPtrTraits,size_t>(),
        d_unary_h_right.packed_accessor<scalar_t,4,torch::RestrictPtrTraits,size_t>(),
        d_unary_v_top.packed_accessor<scalar_t,4,torch::RestrictPtrTraits,size_t>(),
        d_unary_v_bottom.packed_accessor<scalar_t,4,torch::RestrictPtrTraits,size_t>(),
        d_pairwise_h.packed_accessor<scalar_t,5,torch::RestrictPtrTraits,size_t>(),
        d_pairwise_v.packed_accessor<scalar_t,5,torch::RestrictPtrTraits,size_t>());
  }));

  auto d_unary_h = d_unary_h_left + d_unary_h_right - grad_marginals_h;
  auto d_unary_v = d_unary_v_top + d_unary_v_bottom - grad_marginals_v;

  return {d_unary_h, d_unary_v, d_pairwise_h, d_pairwise_v};
}
