/*
 * Copyright (c) 2020, NVIDIA CORPORATION.
 *
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 *     http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 */

#pragma once

#include <functional>
#include <vector>
#include "HugeCTR/include/general_buffer.hpp"
#include "HugeCTR/include/layer.hpp"
#include "cublas_v2.h"

namespace HugeCTR {
/**
 * @brief
 * This class implements the fully connected layer.
 */
class FullyConnectedLayer : public Layer {
 private:
  const cublasHandle_t cublas_handle_;
  const bool use_mixed_precision_{false};

 public:
  /**
   * forward pass
   */
  void fprop(cudaStream_t stream) final;
  /**
   * backward pass
   */
  void bprop(cudaStream_t stream) final;
  /**
   * This is the constructor of the FullyConnectedLayer.
   * It will check whether the format combination of all tensors is supported or not.
   * Only two kinds of tensor formats are supported:
   * (1) weight, input, output, wgrad are all in row-major.
   * (2) weight, input, output, wgrad are all in column-major.
   * @param weight_buff: stores the weight tensor
   * @param wgrad_buff: stores the gradient values of the weight calculated in backward pass
   * @param in_tensor: stores the input tensor
   * @param out_tensor: stores the output tensor
   * @param weight_format: specifies the format of the weight tensor, either HW (row major) or WH
   * (col-major)
   */
  FullyConnectedLayer(const std::shared_ptr<GeneralBuffer<float>>& weight_buff,
                      const std::shared_ptr<GeneralBuffer<float>>& wgrad_buff,
                      const std::shared_ptr<Tensor<float>>& in_tensor,
                      const std::shared_ptr<Tensor<float>>& out_tensor,
                      TensorFormat_t weight_format, cublasHandle_t const& cublas_handle,
                      int device_id, bool use_mixed_precision = false);
  FullyConnectedLayer(const FullyConnectedLayer& C) = delete;
  FullyConnectedLayer& operator=(const FullyConnectedLayer&);

 private:
  /**
   * Use Gaussian initialization.
   */
  std::vector<float> get_initializer() override;
};
}  // namespace HugeCTR
