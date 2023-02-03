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

namespace HugeCTR {

struct MultiCrossForwardFunctor {
  MultiCrossForwardFunctor() = default;
  MultiCrossForwardFunctor(const MultiCrossForwardFunctor&) = delete;
  MultiCrossForwardFunctor& operator=(const MultiCrossForwardFunctor&) = delete;

  void operator()(cudaStream_t stream, const Tensor<float>& input_tensor,
                  const std::vector<const Tensor<float>*>& kernel_tensors,
                  const std::vector<const Tensor<float>*>& bias_tensors,
                  const std::vector<Tensor<float>*>& layer_output_tensors,
                  const std::vector<Tensor<float>*>& layer_hidden_tensors, int num_layers) const;
};

struct MultiCrossBackwardFunctor {
  MultiCrossBackwardFunctor() = default;
  MultiCrossBackwardFunctor(const MultiCrossBackwardFunctor&) = delete;
  MultiCrossBackwardFunctor& operator=(const MultiCrossBackwardFunctor&) = delete;

  void operator()(cudaStream_t stream, const Tensor<float>& input_tensor,
                  const std::vector<const Tensor<float>*>& kernel_tensors,
                  const std::vector<const Tensor<float>*>& layer_output_tensors,
                  const std::vector<const Tensor<float>*>& layer_hidden_tensors,
                  const Tensor<float>& grad_tensor, Tensor<float>& output_tensor,
                  const std::vector<Tensor<float>*>& kernel_output_tensors,
                  const std::vector<Tensor<float>*>& bias_output_tensors,
                  Tensor<float>& tmp_vec_tensor, const std::vector<Tensor<float>*>& tmp_mat_tensors,
                  int num_layers) const;
};

class MultiCrossLayer : public Layer {
 private:
  const int num_layers_;
  GeneralBufferPtr<float> blobs_buff_; /**< internal blobs' general buffer */
  Tensors<float> blob_tensors_;        /**< vector of internal blobs' tensors */
  Tensors<float> vec_tensors_;         //[h,1]

  TensorPtr<float> tmp_mat_tensors_[3];  //[h,w]
  TensorPtr<float> tmp_vec_tensor_;      //[h,1]
 public:
  /**
   * forward pass
   */
  void fprop(cudaStream_t stream) final;
  /**
   * backward pass
   */
  void bprop(cudaStream_t stream) final;

  MultiCrossLayer(const GeneralBufferPtr<float>& weight_buff,
                  const GeneralBufferPtr<float>& wgrad_buff, const TensorPtr<float>& in_tensor,
                  const TensorPtr<float>& out_tensor, int num_layers, int device_id);
  MultiCrossLayer(const MultiCrossLayer&) = delete;
  MultiCrossLayer& operator=(const MultiCrossLayer&) = delete;

 private:
  /**
   * Use Gaussian initialization.
   */
  std::vector<float> get_initializer() override;
};
}  // namespace HugeCTR
