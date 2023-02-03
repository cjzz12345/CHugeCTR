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

#include "HugeCTR/include/optimizer.hpp"

namespace HugeCTR {

typedef struct MomentumSGDHyperParameters_ {
  float lr;
  float momentum_factor;
} MomentumSGDHyperParameters;

/**
 * SGD optimizer with Momentum
 */
class MomentumSGD : public Optimizer {
 public:
  /**
   * Constructor of MomentumSGD.
   * @param weight weights to be updated
   * @param wgrad gradient for weights
   * @param device_id the id of GPU where update kernel is launched
   * @param learning_rate learning rate
   * @param momentum_factor momentum factor
   */
  MomentumSGD(const std::shared_ptr<GeneralBuffer<float>>& weight,
              const std::shared_ptr<GeneralBuffer<float>>& wgrad, int device_id,
              float learning_rate, float momentum_factor, float scaler = 1.f)
      : Optimizer(weight, wgrad, device_id, learning_rate, scaler),
        momentum_factor_(momentum_factor) {
    momentum_.reset(new GeneralBuffer<float>(weight_->get_num_elements(), device_id_));
    momentum_->reset_sync();
  }

  /**
   * update the weights using gradient
   * @param stream cuda stream used by update kernel
   */
  void update(cudaStream_t stream) final;

 private:
  std::unique_ptr<GeneralBuffer<float>> momentum_;
  float momentum_factor_;
};

}  // namespace HugeCTR
