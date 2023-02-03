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

#include "HugeCTR/include/common.hpp"
#include "HugeCTR/include/general_buffer.hpp"
#include "HugeCTR/include/tensor.hpp"
#include "HugeCTR/include/utils.hpp"

#include <memory>

namespace HugeCTR {

/**
 * @brief Abstract base class of Regularizer
 */
class Regularizer {
 public:
  /*
   * Constructor of Regularizer
   * @param weight_buff GeneralBuffer containing all the layers' weights
   * @param wgrad_buff GeneralBuffer containing all the layers' wgrads
   * @param batch_size Network batch size
   * @param device_id Device to be used
   */
  Regularizer(const std::shared_ptr<GeneralBuffer<float>>& weight_buff,
              const std::shared_ptr<GeneralBuffer<float>>& wgrad_buff, const int batch_size,
              const int device_id);

  /*
   * Destructor of Regularizer
   */
  virtual ~Regularizer() {}

  /*
   * Function that computes the regularization term
   * To customize it, override th private function do_compute_rterm
   * @param stream CUDA Stream where the kernel is executed
   */
  void compute_rterm(cudaStream_t stream);
  /*
   * Function that initialize wgrad
   * To customize it, override th private function do_initialize_wgrad
   * @param stream CUDA Stream where the kernel is executed
   */
  void initialize_wgrad(cudaStream_t stream);
  /*
   * Return the calculated regularization term
   */
  float get_rterm() const { return *h_rterm_; }

 protected:
  int get_batch_size() const { return batch_size_; }
  int get_device_id() const { return device_id_; }
  int get_n_sms() const { return n_sms_; }

 private:
  /*
   * To compute the regularization term, override this function.
   * It is called inside the public function compute_rterm
   * @param weight the device buffer of weight
   * @param h_rterm the host pointer to the regularization term
   * @param num_elements the number of weight values across layers
   * @param stream CUDA Stream where the kernel is executed
   */
  virtual void do_compute_rterm(const float* weight, float* h_rterm, int num_elements,
                                cudaStream_t stream) = 0;
  /*
   * To initialize wgrad, override this function.
   * It is called inside the public function initialize_wgrad
   * @param weight the device buffer of weight
   * @param wgrad the device buffer of wgrad
   * @param num_elements the number of weight values across layers
   * @param stream CUDA Stream where the kernel is executed
   */
  virtual void do_initialize_wgrad(const float* weight, float* wgrad, int num_elements,
                                   cudaStream_t stream) = 0;

  std::shared_ptr<GeneralBuffer<float>> weight_buff_;
  std::shared_ptr<GeneralBuffer<float>> wgrad_buff_;
  int batch_size_;
  int device_id_;
  int n_sms_;
  std::unique_ptr<float> h_rterm_;
};

}  // namespace HugeCTR
