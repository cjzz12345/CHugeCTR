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

//#include <cooperative_groups.h>
// using namespace cooperative_groups;

namespace HugeCTR {

template <typename TypeKey>
__global__ void update_value_kernel(TypeKey* hash_value_index, int num, int embedding_vec_size,                                                   
                                    float *old_hash_table_value, float *new_hash_table_value) {
  int bid = blockIdx.x;   // each block corresponding to one sample
  int tid = threadIdx.x;  // each thread corresponding to one element in the embedding vector
  if (bid < num && tid < embedding_vec_size) {
 
      // store the embedding vector
      TypeKey feature_row_index=hash_value_index[bid]; 
  
      old_hash_table_value[feature_row_index * embedding_vec_size + tid] = new_hash_table_value[bid*embedding_vec_size+tid];
    }
  }

  template <typename TypeKey>
__global__ void global_to_virtual(TypeKey* virtual_id, TypeKey* global_id, int batch_key_size) {
  int bid = blockIdx.x;   // each block corresponding to one sample
  int tid = threadIdx.x;  // each thread corresponding to one element in the embedding vector
  int index=bid*16+tid;
  if (index < batch_key_size) {

      // store the embedding vector
      TypeKey key_index=virtual_id[index];
      virtual_id[index] = global_id[key_index];
    }
  }

  template <typename TypeKey>
__global__ void get_replaced_param(float *old_hash_table_value, float *new_hash_table_value, int num,                                                   
                                    int embedding_vec_size, TypeKey* hash_value_index) {
  int bid = blockIdx.x;   // each block corresponding to one sample
  int tid = threadIdx.x;  // each thread corresponding to one element in the embedding vector
  if (bid < num && tid < embedding_vec_size) {
 
      // store the embedding vector
      TypeKey feature_row_index=hash_value_index[bid];;

      new_hash_table_value[bid*embedding_vec_size+tid]=old_hash_table_value[feature_row_index * embedding_vec_size + tid] ;
    }
  }
}  // end of namespace HugeCTR
