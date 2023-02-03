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

#include "HugeCTR/include/embedding.hpp"
#include "HugeCTR/include/embeddings/distributed_slot_sparse_embedding_hash.hpp"
#include "HugeCTR/include/embeddings/localized_slot_sparse_embedding_hash.hpp"


namespace HugeCTR {

Embedding<EmbeddingCreator::TYPE_1>* EmbeddingCreator::create_distributed_sparse_embedding_hash(
    const Tensors<TYPE_1>& row_offsets_tensors, const Tensors<TYPE_1>& value_tensors,
    SparseEmbeddingHashParams embedding_params,
    const std::shared_ptr<GPUResourceGroup>& gpu_resource_group) {
  Embedding<TYPE_1>* sparse_embedding = new DistributedSlotSparseEmbeddingHash<TYPE_1>(
      row_offsets_tensors, value_tensors, embedding_params, gpu_resource_group);
  return sparse_embedding;
}

Embedding<EmbeddingCreator::TYPE_2>* EmbeddingCreator::create_distributed_sparse_embedding_hash(
    const Tensors<TYPE_2>& row_offsets_tensors, const Tensors<TYPE_2>& value_tensors,
    SparseEmbeddingHashParams embedding_params,
    const std::shared_ptr<GPUResourceGroup>& gpu_resource_group) {
  Embedding<TYPE_2>* sparse_embedding = new DistributedSlotSparseEmbeddingHash<TYPE_2>(
      row_offsets_tensors, value_tensors, embedding_params, gpu_resource_group);
  return sparse_embedding;
}

Embedding<EmbeddingCreator::TYPE_1>* EmbeddingCreator::create_localized_sparse_embedding_hash(
    const Tensors<TYPE_1>& row_offsets_tensors, const Tensors<TYPE_1>& value_tensors,
    SparseEmbeddingHashParams embedding_params, std::string plan_file,
    const std::shared_ptr<GPUResourceGroup>& gpu_resource_group) {
  Embedding<TYPE_1>* sparse_embedding = new LocalizedSlotSparseEmbeddingHash<TYPE_1>(
      row_offsets_tensors, value_tensors, embedding_params, plan_file, gpu_resource_group);
  return sparse_embedding;
}

Embedding<EmbeddingCreator::TYPE_2>* EmbeddingCreator::create_localized_sparse_embedding_hash(
    const Tensors<TYPE_2>& row_offsets_tensors, const Tensors<TYPE_2>& value_tensors,
    SparseEmbeddingHashParams embedding_params, std::string plan_file,
    const std::shared_ptr<GPUResourceGroup>& gpu_resource_group) {
  Embedding<TYPE_2>* sparse_embedding = new LocalizedSlotSparseEmbeddingHash<TYPE_2>(
      row_offsets_tensors, value_tensors, embedding_params, plan_file, gpu_resource_group);
  return sparse_embedding;
}

}  // namespace HugeCTR
