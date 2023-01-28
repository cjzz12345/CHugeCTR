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
#include "HugeCTR/include/csr.hpp"
#include "HugeCTR/include/csr_chunk.hpp"
#include "HugeCTR/include/general_buffer.hpp"
#include "HugeCTR/include/gpu_resource.hpp"
#include "HugeCTR/include/heapex.hpp"
#include "HugeCTR/include/tensor.hpp"
#include "HugeCTR/include/data_collector.cuh"
#include "HugeCTR/include/hashtable/nv_hashtable.h"
#include <unordered_map>

#ifdef ENABLE_MPI
#include <mpi.h>
#endif
namespace HugeCTR
{

#ifdef ENABLE_MPI
  template <typename TypeKey>
  struct ToMpiType;

  template <>
  struct ToMpiType<long long>
  {
    static MPI_Datatype T() { return MPI_LONG_LONG; }
  };

  template <>
  struct ToMpiType<unsigned int>
  {
    static MPI_Datatype T() { return MPI_UNSIGNED; }
  };

  template <>
  struct ToMpiType<float>
  {
    static MPI_Datatype T() { return MPI_FLOAT; }
  };

#endif

  void split(std::shared_ptr<Tensor<float>> label_tensor, std::shared_ptr<Tensor<float>> dense_tensor,
             const std::shared_ptr<GeneralBuffer<float>> label_dense_buffer, cudaStream_t stream);

  /**
   * @brief A helper class of data reader.
   *
   * This class implement asynchronized data collecting from heap
   * to output of data reader, thus data collection and training
   * can work in a pipeline.
   */
  template <typename TypeKey>
  class DataCollector
  {
    using NvHashTable =
        HashTable<TypeKey, TypeKey, std::numeric_limits<TypeKey>::max()>;

  private:
    enum STATUS
    {
      READY_TO_WRITE,
      READY_TO_READ,
      STOP
    };
    STATUS stat_{READY_TO_WRITE};
    std::mutex stat_mtx_;
    std::condition_variable stat_cv_;
    std::shared_ptr<HeapEx<CSRChunk<TypeKey>>> csr_heap_;

    Tensors<float> label_tensors_;
    Tensors<float> dense_tensors_;
    GeneralBuffers<TypeKey> csr_buffers_;
    GeneralBuffers<float> label_dense_buffers_internal_;
    GeneralBuffers<TypeKey> csr_buffers_internal_;
    std::shared_ptr<GPUResourceGroup> device_resources_;
    int num_params_;
    long long counter_{0};
    int pid_{0}, num_procs_{1};

    Tensors<float> hash_table_value_tensors_;

    int em_param_counter{0};
    float *em_param_table;

    /**< Hash table.  */
  public:
    std::vector<std::shared_ptr<NvHashTable>> hash_tables_;
    std::unordered_map<TypeKey, int> em_param_map;
    /**
     * Ctor.
     * @param label_tensors label tensors (GPU) of data reader.
     * @param dense_tensors dense tensors (GPU) of data reader.
     * @param csr_buffers csr buffers (GPU) of data reader.
     * @param device_resources gpu resources.
     * @param csr_heap heap of data reader.
     */

    DataCollector(const Tensors<float> &label_tensors, const Tensors<float> &dense_tensors,
                  const GeneralBuffers<TypeKey> &csr_buffers,
                  const std::shared_ptr<GPUResourceGroup> &device_resources,
                  const std::shared_ptr<HeapEx<CSRChunk<TypeKey>>> &csr_heap = nullptr);

    void set_hashtabel(std::vector<std::unique_ptr<NvHashTable>> oldtable);
    void set_hashvalue(Tensors<float> oldvalue);

    void set_ready_to_write();

    /**
     * Collect data from heap to each GPU (node).
     */
    void collect();

    /**
     * Read a batch to device.
     */
    void read_a_batch_to_device();

    /**
     * Break the collecting and stop. Only used in destruction.
     */
    void stop()
    {
#ifdef ENABLE_MPI
      CK_MPI_THROW_(MPI_Barrier(MPI_COMM_WORLD));
#endif
      stat_ = STOP;
      stat_cv_.notify_all();
    }

    /**
     * Dtor.
     */
    ~DataCollector()
    {
      if (stat_ != STOP)
        stop();
    }
  };

  template <typename TypeKey>
  DataCollector<TypeKey>::DataCollector(const Tensors<float> &label_tensors,
                                        const Tensors<float> &dense_tensors,
                                        const GeneralBuffers<TypeKey> &csr_buffers,
                                        const std::shared_ptr<GPUResourceGroup> &device_resources,
                                        const std::shared_ptr<HeapEx<CSRChunk<TypeKey>>> &csr_heap)
      : csr_heap_(csr_heap),
        label_tensors_(label_tensors),
        dense_tensors_(dense_tensors),
        csr_buffers_(csr_buffers),
        device_resources_(device_resources)
  {
    try
    {
      // input check
      if (stat_ != READY_TO_WRITE)
      {
        CK_THROW_(Error_t::WrongInput, "stat_ != READY_TO_WRITE");
      }
      if (label_tensors.size() != dense_tensors.size())
      {
        CK_THROW_(Error_t::WrongInput, "label_tensors.size() != dense_tensors.size()");
      }
      cudaMallocHost(&em_param_table, 2400000 * sizeof(float));
      // create internal buffers
      auto &local_device_list = device_resources_->get_device_list();
      for (unsigned int i = 0; i < local_device_list.size(); i++)
      {
        assert(local_device_list[i] == label_tensors_[i]->get_device_id());
        assert(local_device_list[i] == dense_tensors_[i]->get_device_id());
        int buf_size = label_tensors_[i]->get_num_elements() + dense_tensors_[i]->get_num_elements();
        label_dense_buffers_internal_.emplace_back(
            std::make_shared<GeneralBuffer<float>>(buf_size, local_device_list[i]));
      }
      for (auto &cb : csr_buffers_)
      {
        csr_buffers_internal_.emplace_back(
            new GeneralBuffer<TypeKey>(cb->get_num_elements(), cb->get_device_id()));
      }
      num_params_ = csr_buffers_.size() / local_device_list.size();

#ifdef ENABLE_MPI
      CK_MPI_THROW_(MPI_Comm_rank(MPI_COMM_WORLD, &pid_));
      CK_MPI_THROW_(MPI_Comm_size(MPI_COMM_WORLD, &num_procs_));
#endif
    }
    catch (const std::runtime_error &rt_err)
    {
      std::cerr << rt_err.what() << std::endl;
    }
  }

  /**************************************
   * Each node will have one DataCollector.
   * Each iteration, one of the data collector will
   * send it's CSR buffers to remote node.
   ************************************/
  template <typename TypeKey>
  void DataCollector<TypeKey>::collect()
  {
    std::unique_lock<std::mutex> lock(stat_mtx_);

    // my turn
    CSRChunk<TypeKey> *chunk_tmp = nullptr;

    int total_device_count = device_resources_->get_total_gpu_count();
    csr_heap_->data_chunk_checkout(&chunk_tmp);

    while (stat_ != READY_TO_WRITE && stat_ != STOP)
    {
      stat_cv_.wait(lock);
    }
    if (stat_ == STOP)
    {
      return;
    }

    const auto &csr_cpu_buffers = chunk_tmp->get_csr_buffers();
    const auto &label_dense_buffers = chunk_tmp->get_label_buffers();
    const int num_params =
        chunk_tmp->get_num_params(); // equal to the num of output of data reader in json
    if (num_params_ != num_params)
    {
      CK_THROW_(Error_t::WrongInput, "job_ is ???");
    }
    assert(static_cast<int>(label_dense_buffers.size()) == total_device_count);

    for (int i = 0; i < total_device_count; i++)
    {
      int pid = device_resources_->get_pid(i);
      int label_copy_num = (label_dense_buffers[0]).get_num_elements();
      if (pid == pid_)
      {
        int local_id = device_resources_->get_local_id(i);
        CudaDeviceContext context(device_resources_->get_local_device_id(i));
        for (int j = 0; j < num_params; j++)
        {
          int csr_copy_num = (csr_cpu_buffers[i * num_params + j].get_num_rows() +
                              csr_cpu_buffers[i * num_params + j].get_sizeof_value() + 1);

          CK_CUDA_THROW_(cudaMemcpyAsync(
              csr_buffers_internal_[local_id * num_params + j]->get_ptr_with_offset(0),
              csr_cpu_buffers[i * num_params + j].get_buffer(), csr_copy_num * sizeof(TypeKey),
              cudaMemcpyHostToDevice, (*device_resources_)[local_id]->get_data_copy_stream()));

          TypeKey *d_hashkey;
          int batch_key_size = (csr_copy_num - 1) / 2;
          int unduplicated_key_num = csr_cpu_buffers[i * num_params + j].get_undu_key_num();
          cudaMalloc(&d_hashkey, batch_key_size * sizeof(TypeKey));
          const cudaStream_t stream = (*device_resources_)[local_id]->get_data_copy_stream();
          CK_CUDA_THROW_(cudaMemcpyAsync(
              d_hashkey, csr_cpu_buffers[i * num_params + j].get_undu_key(), unduplicated_key_num * sizeof(TypeKey),
              cudaMemcpyHostToDevice, stream));

          int vec_size = 11;

          if (counter_ == 0)
          {
            TypeKey *kkey;
            cudaMallocHost(&kkey, unduplicated_key_num * sizeof(TypeKey));
            CK_CUDA_THROW_(cudaMemcpyAsync(
                kkey, csr_cpu_buffers[i * num_params + j].get_undu_key(), unduplicated_key_num * sizeof(TypeKey),
                cudaMemcpyHostToHost, stream));
            cudaStreamSynchronize(stream);
            // std::cout<<"start  insert  "<<std::endl;
            for (int i = 0; i < unduplicated_key_num; i++)
            {
              // std::cout<<"in map:  " <<  *h_counter<<std::endl<<std::endl;

              const TypeKey key = kkey[i];
              auto iterator = em_param_map.find(key);
              if (iterator == em_param_map.end())
              {
                //  std::cout<<"        " <<  key;
                em_param_map.insert(std::pair<TypeKey, int>(key, em_param_counter));
                em_param_counter++;
              }
            }
          }

          if (counter_ >= 1)
          {
            const auto &hash_table = hash_tables_[local_id * 3].get();
            const auto &hash_table1 = hash_tables_[local_id * 3 + 1].get();
            const auto &hash_table2 = hash_tables_[local_id * 3 + 2].get();
            const auto &hash_table_value = hash_table_value_tensors_[local_id]->get_ptr();

            float *h_hashvalue, *d_hashvalue, *d_exchashvalue, *h_exchashvalue;
            int *counter, *h_counter, cc, *d_dump_counter;
            TypeKey *missingkey, *h_missingkey, *h_hashkey, *d_hash_table_key, *d_hash_table_value_index, *d_replacedkey, *h_replacedkey, *d_replace_index;
            cudaMalloc(&missingkey, batch_key_size * sizeof(TypeKey));
            cudaMallocHost(&h_missingkey, batch_key_size * sizeof(TypeKey));
            cudaMalloc(&counter, sizeof(int));
            cudaMallocHost(&h_counter, sizeof(int));
            cudaMallocHost(&h_hashkey, batch_key_size * sizeof(TypeKey));

            cudaMalloc(&d_hash_table_key, batch_key_size * sizeof(TypeKey));
            cudaMalloc(&d_hash_table_value_index, batch_key_size * sizeof(TypeKey));
            cudaMalloc(&d_dump_counter, sizeof(int));
            cudaMalloc(&d_replacedkey, batch_key_size * sizeof(TypeKey));
            cudaMallocHost(&h_replacedkey, batch_key_size * sizeof(TypeKey));
            cudaMalloc(&d_replace_index, batch_key_size * sizeof(TypeKey));
            *h_counter = 0;

            if ((counter_ % 2) == 1)
            {
              // std::cout<<"1  1"<<hash_table1->get_size(stream)<<std::endl;
              hash_table1->clear(stream);
              //  std::cout<<"2  1"<<hash_table1->get_size(stream)<<std::endl;
              hash_table1->insert(d_hashkey, d_hash_table_value_index, unduplicated_key_num, stream);
              //   std::cout<<"3  1"<<hash_table1->get_size(stream)<<std::endl;
            }
            else
            {
              //    std::cout<<"1  2"<<hash_table2->get_size(stream)<<std::endl;
              hash_table2->clear(stream);
              //     std::cout<<"2  2"<<hash_table2->get_size(stream)<<std::endl;
              hash_table2->insert(d_hashkey, d_hash_table_value_index, unduplicated_key_num, stream);
              //  std::cout<<"3  2"<<hash_table2->get_size(stream)<<std::endl;
            }

            CK_CUDA_THROW_(cudaMemcpyAsync(counter, h_counter,
                                           sizeof(int), cudaMemcpyHostToDevice, stream));

            //               CK_CUDA_THROW_(cudaMemcpyAsync(h_hashkey, d_hashkey,
            //                 104*sizeof(TypeKey), cudaMemcpyDeviceToHost,stream));
            //    std::cout <<"   hashkeys part                     "<<std::endl<<std::endl;
            //  for(int i=0;i<104 ;i++)
            //   {
            //     std::cout <<"    hashkeys                    "<<i<<"       " <<h_hashkey[i];
            //   }
            // std::cout<<std::endl<<std::endl;

            int *s_counter, *s_counter1;
            cudaMalloc(&s_counter, sizeof(int));
            cudaMallocHost(&s_counter1, sizeof(int));
            *s_counter1 = 0;
            CK_CUDA_THROW_(cudaMemcpyAsync(s_counter, s_counter1,
                                           sizeof(int), cudaMemcpyHostToDevice,
                                           stream));
            hash_table->get(d_hashkey, d_hash_table_value_index, unduplicated_key_num, stream, missingkey, counter);

            CK_CUDA_THROW_(cudaMemcpyAsync(h_counter, counter,
                                           sizeof(int), cudaMemcpyDeviceToHost,
                                           stream));
            cudaStreamSynchronize(stream);

            //  std::cout<<"h counter:  " << *h_counter<<  std::endl<<std::endl;

            if (*h_counter >= 1)
            {

              cudaMalloc(&d_hashvalue, *h_counter * vec_size * sizeof(float));
              cudaMallocHost(&h_hashvalue, *h_counter * vec_size * sizeof(float));

              CK_CUDA_THROW_(cudaMemcpyAsync(h_missingkey, missingkey,
                                             *h_counter * sizeof(TypeKey), cudaMemcpyDeviceToHost, stream));

              hash_table->replace_insert(missingkey, d_hash_table_value_index, *h_counter, stream, hash_table1->table_, hash_table2->table_, s_counter, d_replacedkey, d_replace_index);
              CK_CUDA_THROW_(cudaMemcpyAsync(s_counter1, s_counter,
                                             sizeof(int), cudaMemcpyDeviceToHost,
                                             stream));
              cudaStreamSynchronize(stream);

              if (*s_counter1 > 0)
              {
                cudaMalloc(&d_exchashvalue, *s_counter1 * vec_size * sizeof(float));
                cudaMallocHost(&h_exchashvalue, *s_counter1 * vec_size * sizeof(float));
                //  std::cout<<"before  update  replaced :  " <<  std::endl<<std::endl;
                get_replaced_param<TypeKey><<<*s_counter1, vec_size, 0, stream>>>(
                    hash_table_value, d_exchashvalue, *s_counter1, vec_size, d_replace_index);

                CK_CUDA_THROW_(cudaMemcpyAsync(h_exchashvalue, d_exchashvalue,
                                               *s_counter1 * vec_size * sizeof(float), cudaMemcpyDeviceToHost, stream));

                CK_CUDA_THROW_(cudaMemcpyAsync(h_replacedkey, d_replacedkey,
                                               *s_counter1 * sizeof(TypeKey), cudaMemcpyDeviceToHost, stream));
              }

              printf("%f    ", (float)(*h_counter) / (batch_key_size));
              //  std::cout<<"start  insert  "<<std::endl;
              for (int i = 0; i < *h_counter; i++)
              {
                // std::cout<<"in map:  " <<  *h_counter<<std::endl<<std::endl;
                const TypeKey key = h_missingkey[i];
                auto iterator = em_param_map.find(key);
                if (iterator == em_param_map.end())
                {
                  //  std::cout<<"        " <<  key;
                  em_param_map.insert(std::pair<TypeKey, int>(key, em_param_counter));
                  for (int j = 0; j < vec_size; j++)
                  {
                    h_hashvalue[i * vec_size + j] = em_param_table[em_param_counter * vec_size + j];
                  }
                  em_param_counter++;
                }
                else
                {
                  int index = iterator->second;
                  for (int j = 0; j < vec_size; j++)
                  {
                    h_hashvalue[i * vec_size + j] = em_param_table[index * vec_size + j];
                  }
                }
              }

              // TypeKey * h_hash_table_value_index;
              // cudaMallocHost(&h_hash_table_value_index,*h_counter*sizeof(TypeKey));
              // CK_CUDA_THROW_(cudaMemcpyAsync(h_hash_table_value_index, d_hash_table_value_index,
              //                          *h_counter*sizeof(TypeKey), cudaMemcpyDeviceToHost,
              //                          stream));
              CK_CUDA_THROW_(cudaMemcpyAsync(d_hashvalue, h_hashvalue,
                                             *h_counter * vec_size * sizeof(float), cudaMemcpyHostToDevice,
                                             stream));

              //  for(int i=0;i<*h_counter;i++){
              //    std::cout<<"          " <<  h_hash_table_value_index[i];
              //  }

              // std::cout<<"after d_hash:  " <<  std::endl<<std::endl;
              //     dim3 blockSize(vec_size, 1, 1);
              // dim3 gridSize(*h_counter, 1, 1);
              update_value_kernel<TypeKey><<<*h_counter, vec_size, 0, stream>>>(
                  d_hash_table_value_index, *h_counter, vec_size, hash_table_value, d_hashvalue);
              cudaStreamSynchronize(stream);
              //  std::cout<<"after  update  value :  " <<  std::endl<<std::endl;
              if (*s_counter1 > 0)
              {

                for (int i = 0; i < *s_counter1; i++)
                {
                  //  std::cout<<" find   " <<  h_replacedkey[i]<<std::endl;
                  auto iterator = em_param_map.find(h_replacedkey[i]);
                  if (iterator == em_param_map.end())
                  {
                    //  std::cout<<"can't find   " <<  h_replacedkey[i]<<std::endl<<std::endl;
                  }

                  int index = iterator->second;
                  //  std::cout<<"index :   " <<  index<<std::endl;
                  for (int j = 0; j < vec_size; j++)
                  {
                    em_param_table[index * vec_size + j] = h_exchashvalue[i * vec_size + j];
                  }
                }
                // std::cout<<"after  update  cpu em :  " <<  std::endl<<std::endl;
              }

              // cudaStreamSynchronize(stream);
              //  std::cout <<"    after update                     "<<std::endl<<std::endl;
            }
            // dim blockSize=(16,1,1);
            //  dim  gridSize=(batch_key_size/16+1,1,1);

            //           cc=hash_table->get_size(stream);
            // // hash_table->dump(d_hash_table_key, d_hash_table_value_index, 0,
            // //                             150, d_dump_counter,stream);
            // //   CK_CUDA_THROW_(cudaMemcpyAsync(h_hashkey, d_hash_table_key,
            // //                                       150*sizeof(TypeKey), cudaMemcpyDeviceToHost,stream));
            //                                         cudaStreamSynchronize(stream);
            // std::cout <<"    after get size                    "<<std::endl<<std::endl;
            //         for(int i=0;i<150 ;i++)
            //   {
            //     std::cout <<"     after replace insert                    "<<i<<"       " <<h_hashkey[i];
            //   }
            // std::cout<<std::endl<<std::endl;

            // * h_counter=0;

            //   CK_CUDA_THROW_(cudaMemcpyAsync(counter, h_counter,
            //                                      sizeof(int), cudaMemcpyHostToDevice,  stream));
            //     hash_table->get(d_hashkey, d_hash_table_value_index, (csr_copy_num-1)/2, stream,missingkey,counter);
            //         CK_CUDA_THROW_(cudaMemcpyAsync(h_counter, counter,
            //                                      sizeof(int), cudaMemcpyDeviceToHost,
            //                                     stream));
            //                                      cudaStreamSynchronize(stream);
            //                                      CK_CUDA_THROW_(cudaMemcpyAsync(h_missingkey, missingkey,
            //                                       512*sizeof(TypeKey), cudaMemcpyDeviceToHost,stream));
            //                                      cudaStreamSynchronize(stream);
            //        std::cout <<"    missingkeys part          222          "<<std::endl<<std::endl;
            //   for(int i=0;i<*h_counter ;i++)
            //   {
            //     std::cout <<"    missingkeys                    "<<i<<"      " <<h_missingkey[i];
            //   }
            // std::cout<<std::endl<<std::endl;
            cudaFree(missingkey);
            cudaFreeHost(h_missingkey);
            cudaFree(counter);
            cudaFreeHost(h_counter);
            cudaFreeHost(h_hashkey);
            cudaFree(d_hash_table_key);
            cudaFree(d_hash_table_value_index);
            cudaFree(d_dump_counter);
            cudaFree(s_counter);
            cudaFreeHost(s_counter1);
            cudaFree(d_hashvalue);
            cudaFreeHost(h_hashvalue);
            cudaFree(d_exchashvalue);
            cudaFreeHost(h_replacedkey);
            cudaFree(d_replacedkey);
            cudaFree(d_replace_index);
          }

          global_to_virtual<TypeKey><<<batch_key_size / 16 + 1, 16, 0, stream>>>(
              csr_buffers_internal_[local_id * num_params + j]->get_ptr_with_offset(csr_cpu_buffers[i * num_params + j].get_num_rows() + 1), d_hashkey, batch_key_size);
          cudaFree(d_hashkey);
          csr_cpu_buffers[i * num_params + j].reset_undu_key_num();
        }

        CK_CUDA_THROW_(cudaMemcpyAsync(
            label_dense_buffers_internal_[local_id]->get_ptr_with_offset(0),
            label_dense_buffers[i].get(), label_copy_num * sizeof(float), cudaMemcpyHostToDevice,
            (*device_resources_)[local_id]->get_data_copy_stream()));
      }
    }

    // sync
    for (int i = 0; i < total_device_count; i++)
    {
      int pid = device_resources_->get_pid(i);
      if (pid_ == pid)
      {
        int local_id = device_resources_->get_local_id(i);
        CudaDeviceContext context(device_resources_->get_local_device_id(i));
        CK_CUDA_THROW_(cudaStreamSynchronize((*device_resources_)[local_id]->get_data_copy_stream()));
      }
    }

    csr_heap_->chunk_free_and_checkin();
    counter_++;
    stat_ = READY_TO_READ;
    stat_cv_.notify_all();
  }

  template <typename TypeKey>
  void DataCollector<TypeKey>::read_a_batch_to_device()
  {
    HugeCTR::Timer timer;
    timer.start();

    std::unique_lock<std::mutex> lock(stat_mtx_);
    while (stat_ != READY_TO_READ && stat_ != STOP)
    {
      stat_cv_.wait(lock);
    }
    if (stat_ == STOP)
    {
      return;
    }
    timer.stop();
    //  MESSAGE_(" Time Gap :   "  +
    //                        std::to_string(timer.elapsedMicroseconds()) );
    for (unsigned int i = 0; i < device_resources_->size(); i++)
    {
      CudaDeviceContext context((*device_resources_)[i]->get_device_id());
      for (int j = 0; j < num_params_; j++)
      {
        int csr_id = i * num_params_ + j;
        CK_CUDA_THROW_(cudaMemcpyAsync(csr_buffers_[csr_id]->get_ptr_with_offset(0),
                                       csr_buffers_internal_[csr_id]->get_ptr_with_offset(0),
                                       csr_buffers_[csr_id]->get_size(), cudaMemcpyDeviceToDevice,
                                       (*device_resources_)[i]->get_stream()));
      }

      split(label_tensors_[i], dense_tensors_[i], label_dense_buffers_internal_[i],
            (*device_resources_)[i]->get_stream());
    }
    for (unsigned int i = 0; i < device_resources_->size(); i++)
    {
      CudaDeviceContext context((*device_resources_)[i]->get_device_id());
      CK_CUDA_THROW_(cudaStreamSynchronize((*device_resources_)[i]->get_stream()));
    }
  }

  template <typename TypeKey>
  void DataCollector<TypeKey>::set_ready_to_write()
  {
    stat_ = READY_TO_WRITE;
    stat_cv_.notify_all();
  }

  template <typename TypeKey>
  void DataCollector<TypeKey>::set_hashtabel(std::vector<std::unique_ptr<NvHashTable>> oldtable)
  {
    hash_tables_ = oldtable;
  }

  template <typename TypeKey>
  void DataCollector<TypeKey>::set_hashvalue(Tensors<float> oldvalue)
  {
    hash_table_value_tensors_ = oldvalue;
  }

} // namespace HugeCTR
