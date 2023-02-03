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

#include "HugeCTR/include/session.hpp"
#include <nvToolsExt.h>
#include "HugeCTR/include/embedding.hpp"
#include "HugeCTR/include/utils.hpp"


namespace HugeCTR {

/**
 * check if device is avaliable.
 * lowest avaliable CC is min_major.min_minor
 * @param device_id gpu id
 * @param min_major minimum compute compatibility required
 * @param min_minor minimum compute compatibility required
 */
static void check_device(int device_id, int min_major, int min_minor) {
  int device_count = 0;
  CK_CUDA_THROW_(cudaGetDeviceCount(&device_count));
  if (device_id >= device_count) {
    CK_THROW_(Error_t::WrongInput, "device is not avaliable");
  }
  CudaDeviceContext context(device_id);
  cudaDeviceProp deviceProp;
  if (cudaGetDeviceProperties(&deviceProp, device_id) != cudaSuccess) {
    CK_THROW_(Error_t::InvalidEnv, "Invalid device:" + std::to_string(device_id));
    return;
  }
  std::cout << "Device " << device_id << ": " << deviceProp.name << std::endl;
  int major = deviceProp.major;
  int minor = deviceProp.minor;
  if (major < min_major) {
    CK_THROW_(Error_t::InvalidEnv, "Device Compute Compacity is low");
  } else if (major == min_major && minor < min_minor) {
    CK_THROW_(Error_t::InvalidEnv, "Device Compute Compacity is low");
  }
  return;
}

Session::Session(const std::string& json_name)
    : solver_config_(json_name),
      gpu_resource_group_(new GPUResourceGroup(solver_config_.device_map)) {
  try {
    int pid = 0;
#ifdef ENABLE_MPI
    int numprocs = 1;
    CK_MPI_THROW_(MPI_Comm_rank(MPI_COMM_WORLD, &pid));
    CK_MPI_THROW_(MPI_Comm_size(MPI_COMM_WORLD, &numprocs));
#endif

    for (auto dev : gpu_resource_group_->get_device_list()) {
      if (solver_config_.use_mixed_precision) {
        check_device(dev, 7,
                     0);  // to support mixed precision training earliest supported device is CC=70
      } else {
        check_device(dev, 6, 0);  // earliest supported device is CC=60
      }
    }
    parser_.reset(new Parser(json_name, solver_config_.batchsize,
                             solver_config_.use_mixed_precision, solver_config_.scaler));
    parser_->create_pipeline(data_reader_, data_reader_eval_, embedding_, networks_,
                             gpu_resource_group_);

    // init networks.
    const std::string TMP_DENSE_NAME = "tmp_dense_model.bin";
    if (pid == 0) {
      networks_[0]->init_params(TMP_DENSE_NAME);
    }
#ifdef ENABLE_MPI
    MPI_Barrier(MPI_COMM_WORLD);
#endif
    for (auto& network : networks_) {
      network->upload_params_to_device(TMP_DENSE_NAME);
    }
#ifdef ENABLE_MPI
    MPI_Barrier(MPI_COMM_WORLD);
#endif
    if (pid == 0) {
      if (std::remove(TMP_DENSE_NAME.c_str()) != 0) {
        CK_THROW_(Error_t::WrongInput, TMP_DENSE_NAME + " cannot be removed.");
      }
    }

    load_params_(solver_config_.model_file, solver_config_.embedding_files);
  } catch (const internal_runtime_error& rt_err) {
    std::cerr << rt_err.what() << std::endl;
  } catch (const std::exception& err) {
    std::cerr << err.what() << std::endl;
  }
}

/**
 * load the model (binary) from model_file.
 * In model file, model should be saved as
 * the sequence as discribed in configure file.
 **/
Error_t Session::load_params_(const std::string& model_file,
                              const std::vector<std::string>& embedding_files) {
  try {
    if (!embedding_files.empty()) {
      int i = 0;
      for (auto& embedding_file : embedding_files) {
        std::ifstream embedding_stream(embedding_file, std::ifstream::binary);
        if (!embedding_stream.is_open()) {
          CK_THROW_(Error_t::WrongInput, "Cannot open sparse model file");
        }
        std::cout << "Loading sparse model: " << embedding_file << std::endl;
        embedding_[i]->upload_params_to_device(embedding_stream);
        embedding_stream.close();
        i++;
      }
    }
    if (!model_file.empty()) {
      std::ifstream model_stream(model_file, std::ifstream::binary);
      if (!model_stream.is_open()) {
        CK_THROW_(Error_t::WrongInput, "Cannot open dense model file");
      }
      std::unique_ptr<float[]> weight(new float[networks_[0]->get_params_num()]());
      model_stream.read(reinterpret_cast<char*>(weight.get()),
                        networks_[0]->get_params_num() * sizeof(float));

      std::cout << "Loading dense model: " << model_file << std::endl;
      for (auto& network : networks_) {
        network->upload_params_to_device(weight.get());
      }
      model_stream.close();
    }
  } catch (const internal_runtime_error& rt_err) {
    std::cerr << rt_err.what() << std::endl;
    return rt_err.get_error();
  } catch (const std::exception& err) {
    std::cerr << err.what() << std::endl;
    return Error_t::UnspecificError;
  }
  return Error_t::Success;
}

void network_train_helper(int id, Network* n) {
  try {
    n->train();
  } catch (const internal_runtime_error& rt_err) {
    std::cerr << rt_err.what() << std::endl;
  } catch (const std::exception& err) {
    std::cerr << err.what() << std::endl;
  }
  return;
}

Error_t Session::train() {
  try {
    data_reader_->read_a_batch_to_device_delay_release();
    // HugeCTR::Timer timer;
    //     timer.start();
    for (auto& one_embedding : embedding_) {
      one_embedding->forward();
    }
//     timer.stop();
//  MESSAGE_(" train forward   :   "  +
//                        std::to_string(timer.elapsedMicroseconds()) );
//                           timer.start();
    data_reader_->ready_to_collect();
    if (networks_.size() > 1) {
      // execute dense forward and backward with multi-cpu threads
      for (unsigned int i = 0; i < networks_.size(); i++) {
        gpu_resource_group_->results[i] = gpu_resource_group_->train_thread_pool.push(
            [this, i](int id) { network_train_helper(id, networks_[i].get()); });
      }
      for (unsigned int i = 0; i < networks_.size(); i++) {
        gpu_resource_group_->results[i].get();
      }
    } else if (networks_.size() == 1) {
      networks_[0]->train();
    } else {
      assert(!"networks_.size() should not less than 1.");
    }
    // wgrad exchange
    if (gpu_resource_group_->get_total_gpu_count() > 1) {
      CK_NCCL_THROW_(ncclGroupStart());
      for (auto& network : networks_) {
        network->exchange_wgrad();
      }
      CK_NCCL_THROW_(ncclGroupEnd());
    }
    for (auto& network : networks_) {
      network->update_params();
    }
//         timer.stop();
//  MESSAGE_(" train network   :   "  +
//                        std::to_string(timer.elapsedMicroseconds()) );
//                           timer.start();

    for (auto& one_embedding : embedding_) {
      one_embedding->backward();
      one_embedding->update_params();
    }
//             timer.stop();
//  MESSAGE_(" train backward   :   "  +
//                        std::to_string(timer.elapsedMicroseconds()) );
  } catch (const internal_runtime_error& rt_err) {
    std::cerr << rt_err.what() << std::endl;
    return rt_err.get_error();
  } catch (const std::exception& err) {
    std::cerr << err.what() << std::endl;
    return Error_t::UnspecificError;
  }
  return Error_t::Success;
}

void network_eval_helper(int id, Network* n) {
  try {
    n->eval();
  } catch (const internal_runtime_error& rt_err) {
    std::cerr << rt_err.what() << std::endl;
  } catch (const std::exception& err) {
    std::cerr << err.what() << std::endl;
  }
}
Error_t Session::getmap(){
   data_reader_eval_->data_collector_->em_param_map=data_reader_->data_collector_->em_param_map;
   return Error_t::Success;
} 
Error_t Session::setmap(){
   data_reader_->data_collector_->em_param_map=data_reader_eval_->data_collector_->em_param_map;
   return Error_t::Success;
} 
Error_t Session::eval() {
  try {
    if (data_reader_eval_ == nullptr) return Error_t::NotInitialized;
    data_reader_eval_->read_a_batch_to_device_delay_release();
    for (auto& one_embedding : embedding_) {
      one_embedding->forward();
    }
    data_reader_eval_->ready_to_collect();
    if (networks_.size() > 1) {
      for (unsigned int i = 0; i < networks_.size(); i++) {
        gpu_resource_group_->results[i] = gpu_resource_group_->train_thread_pool.push(
            [this, i](int id) { network_eval_helper(id, networks_[i].get()); });
      }
      for (unsigned int i = 0; i < networks_.size(); i++) {
        gpu_resource_group_->results[i].get();
      }
    } else if (networks_.size() == 1) {
      networks_[0]->eval();
    } else {
      assert(!"networks_.size() should not less than 1.");
    }
  } catch (const internal_runtime_error& rt_err) {
    std::cerr << rt_err.what() << std::endl;
    return rt_err.get_error();
  } catch (const std::exception& err) {
    std::cerr << err.what() << std::endl;
    return Error_t::UnspecificError;
  }
  return Error_t::Success;
}

Error_t Session::download_params_to_files(std::string prefix, int iter) {
  std::string snapshot_dense_name = prefix + "_dense_" + std::to_string(iter) + ".model";
  std::vector<std::string> snapshot_sparse_names;
  if (iter <= 0) {
    return Error_t::WrongInput;
  }

  for (unsigned int i = 0; i < embedding_.size(); i++) {
    snapshot_sparse_names.push_back(prefix + std::to_string(i) + "_sparse_" + std::to_string(iter) +
                                    ".model");
  }
  return download_params_to_files_(snapshot_dense_name, snapshot_sparse_names);
}

Error_t Session::download_params_to_files_(std::string weights_file,
                                           const std::vector<std::string>& embedding_files) {
  try {
    {
      int i = 0;
      for (auto& embedding_file : embedding_files) {
        std::ofstream out_stream_embedding(embedding_file, std::ofstream::binary);
        embedding_[i]->download_params_to_host(out_stream_embedding);
        out_stream_embedding.close();
        i++;
      }
    }
    int pid = 0;
#ifdef ENABLE_MPI
    int numprocs = 1;
    CK_MPI_THROW_(MPI_Comm_rank(MPI_COMM_WORLD, &pid));
    CK_MPI_THROW_(MPI_Comm_size(MPI_COMM_WORLD, &numprocs));
#endif
    if (pid == 0) {
      std::ofstream out_stream_weight(weights_file, std::ofstream::binary);
      networks_[0]->download_params_to_host(out_stream_weight);
      std::string no_trained_params = networks_[0]->get_no_trained_params_in_string();
      if (no_trained_params.length() != 0) {
        std::string ntp_file = weights_file + ".ntp.json";
        std::ofstream out_stream_ntp(ntp_file, std::ofstream::out);
        out_stream_ntp.write(no_trained_params.c_str(), no_trained_params.length());
        out_stream_ntp.close();
      }
      out_stream_weight.close();
    }

  } catch (const internal_runtime_error& rt_err) {
    std::cerr << rt_err.what() << std::endl;
    return rt_err.get_error();
  } catch (const std::exception& err) {
    std::cerr << err.what() << std::endl;
    return Error_t::UnspecificError;
  }
  return Error_t::Success;
}

Error_t Session::get_current_loss(float* loss) {
  try {
    float loss_sum = 0.f;
    float loss_reduced = 0.f;
    int numprocs = 1;
#ifdef ENABLE_MPI
    int pid = 0;
    CK_MPI_THROW_(MPI_Comm_rank(MPI_COMM_WORLD, &pid));
    CK_MPI_THROW_(MPI_Comm_size(MPI_COMM_WORLD, &numprocs));
#endif
    // Collect all the loss from every network and average
    for (auto& network : networks_) {
      loss_sum += network->get_loss();
    }
    if (numprocs > 1) {
#ifdef ENABLE_MPI
      CK_MPI_THROW_(MPI_Reduce(&loss_sum, &loss_reduced, 1, MPI_FLOAT, MPI_SUM, 0, MPI_COMM_WORLD));
#endif
    } else {
      loss_reduced = loss_sum;
    }
    *loss = loss_reduced / networks_.size() / numprocs;
  } catch (const internal_runtime_error& rt_err) {
    std::cerr << rt_err.what() << std::endl;
    return rt_err.get_error();
  } catch (const std::exception& err) {
    std::cerr << err.what() << std::endl;
    return Error_t::UnspecificError;
  }
  return Error_t::Success;
}

Session::~Session() {
  try {
    for (auto device : gpu_resource_group_->get_device_list()) {
      CudaDeviceContext context(device);
      CK_CUDA_THROW_(cudaDeviceSynchronize());
    }

  } catch (const internal_runtime_error& rt_err) {
    std::cerr << rt_err.what() << std::endl;
  } catch (const std::exception& err) {
    std::cerr << err.what() << std::endl;
  }
}

}  // namespace HugeCTR
