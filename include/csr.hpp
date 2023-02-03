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
#include <cuda_runtime_api.h>
#include <iostream>
#include <stdexcept>
#include "HugeCTR/include/common.hpp"
#include "HugeCTR/include/pinned_buffer.hpp"


namespace HugeCTR {

/**
 * @brief A wrapper of simplified CSR buffer and related method.
 *
 * This class contains all the routines of data loading in CSR format, and
 * export the formated data (in CPU buffer) to users.
 * @verbatim
 * For example data:
 *   4,5,1,2
 *   3,5,1
 *   3,2
 * Will be convert to the form of:
 * row offset: 0,4,7,9
 * value: 4,5,1,2,3,5,1,3,2
 * @endverbatim
 */
template <typename T>
class CSR {
 private:

 int *unduplicated_key_num;
 PinnedBuffer<T>  unduplicated_key; 
  PinnedBuffer<T> row_offset_value_buffer_; /**< a unified buffer for row offset and value. */
  T* row_offset_;   /**< just offset on the buffer, note that the length of it is slot*batchsize+1.
                     */
  T* value_;        /**< pointer of value buffer. */
  int num_rows_{0}; /**< num rows. */
  int size_of_value_{0};      /**< num of values in this CSR buffer */
  int size_of_row_offset_{0}; /**< num of rows in this CSR buffer */
  const int max_value_size_;  /**< number of element of value the CSR matrix will have for num_rows
                                 rows. */
  int check_point_row_;       /**< check point of size_of_row_offset_. */
  int check_point_value_;     /**< check point of size_of_value__. */
 public:
  /**
   * Ctor
   * @param num_rows num of rows is expected
   * @param max_value_size max size of value buffer.
   */
  CSR(int num_rows, int max_value_size)
      : row_offset_value_buffer_(num_rows + 1 + max_value_size),
        row_offset_(row_offset_value_buffer_.get()),
        value_(row_offset_value_buffer_.get() + num_rows + 1),
        num_rows_(num_rows),
        max_value_size_(max_value_size) ,
        unduplicated_key(max_value_size){
           unduplicated_key_num=(int*)malloc(sizeof(int));
          *unduplicated_key_num=0;
    static_assert(std::is_same<T, long long>::value || std::is_same<T, unsigned int>::value,
                  "type not support");
    //    std::cout << "num_rows: " << num_rows << ";max_value_size: " << max_value_size <<
    //    std::endl;
    if (max_value_size <= 0 && num_rows <= 0) {
      CK_THROW_(Error_t::WrongInput, "max_value_size <= 0 && num_rows <= 0");
    }
  }
  CSR(const CSR&) = delete;
  CSR& operator=(const CSR&) = delete;
  CSR(CSR&&) = default;

  /**
   * Push back a value to this object.
   * @param value the value to be pushed back.
   */
  void push_back(const T& value) {
    if (size_of_value_ >= max_value_size_)
      CK_THROW_(Error_t::OutOfBound, "CSR out of bound " + std::to_string(max_value_size_) +
                                         "offset" + std::to_string(size_of_value_));
    if(size_of_value_>0){
      int index=*unduplicated_key_num;
      for(int i=0;i<index;i++){
        if(value==unduplicated_key[i]){
          index=i;
          break;
        }
     }
     if(index==*unduplicated_key_num){
       unduplicated_key[index]=value;
       *unduplicated_key_num=index+1;
     }
      value_[size_of_value_]=index;
       size_of_value_++;

    }else{
      unduplicated_key[0]=value;
      value_[0]=0;
      *unduplicated_key_num=1;
      size_of_value_++;
    }
  }

  /**
   * Insert a new row to CSR
   * Whenever you want to add a new row, you need to call this.
   * When you have pushed back all the values, you need to call this method
   * again.
   */
  void new_row() {  // call before push_back values in this line
    if (size_of_row_offset_ > num_rows_) CK_THROW_(Error_t::OutOfBound, "CSR out of bound");
    row_offset_[size_of_row_offset_] = static_cast<T>(size_of_value_);
    size_of_row_offset_++;
  }

  /**
   * Set check point.
   */
  void set_check_point() {
    check_point_row_ = size_of_row_offset_;
    check_point_value_ = size_of_value_;
  }
  /**
   * Give up current row.
   */
  void roll_back() {
    size_of_row_offset_ = check_point_row_;
    size_of_value_ = check_point_value_;
  }

  /**
   * To reset the CSR buffer.
   * You need to call reset when you want to overwrite the origial data in this CSR.
   */
  void reset() {
    size_of_value_ = 0;
    size_of_row_offset_ = 0;
  }
  const T* get_undu_key() const { return unduplicated_key.get(); }
  const T* get_row_offset() const { return row_offset_; }
  const T* get_value() const { return value_; }
  void reset_undu_key_num() const {*unduplicated_key_num=0;}
  int get_undu_key_num() const{return *unduplicated_key_num;}
  int get_sizeof_value() const { return size_of_value_; }
  int get_num_rows() const { return num_rows_; }
  int get_max_value_size() const { return max_value_size_; }
  const T* get_buffer() const { return row_offset_value_buffer_.get(); }
};

}  // namespace HugeCTR
