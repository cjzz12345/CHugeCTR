# CHugeCTR
  CHugeCTR is a large scale distributed training framework, based on HugeCTR v2.0 of NNVIDIA. My main contribution is that I optimized the workflow and increased the training capacity by 70% of orginal HugeCTR. 
  I designed the Embedding Table both on CPU and GPU to solve the problem that Embedding Table is too big for GPU. Created Hashtable on GPU as a cache adopted a specially designed replacement strategy to store frequently used Embedding Features using CUDA language.
  I Enhanced the work-flow by adding data prefetch and reducing the redundant Embedding Features that overlap the I/O by 50%.
  My codes is mainly in embeddings, hashtable, csr and data_collector.
  In this repository, only part of the core code realted to my work is given, some other dependency and tools are not displayed. 
