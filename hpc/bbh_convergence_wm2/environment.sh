#!/usr/bin/env bash
module load gcc/12.2.0 cuda/12.6.0 cmake/3.31.9 anaconda3/2024.10.1
export PATH=/lustre/home/2601110206/openmpi/bin:$PATH
export LD_LIBRARY_PATH=/lustre/home/2601110206/openmpi/lib:/lustre/home/2601110206/hdf5/lib:${LD_LIBRARY_PATH:-}
export OMP_NUM_THREADS=1 OMP_DYNAMIC=FALSE
export NVCC_WRAPPER_DEFAULT_COMPILER=g++
export OMPI_MCA_pml=ob1 OMPI_MCA_btl=self,smcuda,tcp OMPI_MCA_opal_cuda_support=true
export OMPI_MCA_io=romio341 OMPI_MCA_coll='^hcoll,ucc'
export OMPI_MCA_btl_tcp_if_include=eno1 PRTE_MCA_oob_tcp_if_include=eno1
unset KOKKOS_TOOLS_LIBS KOKKOS_PROFILE_LIBRARY CUDA_LAUNCH_BLOCKING KOKKOS_DEVICE_ID
