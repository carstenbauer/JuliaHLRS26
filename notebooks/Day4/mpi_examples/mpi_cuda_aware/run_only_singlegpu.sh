# uncomment this if you are on cluster gpu node
# ml juliahpc
# ml openmpi
# export JULIA_CUDA_MEMORY_POOL=none

mpiexecjl --project -n 4 julia 1_cudampi_singlegpu.jl
