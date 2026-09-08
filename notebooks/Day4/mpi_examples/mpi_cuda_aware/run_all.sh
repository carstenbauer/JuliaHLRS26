# uncomment this if you are on cluster gpu node
# ml juliahpc
# ml openmpi
# export JULIA_CUDA_MEMORY_POOL=none

for f in *.jl
do
    echo "Running $f"
    mpiexecjl --project -n 4 julia "$f"
done
