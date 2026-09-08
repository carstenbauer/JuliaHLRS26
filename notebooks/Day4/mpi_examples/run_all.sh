# uncomment this if you are on cluster gpu node
# ml juliahpc
# ml openmpi

for f in *.jl
do
    echo "Running $f"
    mpiexecjl --project -n 5 julia "$f"
done
