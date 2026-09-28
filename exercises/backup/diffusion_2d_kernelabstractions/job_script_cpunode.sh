#!/bin/bash
#PBS -N diff2dgpu
#PBS -l select=1:node_type=skl:ncpus=10:mem=10gb
#PBS -l walltime=00:10:00
#PBS -q smp
#PBS -j oe
#PBS -o job_script_cpunode.out

WORKDIR=$(pwd)
if [[ -n "${PBS_O_WORKDIR}" ]]; then
    # we're running as a cluster job
    # change to the directory that the job was submitted from ...
    WORKDIR=$PBS_O_WORKDIR
    # ... and load the module(s)
    ml juliahpc
fi
cd $WORKDIR

for i in 256 512 1024 2048 4096 8192 16384
do
    echo -e "\n\n#### Run ns=$i"

    for backend in cpu
    do
        echo -e "\n#### Backend: $backend"
        julia --project --threads=10 diffusion_2d_ka.jl $i $backend
    done
done
