# 2D linear diffusion solver - GPU version using KernelAbstractions.jl
#
# Unlike the plain CUDA.jl version, this kernel is backend-agnostic: the very
# same `diffusion_step_kernel!` runs on NVIDIA GPUs (CUDA.jl), AMD GPUs
# (AMDGPU.jl), Apple GPUs (Metal.jl), or the CPU -- whichever backend the
# input arrays live on. Only the backend package loaded below and the
# `ArrayType` passed to `run_diffusion` need to change.
using Printf
using JLD2
using KernelAbstractions
using CUDA
# using Metal
using Plots
include(joinpath(@__DIR__, "shared.jl"))

# convenience macros simply to avoid writing nested finite-difference expression
macro qx(ix, iy) esc(:(-D * (C[$ix+1, $iy] - C[$ix, $iy]) * inv(dx))) end
macro qy(ix, iy) esc(:(-D * (C[$ix, $iy+1] - C[$ix, $iy]) * inv(dy))) end

#
# KernelAbstractions kernel: written once, compiled for whatever backend the
# arrays live on. `@index(Global, NTuple)` gives the (ix, iy) position of the
# work item, analogous to the (blockIdx, threadIdx) arithmetic in the CUDA
# version. Scalars are passed explicitly (no NamedTuple) so the kernel body
# stays fully bits-typed and portable across backends.
#
@kernel function diffusion_step_kernel!(C2, @Const(C), dx, dy, dt, D)
    ix, iy = @index(Global, NTuple)
    if ix <= size(C, 1) - 2 && iy <= size(C, 2) - 2
        @inbounds C2[ix+1, iy+1] = C[ix+1, iy+1] - dt * ((@qx(ix + 1, iy + 1) - @qx(ix, iy + 1)) * inv(dx) +
                                                          (@qy(ix + 1, iy + 1) - @qy(ix + 1, iy)) * inv(dy))
    end
end

function diffusion_step!(params, C2, C)
    (; dx, dy, dt, D, workgroupsize) = params
    backend = KernelAbstractions.get_backend(C)
    kernel! = diffusion_step_kernel!(backend, workgroupsize)
    kernel!(C2, C, dx, dy, dt, D; ndrange=(size(C, 1) - 2, size(C, 2) - 2))
    return nothing
end

function run_diffusion(; ns=128, nt=ns^2÷40, do_visualize=false, ArrayType=Array, dtype=Float32)
    params   = init_params_gpu(; ns, nt, do_visualize, dtype)
    C, C2    = init_arrays(params)

    # Move C and C2 onto whichever device ArrayType represents
    # (CuArray, ROCArray, MtlArray, or plain Array for the CPU backend).
    C  = ArrayType(C)
    C2 = ArrayType(C2)

    maybe_visualize(params, C)
    t_tic    = 0.0
    backend  = KernelAbstractions.get_backend(C)
    # Time loop
    for it in 1:nt
        # time after warmup (ignore first 10 iterations)
        (it == 11) && (t_tic = Base.time())
        # diffusion
        diffusion_step!(params, C2, C)
        C, C2 = C2, C # pointer swap
        # visualization
        maybe_visualize(params, C, it)
    end
    # synchronize the device before querying the final time
    KernelAbstractions.synchronize(backend)
    t_toc = (Base.time() - t_tic)
    print_perf(params, t_toc)
    return nothing
end

# maps a CLI device string to the corresponding array type
const DEVICE_ARRAYTYPES = Dict(
    "cpu"   => Array,
    "cuda"  => CuArray,
    # "metal" => MtlArray,
)

# Running things...

# enable saving by default
(!@isdefined do_visualize) && (do_visualize = true)
# enable execution by default
(!@isdefined do_run) && (do_run = true)

if do_run
    if !isempty(ARGS)
        ns     = parse(Int, ARGS[1])
        device = length(ARGS) >= 2 ? ARGS[2] : "cpu"
        haskey(DEVICE_ARRAYTYPES, device) ||
            error("Unknown device \"$device\": choose one of $(join(keys(DEVICE_ARRAYTYPES), ", ")).")
        run_diffusion(; ns, nt=500, do_visualize=false, ArrayType=DEVICE_ARRAYTYPES[device])
    else
        run_diffusion(; do_visualize)
    end
end
