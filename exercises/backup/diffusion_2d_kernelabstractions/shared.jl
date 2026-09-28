## PARAMETER INITIALIZATION
function init_params_gpu(; ns=64, nt=ns^2÷40, dtype=Float32, kwargs...)
    L    = dtype(10.0)                       # physical domain length
    D    = dtype(1.0)                        # diffusion coefficient
    dx   = L / ns                            # grid spacing (dtype / Int -> dtype)
    dy   = L / ns
    dt   = (dx * dy) / D / dtype(4.1)        # time step
    cs   = range(dx / 2, L - dx / 2, length=ns) .- L / 2   # already dtype end-to-end
    nout = floor(Int, nt / 5)                # plotting frequency
    return (; L, D, ns, nt, dx, dy, dt, cs, nout, dtype, kwargs...)
end

## ARRAY INITIALIZATION
function init_arrays(params)
    (; cs) = params
    C  = @. exp(-cs^2 - (cs')^2)
    C2 = copy(C)
    return C, C2
end

## VISUALIZATION & PRINTING
function maybe_visualize(params, C, it=0)
    if params.do_visualize && (it % params.nout == 0)
        p = it ÷ params.nout
        plt = Plots.heatmap(params.cs, params.cs, Array(C); clims=(0,1), c=:turbo, dpi=300)
        isinteractive() && display(plt)
        savefig(plt, "visualization_$p.png")
    end
    return nothing
end

function print_perf(params, t_toc)
    (; ns, nt, dtype) = params
    @printf("Time = %1.4e s, T_eff = %1.2f GB/s \n", t_toc, round((2 / 1e9 * ns^2 * sizeof(dtype)) / (t_toc / (nt - 10)), sigdigits=6))
    return nothing
end
