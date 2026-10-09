# =============================================================================
# Save and load a solvers convergence information 
# (the `convergence` group in the HDF5 file).
# =============================================================================

using HDF5
using Printf

"""
    save_convergence!(f, result; tol, maxiter)

Save convergence information, in the same file as the solver.

`result` is what the solve returns. This picks out the convergence information
(converged or not, final residual, number of iterations) and writes it with
[`write_convergence!`](@ref).
"""
function save_convergence!(f, result; tol, maxiter)
    res = result.res
    write_convergence!(f, (; converged   = res.f_converged || res.x_converged,
                             res.f_converged, res.x_converged, res.residual_norm,
                             res.iterations, res.f_calls, tol, maxiter))
    return nothing
end

"""
    write_convergence!(f, conv)

Write the convergence information `conv` to the file `f`.

Kept separate from `save_convergence!` because the precompute cache has no solver
result: it copies `conv` from [`load_convergence`](@ref) and writes it here.
True/false values are stored as 1/0 so Python (h5py) can read them too.
"""
function write_convergence!(f, conv)
    g = create_group(f, "convergence")
    g["converged"]     = Int(conv.converged)
    g["f_converged"]   = Int(conv.f_converged)
    g["x_converged"]   = Int(conv.x_converged)
    g["residual_norm"] = conv.residual_norm
    g["iterations"]    = conv.iterations
    g["f_calls"]       = conv.f_calls
    g["tol"]           = conv.tol
    g["maxiter"]       = conv.maxiter
    return nothing
end

"""
    load_convergence(path) -> NamedTuple
    load_convergence(U, T, nmax, nq, NL_method, strategy, iteration_scheme) -> NamedTuple

Read the convergence information from a solver file or a precompute cache,
without loading the (large) solver itself. The file is given by its path or by
the run settings.

Returns `(; converged, f_converged, x_converged, residual_norm, iterations,
f_calls, tol, maxiter)`. If the information is missing (e.g. the save did not
finish), this throws an error instead of returning a made-up value.
"""
function load_convergence(path::AbstractString)
    isfile(path) || error("No saved run at $path.")
    return h5open(path, "r") do f
        g = f["convergence"]
        num(key) = read(g[key])     # read one value from the group

        (; converged     = num("converged")   == 1, # stored as 1/0;
           f_converged   = num("f_converged") == 1, # `== 1` turns it back
           x_converged   = num("x_converged") == 1, # into true/false
           residual_norm = Float64(num("residual_norm")),
           iterations    = Int(num("iterations")),
           f_calls       = Int(num("f_calls")),
           tol           = Float64(num("tol")),
           maxiter       = Int(num("maxiter")))
    end
end

# Same, but find the file from the 7 run settings.
load_convergence(a::Vararg{Any,7}) =load_convergence(solver_path(a...))

"""
    report_convergence(conv) -> nothing

Print the convergence information in one line.
"""
function report_convergence(conv::NamedTuple)
    @printf "Convergence: %s  (residual = %.3e, iterations = %d/%d, tol = %.1e)\n" (
        conv.converged ? "CONVERGED" : "NOT converged") conv.residual_norm conv.iterations conv.maxiter conv.tol
    return nothing
end
