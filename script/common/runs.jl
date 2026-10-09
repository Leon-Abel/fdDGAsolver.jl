# =============================================================================
# Build, solve and save one run.
# =============================================================================

using fdDGAsolver
using MatsubaraFunctions
using StaticArrays
using HDF5
using Printf

# -----------------------------------------------------------------------------
# Building the solver
# -----------------------------------------------------------------------------
# Constructors for NL_method = 1, 2, 3, so `PA_SOLVERS[NL_method]` picks the right one.
const PA_SOLVERS  = (parquet_solver_hubbard_parquet_approximation,
                     parquet_solver_hubbard_parquet_approximation_NL2,
                     parquet_solver_hubbard_parquet_approximation_NL3)
const NL_VERTICES = (NL_Vertex, NL2_Vertex, NL3_Vertex)
const NL_SOLVERS  = (NL_ParquetSolver, NL2_ParquetSolver, NL3_ParquetSolver)

"""
    box_sizes(nmax) -> NamedTuple

Frequency box sizes, all set by `nmax`: `nG = nK1 = 4·nmax` and
`nK2 = nK3 = (nmax, nmax)`.
"""
box_sizes(nmax) = (; nG = 4nmax, nK1 = 4nmax, nK2 = (nmax, nmax), nK3 = (nmax, nmax))

"""
    momentum_meshes(nq; nk_G = 48) -> NamedTuple

The k-meshes: a fine one (`nk_G`) for G and Σ, and a coarse one (`nq`) for the
vertex (K1, K2, K3).
"""
function momentum_meshes(nq; nk_G = 48)
    k1_vec = 2π * SVector(1., 0.)
    k2_vec = 2π * SVector(0., 1.)
    return (; mk_G = BrillouinZoneMesh(BrillouinZone(nk_G, k1_vec, k2_vec)),
              mk_F = BrillouinZoneMesh(BrillouinZone(nq,   k1_vec, k2_vec)))
end

"""
    build_solver(nmax, nq, NL_method; strategy, T, U, μ, t1, t2, t3, Γ_dmft, nk_G, mode)

Build the solver for one run.

`strategy`:
  `:scPA`  bare vertex as the reference F0 — the parquet approximation, no DMFT input.
  `:fdPA`  the DMFT local vertex as F0 — finite-difference parquet DΓA.

`NL_method` picks the momentum structure of K2/K3:
  1  K2(q,k) ≈ K2(q) and K3(q,k,k') ≈ K3(q)
  2  K3(q,k,k') ≈ K3(q)
  3  full K2 and K3 momentum dependence

`:fdPA` needs the DMFT vertex `Γ_dmft`.

Call `init_sym_grp!` on the returned solver before solving it.
"""
function build_solver(nmax, nq, NL_method; strategy = :fdPA,
                      T = 0.5, U = 1.0, μ = 0.0, t1 = 1.0, t2 = 0.0, t3 = 0.0,
                      Γ_dmft = nothing, G_dmft = nothing, Σ_dmft = nothing,
                      nk_G = 48, mode = :hybrid)
    check_NL(NL_method)
    strategy in STRATEGIES || error("strategy must be one of $(STRATEGIES), got :$strategy.")

    (; nG, nK1, nK2, nK3) = box_sizes(nmax)
    (; mk_G, mk_F)        = momentum_meshes(nq; nk_G)

    if strategy === :scPA
        return PA_SOLVERS[NL_method](nG, nK1, nK2, nK3, mk_G, mk_F;
                                     T, U, μ, t1, t2, t3, mode)
    end

    # --- :fdPA: DMFT vertex as the reference ---------------------------------
    Gbare = hubbard_bare_Green(MatsubaraMesh(T, nG, Fermion), mk_G; μ, t1, t2, t3)

    # Reference G0 and Σ0: the local DMFT values if given, otherwise zero.
    G0 = copy(Gbare); set!(G0, 0)
    Σ0 = copy(Gbare); set!(Σ0, 0)
    if G_dmft !== nothing && Σ_dmft !== nothing
        for ν in meshes(G0, Val(1))
            view(G0, ν, :) .= G_dmft(value(ν))
            view(Σ0, ν, :) .= Σ_dmft(value(ν))
        end
    end

    Γ_dmft === nothing && error("strategy = :fdPA needs the DMFT vertex `Γ_dmft`.")
    F0    = NL_VERTICES[NL_method](Γ_dmft, T, nK1, nK2, nK3, mk_F)
    return NL_SOLVERS[NL_method](nK1, nK2, nK3, mk_F, Gbare, G0, Σ0, F0; mode)
end

"""
    build_solver_from_dmft(data_triqs, nmax, nq, NL_method; strategy, nk_G, mode)

Like [`build_solver`](@ref), but takes all inputs from loaded DMFT data
`data_triqs`: model parameters, and for `:fdPA` also the vertex, G and Σ.
"""
function build_solver_from_dmft(data_triqs, nmax, nq, NL_method;
                                strategy = :fdPA, nk_G = 48, mode = :hybrid)
    (; T, U, μ, t1, t2, t3) = data_triqs.params
    return build_solver(nmax, nq, NL_method;
        strategy, T, U, μ, t1, t2, t3,
        Γ_dmft = strategy === :fdPA ? data_triqs.Γ : nothing,
        G_dmft = strategy === :fdPA ? data_triqs.G : nothing,
        Σ_dmft = strategy === :fdPA ? data_triqs.Σ : nothing,
        nk_G, mode)
end

# -----------------------------------------------------------------------------
# Solving
# -----------------------------------------------------------------------------

"""
    solve_run!(S; strategy, iteration_scheme, maxiter, tol, update_Σ, verbose, kwargs...)

Solve `S` with the chosen `iteration_scheme`: `:anderson` (Anderson acceleration)
or `:mfRG` (mfRG-preconditioned). Returns `(; res, history)` either way.

`update_Σ = true` is passed to both, so both schemes solve the same equations
(their own defaults differ).

Extra keywords such as `mem` and `δ` are passed on; they only matter for `:anderson`.
"""
function solve_run!(S; strategy, iteration_scheme, maxiter, tol,
                    update_Σ = true, verbose = mpi_ismain(), kwargs...)
    check_run(strategy, iteration_scheme)
    # Written as `fdDGAsolver.solve!` because another package also defines `solve!`.
    solve_fn = iteration_scheme === :mfRG ? fdDGAsolver.solve_using_mfRG_without_mixing! :
                                            fdDGAsolver.solve!
    return solve_fn(S; strategy, maxiter, tol, update_Σ, verbose, kwargs...)
end

# -----------------------------------------------------------------------------
# Saving
# -----------------------------------------------------------------------------
# The convergence information itself is handled in convergence.jl.

"""
    save_run!(path, S, result; tol, maxiter) -> path

Save the solver and its convergence information to `path`, creating the folder
if needed. Call this from the main MPI process only.
"""
function save_run!(path, S, result; tol, maxiter)
    mkpath(dirname(path))
    h5open(path, "w") do f
        save!(f, "S", S)
        save_convergence!(f, result; tol, maxiter)
    end
    return path
end
