# =============================================================================
# One iteration of the parquet equations, and the plain fixed-point solver.
#
# The unknowns are the fd channels Φ̃_r of the target vertex and the target
# self-energy Σ. One sweep (iterate_solver!) evaluates the right-hand sides of
# all equations with the current unknowns:
#   1. Dyson equation G = (Gbare⁻¹ - Σ)⁻¹ and the bubbles Π (if Σ is updated),
#   2. the K3 caches (build_K3_cache.jl),
#   3. the left part FL = Ĩ_r π_r f of the BSE (strategy :fdPA),
#   4. the BSEs for K1, K2, K3 in the channel order p, a, t (Lihm 2025,
#      Eq. (S29), mirrored), written to the buffer S.Fbuff,
#   5. F ← Fbuff,
#   6. the Schwinger-Dyson equation for Σ (SDE.jl).
# All BSEs of one sweep read the old F; the order p, a, t is needed because
# the t channel uses the new a channel (BSE_templates.jl).
#
# Writing all unknowns as one vector x, a sweep is a map x ↦ G(x), and the
# solution is its fixed point G(x) = x. solve! finds it with Anderson
# acceleration (NLsolve), using the residual G(x) - x (Lihm 2025, Eq. (S68)).
# The preconditioned iteration of Eq. (S74) is in mfRG.jl.
#
# Strategies (`strategy`):
#   :scPA       no left part; Σ = SDE[F](G)
#   :fdPA       full fd-BSE with the left part; fd form of the SDE
#   :fdPA_new   K1, K2 from the U-closed forms, K3 as in :fdPA
#   :scPA_new   the same without the left part
#   :fdPA_1loop only the one-loop term F Π̃ f (lattice solvers only)
# With the bare reference (g = 0, f = U) the fd strategies reduce to the plain
# parquet approximation.
# =============================================================================

# -----------------------------------------------------------------------------
# One sweep
# -----------------------------------------------------------------------------

"""
    iterate_solver!(S; strategy = :fdPA, update_Σ = true, compute_Hartree = true) -> nothing

One sweep of the parquet equations (see the file header): update G and the
bubbles, solve the BSEs of all channels and classes with the current vertex,
store the result in `S.F`, and update `S.Σ` from the SDE.

- `update_Σ = false`: keep Σ, G and the bubbles fixed and only update the
  vertex.
- `compute_Hartree`: whether the SDE adds the Hartree term. In a DΓA
  calculation whose Σ0 already includes the Hartree term, set this to `false`.
"""
function iterate_solver!(S :: AbstractSolver;
    strategy :: Symbol = :fdPA,
    update_Σ :: Bool = true,
    compute_Hartree :: Bool = true,
    ) ::Nothing

    @assert strategy in (:fdPA, :scPA, :scPA_new, :fdPA_new, :fdPA_1loop) "Calculation strategy unknown"

    if update_Σ
        # If Σ is not updated, G and bubbles are already set at the initialization step,
        # so they don't need to be updated here.

        # update G
        Dyson!(S)

        # update bubbles
        bubbles!(S)
    end

    # vertex slices for the K3 BSEs
    build_K3_cache!(S)


    if strategy == :fdPA_new || strategy == :scPA_new
        if strategy == :fdPA_new
            # calculate FL (K3 class only; the U-closed K1, K2 do not need it)
            BSE_L_K3!(S, pCh)
            BSE_L_K3!(S, aCh)
            BSE_L_K3!(S, tCh)
        end

        BSE_K3!(S, pCh)
        BSE_K3!(S, aCh)
        BSE_K3!(S, tCh)

        # K1 and K2 from the U-closed forms
        BSE_K1_new!(S, pCh)
        BSE_K1_new!(S, aCh)
        BSE_K1_new!(S, tCh)

        BSE_K2_new!(S, pCh)
        BSE_K2_new!(S, aCh)
        BSE_K2_new!(S, tCh)

    elseif strategy == :fdPA_1loop
        # one-loop term F Π̃ f only
        BSE_K3_1loop!(S, pCh)
        BSE_K3_1loop!(S, aCh)
        BSE_K3_1loop!(S, tCh)

        BSE_K1_1loop!(S, pCh)
        BSE_K1_1loop!(S, aCh)
        BSE_K1_1loop!(S, tCh)

        BSE_K2_1loop!(S, pCh)
        BSE_K2_1loop!(S, aCh)
        BSE_K2_1loop!(S, tCh)

    else
        if strategy == :fdPA
            # calculate FL = Ĩ π f (K2 and K3 classes)
            BSE_L_K2!(S, pCh)
            BSE_L_K2!(S, aCh)
            BSE_L_K2!(S, tCh)

            BSE_L_K3!(S, pCh)
            BSE_L_K3!(S, aCh)
            BSE_L_K3!(S, tCh)
        end

        # calculate Fbuff
        BSE_K1!(S, pCh)
        BSE_K1!(S, aCh)
        BSE_K1!(S, tCh)

        BSE_K2!(S, pCh)
        BSE_K2!(S, aCh)
        BSE_K2!(S, tCh)

        BSE_K3!(S, pCh)
        BSE_K3!(S, aCh)
        BSE_K3!(S, tCh)
    end

    # update F
    set!(S.F, S.Fbuff)

    # Not used: average over the fermionic momenta (NL3 experiments)
    # average_fermionic_momenta!(S.F.γa.K3)
    # average_fermionic_momenta!(S.F.γp.K3)
    # average_fermionic_momenta!(S.F.γt.K3)

    # average_fermionic_momenta!(S.F.γa.K2)
    # average_fermionic_momenta!(S.F.γp.K2)
    # average_fermionic_momenta!(S.F.γt.K2)

    if update_Σ
        # update self-energy
        SDE!(S; strategy, include_Hartree = compute_Hartree)

        # Old μ adjustment, hard-wired to the Wu point (now done in mfRG.jl)
        # @time for i in 1:10
        #     occ_target = 0.4798972347771976
        #     hubbard_params = (; t1 = 1.0, t2 = -0.3)
        #     μ = compute_hubbard_chemical_potential(occ_target, S.Σ, hubbard_params)
        #     set!(S.Gbare, hubbard_bare_Green(meshes(S.Gbare)...; μ, hubbard_params...))
        #     Dyson!(S)
        #     bubbles!(S)
        #     SDE!(S; strategy, include_Hartree = compute_Hartree)
        # end

        # self_energy_sanity_check(S.Σ)
    end

    return nothing

end

# -----------------------------------------------------------------------------
# Fixed-point problem
# -----------------------------------------------------------------------------

"""
    fixed_point!(R, x, S; filename_log = nothing, update_Σ = true, callback_chemical_potential_update! = nothing, kwargs...) -> nothing

The fixed-point map in residual form (Lihm 2025, Eq. (S68)): load the unknowns
`x` into the solver, do one sweep, and write `R = G(x) - x`.

- `update_Σ = true`: `x` holds the vertex channels and Σ (see `flatten(S)`).
  `update_Σ = false`: only the vertex is read from `x`.
- `callback_chemical_potential_update!(S)`: called after loading `x`, e.g. to
  re-fit μ to a target density.
- `filename_log`: if given, the solver is saved to this file after each sweep.
- Other keywords are passed on to [`iterate_solver!`](@ref).
"""
function fixed_point!(
    R :: Vector{Q},
    x :: Vector{Q},
    S :: AbstractSolver{Q}
    ;
    filename_log :: Union{Nothing, String} = nothing,
    update_Σ :: Bool = true,
    callback_chemical_potential_update! = nothing,
    kwargs_solver...
    ) :: Nothing where {Q}

    # update solver from input vector
    if update_Σ
        unflatten!(S, x)

        if callback_chemical_potential_update! !== nothing
            callback_chemical_potential_update!(S)
        end
    else
        # NOTE: solve! passes the full vector [F; Σ] (flatten(S)); with only the
        # vertex expected here, the length check in unflatten! would fail.
        unflatten!(S.F, x)
    end

    # Iterate the solver
    iterate_solver!(S; update_Σ, kwargs_solver...)

    if filename_log !== nothing
        if mpi_ismain()
            h5open(filename_log, "w") do f
                save!(f, "S", S)
            end
        end
    end

    # calculate residue R = G(x) - x
    flatten!(S, R)
    R .-= x

    return nothing
end

# -----------------------------------------------------------------------------
# Solver with Anderson acceleration
# -----------------------------------------------------------------------------

"""
    solve!(S; parallel_mode = nothing, maxiter = 100, tol = 1e-4, δ = 0.85, mem = 8, verbose = true, store_trace = true, kwargs...) -> (; res, history)

Solve the parquet equations for the solver `S` by finding the fixed point of
one sweep (`fixed_point!`), with Anderson acceleration from NLsolve (Lihm 2025,
text below Eq. (7)). The start is the current state of `S`.

- `maxiter`, `tol`: maximal number of sweeps and tolerance on the residual.
- `δ`: mixing parameter (NLsolve's `beta`); `mem`: number of previous steps
  Anderson uses.
- `parallel_mode`: overrides `S.mode` if given.
- Other keywords (e.g. `strategy`, `update_Σ`) go to `fixed_point!` and
  `iterate_solver!`.

Returns the NLsolve result `res` and the stored iteration history.
"""
function solve!(
    S::AbstractSolver,
    ;
    parallel_mode :: Union{Nothing, Symbol} = nothing,
    maxiter  :: Int64 = 100,
    tol      :: Float64 = 1e-4,
    δ        :: Float64 = 0.85,
    mem      :: Int64 = 8,
    verbose  :: Bool = true,
    store_trace :: Bool = true,
    kwargs_solver...  # passed to the iterate_solver function
    )

    if parallel_mode !== nothing
        S.mode = parallel_mode
    end

    verbose && mpi_println("Converging parquet equations.")
    verbose && mpi_println("parallelization mode : $(S.mode)")

    ti = time()
    res = nlsolve((R, x) -> fixed_point!(R, x, S; kwargs_solver...), flatten(S),
        method=:anderson,
        iterations=maxiter,
        ftol=tol,
        beta=δ,
        m=mem,
        show_trace = mpi_ismain() && verbose,
        store_trace = store_trace,
    )

    verbose && mpi_println("Done. Calculation took $(round(time() - ti, digits = 3)) seconds.")

    history = res.trace.states

    return (; res, history)
end
