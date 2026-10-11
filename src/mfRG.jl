# =============================================================================
# Preconditioned solution of the fd parquet equations, and older drivers that
# move the reference step by step towards the target.
#
# Iterating the fd-BSE directly converges badly: every step multiplies by
# f π_r, which stays large even when the target is close to the reference
# (Lihm 2025, text around Eq. (S73)). The fix is a preconditioned fixed-point
# iteration (Eqs. (S70), (S74)):
#   1. residual of one sweep: Ψ̃ = G(x) - x                     (Eq. (S74a))
#   2. preconditioned residual: solve the linear equation
#        Ξ̃_r = Ψ̃_r + L(Ξ̃)_r,   L(Ξ̃) = Ĩ π f + f π Ĩ + f π Ĩ π f,
#      with Ĩ_r = ∑_{s≠r} Ξ̃_s. This is the fd-BSE linearized around the
#      reference (F → f, Π → π)                                 (Eq. (S74c))
#   3. update x ← x + Ξ̃                                        (Eq. (S74d))
# Step 2 has the form of a multiloop fRG (mfRG) equation, hence the names. It
# is solved with a Krylov method (dqgmres, a GMRES variant), and Anderson
# acceleration (NLsolve) runs on top of the update. The residual after i steps
# is of order Π̃^(i+1) instead of Π^(i+1) (Eqs. (S75)-(S86)).
#
# Main entry point: solve_using_mfRG_without_mixing! (used by the scripts).
# solve_using_mfRG!, solve_using_mfRG_v2! and solve_using_mfRG_fixed_bubble!
# are older drivers. They mix the bubbles of target and reference and move
# the reference towards the target in steps; they are kept for reference.
#
# Only the lattice solvers have the reference-only K3 caches needed by the
# linear map (build_K3_cache_mfRG!, nonlocal*/build_K3_cache.jl), so this file
# works only for them.
# =============================================================================

"""
    unique_filename_log(filename_log) -> path or nothing

File name for the solver log. A name ending in `.h5` is used as it is.
Otherwise `.iter#.h5` is appended, with `#` the smallest positive integer
whose file does not exist yet.
"""
function unique_filename_log(filename_log :: Union{String, Nothing})
    # If filename_log ends with .h5, return it.
    # Otherwise, append ".iter#.h5" where "#" is the smallest positive integer that
    # the corresponding filename does not exist.
    if filename_log === nothing
        return nothing
    elseif filename_log[end-2:end] == ".h5"
        return filename_log
    else
        for i in 1:10_000
            if ! isfile("$filename_log.iter$i.h5")
                return "$filename_log.iter$i.h5"
            end
        end
        @warn "Cannot find a non-existent filename $filename_log.iter#.h5. Writing to $filename_log.tmp.h5."
        return filename_log * "tmp.h5"
    end
end

# -----------------------------------------------------------------------------
# The linear map 1 - L of the preconditioner
# -----------------------------------------------------------------------------

"""
    mfRGLinearMap(S, strategy)

The linear operator `1 - L` of the preconditioner (Lihm 2025, Eq. (S74c)),
acting on a vector of fd channels Ξ̃ (the same layout as `flatten(S.F)`):

    (1 - L)(Ξ̃) = Ξ̃ - [Ĩ π f + f π Ĩ + f π Ĩ π f],   Ĩ_r = ∑_{s≠r} Ξ̃_s

Applying it runs the BSE kernels in their `is_mfRG = Val(true)` form, which use
the reference vertex f and bubble π in place of F and Π. `strategy` must be
`:fdPA`, `:fdPA_new` or `:fdPA_1loop` (with `:fdPA_new` the K1 and K2 classes use
the U-closed forms). The field `is_first_iteration` lets the first application
build the parts of the K3 caches that do not change.

Applying the map overwrites `S.F`, `S.FL`, `S.Fbuff` and the K3 caches; the
caller loads the vertex again before the next sweep.
"""
mutable struct mfRGLinearMap{ST, Q} <: LinearMaps.LinearMap{Q}
    const S :: ST
    strategy :: Symbol
    is_first_iteration :: Bool
    function mfRGLinearMap(S :: ST, strategy) where {ST <: AbstractSolver{Q}} where {Q}
        if strategy ∉ [:fdPA, :fdPA_new, :fdPA_1loop]
            error("Invalid strategy $strategy. Must be fdPA or fdPA_new or fdPA_1loop.")
        end
        new{ST, Q}(S, strategy, true)
    end
end

# The map acts on the fd channels of the vertex, so it is square with that size.
Base.size(A::mfRGLinearMap) = (length(A.S.F), length(A.S.F))

# Apply the map: y = (1 - L) x, see the docstring of mfRGLinearMap.
function LinearMaps._unsafe_mul!(y, A::mfRGLinearMap, x::AbstractVector)
    S = A.S

    # Load factor * x as the fd channels. For a plain vertex L is linear and the
    # factor cancels below. For the MBE vertex, whose evaluation is not linear in
    # the channels, the small factor makes the map a linearization around zero.
    factor = 1e-2

    unflatten!(S.F, x .* factor)
    # (DEBUG lines: switch K1 or K2 off to test the preconditioner)
    # set!(S.F.γa.K1, 0)  # DEBUG mfRG wo K1
    # set!(S.F.γp.K1, 0)  # DEBUG mfRG wo K1
    # set!(S.F.γt.K1, 0)  # DEBUG mfRG wo K1
    # set!(S.F.γa.K2, 0)  # DEBUG mfRG wo K2
    # set!(S.F.γp.K2, 0)  # DEBUG mfRG wo K2
    # set!(S.F.γt.K2, 0)  # DEBUG mfRG wo K2

    # K3 caches with reference quantities (f instead of F)
    build_K3_cache_mfRG!(S, A.is_first_iteration)
    A.is_first_iteration = false

    if A.strategy === :fdPA || A.strategy === :fdPA_1loop
        # left part Ĩ π f, then the linearized K1, K2 BSEs
        # Takes ~40% of the time
        BSE_L_K2!(S, pCh)
        BSE_L_K2!(S, aCh)
        BSE_L_K2!(S, tCh)

        BSE_K1!(S, pCh, Val(true))
        BSE_K1!(S, aCh, Val(true))
        BSE_K1!(S, tCh, Val(true))

        # Takes ~60% of the time
        BSE_K2!(S, pCh, Val(true))
        BSE_K2!(S, aCh, Val(true))
        BSE_K2!(S, tCh, Val(true))
    elseif A.strategy === :fdPA_new
        # New version: K1 and K2 from the U-closed forms
        BSE_K1_new!(S, pCh, Val(true))  # DEBUG mfRG wo K1
        BSE_K1_new!(S, aCh, Val(true))  # DEBUG mfRG wo K1
        BSE_K1_new!(S, tCh, Val(true))  # DEBUG mfRG wo K1

        BSE_K2_new!(S, pCh, Val(true))  # DEBUG mfRG wo K2
        BSE_K2_new!(S, aCh, Val(true))  # DEBUG mfRG wo K2
        BSE_K2_new!(S, tCh, Val(true))  # DEBUG mfRG wo K2
    end

    # left part and linearized BSE of the K3 class
    BSE_L_K3!(S, pCh)
    BSE_L_K3!(S, aCh)
    BSE_L_K3!(S, tCh)

    BSE_K3!(S, pCh, Val(true))
    BSE_K3!(S, aCh, Val(true))
    BSE_K3!(S, tCh, Val(true))

    set!(S.F, S.Fbuff)

    # y = x - L(x), undoing the factor
    flatten!(S.F, y)
    y .= x .- y ./ factor

    return y
end

# -----------------------------------------------------------------------------
# Preconditioned fixed-point step
# -----------------------------------------------------------------------------

"""
    fixed_point_preconditioned!(R, x, S, krylov_solver = nothing; strategy, update_Σ = false, occ_target = nothing, hubbard_params = nothing, filename_log = nothing, use_preconditioner = true, krylov_maxiter = 400, callback_after_iterate = nothing, kwargs...) -> nothing

One preconditioned fixed-point step (Lihm 2025, Eq. (S74)) in residual form,
for use with Anderson acceleration:

1. Load the unknowns `x` (the fd channels, and Σ if `update_Σ`) into `S`. If
   `occ_target` is given, re-fit μ so that the density stays at `occ_target`
   (Lihm 2025, text below Eq. (7)); `hubbard_params` are then needed.
2. Impose the symmetries and do one sweep (`iterate_solver!`).
3. Vertex residual `Ψ̃ = G(x) - x` (Eq. (S74a)). With `use_preconditioner`, solve
   `(1 - L) Ξ̃ = Ψ̃` with the Krylov method dqgmres (Eq. (S74c)) and return
   `R = Ξ̃`; otherwise `R = Ψ̃`.
4. The Σ part of the residual, if present, is not preconditioned.

`krylov_solver` is a reusable dqgmres workspace. `callback_after_iterate(S)`
is called after the sweep, and the solver is logged to `filename_log` if given.
"""
function fixed_point_preconditioned!(
    R :: Vector{Q},
    x :: Vector{Q},
    S :: AbstractSolver{Q},
    krylov_solver = nothing,
    ;
    strategy,
    update_Σ = false,
    occ_target = nothing,
    hubbard_params = nothing,
    filename_log :: Union{Nothing, String} = nothing,
    use_preconditioner = true,
    krylov_maxiter = 400,
    callback_after_iterate = nothing,
    kwargs_solver...
    ) :: Nothing where {Q}

    # update solver from input vector
    if update_Σ
        unflatten!(S, x)

        if occ_target !== nothing
            # Update chemical potential to fix the occupation
            μ = compute_hubbard_chemical_potential(occ_target, S.Σ, hubbard_params)
            set!(S.Gbare, hubbard_bare_Green(meshes(S.Gbare)...; μ, hubbard_params...))
        end
    else
        unflatten!(S.F, x)
    end
    symmetrize_solver!(S; verbose = false)

    # Iterate the solver
    iterate_solver!(S; strategy, update_Σ, kwargs_solver...)

    if callback_after_iterate !== nothing
        callback_after_iterate(S)
    end

    if filename_log !== nothing
        if mpi_ismain()
            h5open(unique_filename_log(filename_log), "w") do f
                save!(f, "S", S)
            end
        end
    end

    # calculate residue of the vertex: Ψ̃ = G(x) - x (Eq. (S74a))
    R_F = flatten(S.F) .- x[1:length(S.F)]

    # Precondition output: solve (1 - L) Ξ̃ = Ψ̃ (Eq. (S74c))
    if use_preconditioner
        if krylov_solver === nothing
            xsol, stats = Krylov.dqgmres(mfRGLinearMap(S, strategy), R_F;
                atol = 1e-6, rtol = 1e-6, itmax = krylov_maxiter,
                memory = 100, verbose = 0, history = true);
        else
            # In-place solver
            res = Krylov.dqgmres!(krylov_solver, mfRGLinearMap(S, strategy), R_F;
                atol = 1e-6, rtol = 1e-6, itmax = krylov_maxiter,
                verbose = 0, history = true);
            xsol = res.x
            stats = res.stats
        end

        if !stats.solved
            @warn "Krylov solver did not converge after $(stats.niter) iterations, residual $(stats.residuals[end])."
        end
        flush(stdout)
        flush(stderr)

        R[1:length(S.F)] .= xsol
    else
        R[1:length(S.F)] .= R_F
    end

    # self-energy part of the residual (not preconditioned)
    if update_Σ
        R[length(S.F)+1:end] .= flatten(S.Σ) .- x[length(S.F)+1:end]
    end

    return nothing
end

# -----------------------------------------------------------------------------
# Self-energy at fixed vertex (used by the older drivers)
# -----------------------------------------------------------------------------

"""
    iterate_solver_self_energy!(S; strategy, occ_target = nothing, hubbard_params = nothing)

Update the self-energy at fixed vertex: re-fit μ if `occ_target` is given,
then Dyson equation, bubbles and SDE.
"""
function iterate_solver_self_energy!(S;
    strategy,
    occ_target = nothing,
    hubbard_params = nothing,
    )

    if occ_target !== nothing
        # Update chemical potential to fix the occupation
        μ = compute_hubbard_chemical_potential(occ_target, S.Σ, hubbard_params)
        mpi_ismain() && println("Updated μ = $μ")
        set!(S.Gbare, hubbard_bare_Green(meshes(S.Gbare)...; μ, hubbard_params...))
    end

    Dyson!(S)
    bubbles!(S)
    SDE!(S; strategy)
end

"""
    fixed_point_self_energy!(R, x, S; filename_log = nothing, kwargs...) -> nothing

Fixed-point residual for the self-energy alone: load Σ from `x`, update it with
[`iterate_solver_self_energy!`](@ref), and write `R = Σ_new - x`. Not called
anywhere.
"""
function fixed_point_self_energy!(
    R :: Vector{Q},
    x :: Vector{Q},
    S :: AbstractSolver{Q}
    ;
    filename_log :: Union{Nothing, String} = nothing,
    kwargs_solver...
    ) :: Nothing where {Q}

    # update self-energy from input vector
    unflatten!(S.Σ, x)

    # Iterate the solver
    iterate_solver_self_energy!(S; kwargs_solver...)

    # calculate residue
    flatten!(S.Σ, R)
    R .-= x

    return nothing
end

# -----------------------------------------------------------------------------
# Older driver: continuation with mixed bubbles
# -----------------------------------------------------------------------------

"""
    solve_using_mfRG!(S; filename_log = nothing, maxiter = 100, verbose = true, occ_target = nothing, hubbard_params = nothing, mixing_init = 1.0, iter_restart = 0, tol = 1e-4, auto_restart = false, _debug_single_iter = false, strategy = :fdPA, update_Σ = false)

Older driver (lattice solvers only). Move the reference step by step towards
the target, with adaptive mixing of the bubbles:

1. Use the mixed bubble `Π_mix = mixing · Π(G) + (1 - mixing) · π` and
   converge the vertex with the preconditioned iteration.
2. If that fails, halve `mixing` and retry; if it works, increase `mixing` by
   20% (at most 1).
3. Update Σ with the plain SDE, then let the reference absorb the result:
   `F0 += F` (the fd channels move into the reference), `G0 ← G`, `Σ0 ← Σ`,
   `π ← Π_mix`, and reset the target channels to zero.
4. Re-fit μ if `occ_target` is given, update G and Π, and repeat until Σ
   changes by less than `tol` (relative to `mixing`).

Each step is saved to `filename_log.iter#.h5`; `iter_restart` or `auto_restart`
continue from such a file.
"""
function solve_using_mfRG!(
    S :: AbstractSolver
    ;
    filename_log = nothing,
    maxiter = 100,
    verbose = true,
    occ_target = nothing,
    hubbard_params = nothing,
    mixing_init = 1.0,
    iter_restart = 0,
    tol = 1e-4,
    auto_restart = false,
    _debug_single_iter = false,
    strategy = :fdPA,
    update_Σ = false,
    )

    if auto_restart && filename_log !== nothing
        iter = findlast(i -> isfile("$filename_log.iter$i.h5"), 1:100)
        if iter !== nothing
            iter_restart = iter
        end
    end

    if iter_restart !== 0
        # Restart from file
        iter = iter_restart
        filename = "$filename_log.iter$iter_restart.h5"
        mpi_ismain() && println("Restarting from $filename")
        f = h5open(filename, "r")
        set!(S.Gbare,  load_mesh_function(f, "Gbare"))
        set!(S.G,  load_mesh_function(f, "G"))
        set!(S.Σ,  load_mesh_function(f, "Σ"))
        set!(S.G0, load_mesh_function(f, "G0"))
        set!(S.Σ0, load_mesh_function(f, "Σ0"))
        set!(S.Π0pp, load_mesh_function(f, "Π0pp"))
        set!(S.Π0ph, load_mesh_function(f, "Π0ph"))
        set!(S.Πpp, load_mesh_function(f, "Πpp"))
        set!(S.Πph, load_mesh_function(f, "Πph"))
        set!(S.F,  load_vertex(NL2_Vertex, f, "F"))
        if S.F0 isa NL2_Vertex
            set!(S.F0, load_vertex(NL2_Vertex, f, "F0"))
        elseif S.F0 isa NL2_MBEVertex
            set!(S.F0, load_vertex(NL2_MBEVertex, f, "F0"))
        else
            error("Wrong type of S.F0")
        end
        mixing = read(f, "mixing")
        close(f)
    else
        mixing = mixing_init
        iter = 0
    end


    for iter_ in 1:maxiter
        iter += 1

        # Mix bubble
        # Here, S.Π are the unmixed bubble, and S.Π0 is the mixed bubble from the previous iteration.
        Πpp_mixed = S.Πpp * mixing + S.Π0pp * (1 - mixing)
        Πph_mixed = S.Πph * mixing + S.Π0ph * (1 - mixing)
        set!(S.Πpp, Πpp_mixed)
        set!(S.Πph, Πph_mixed)

        if mpi_ismain() && verbose
            println(" === Iteration $iter (including retry: $iter_) ===")
            println("Mix Π and Π0 by $mixing")
            println("max|Πpp - Π0pp| = $(norm((S.Πpp - S.Π0pp).data))")
            println("max|Πph - Π0ph| = $(norm((S.Πph - S.Π0ph).data))")
        end

        # Solve vertex
        if mpi_ismain() && verbose
            println("Solving vertex")
        end
        @time res = nlsolve((R, x) -> fixed_point_preconditioned!(R, x, S; update_Σ, strategy), flatten(S.F),
            method = :anderson,
            iterations = 40,
            ftol = tol,
            beta = 0.85,
            m = 50,
            show_trace = mpi_ismain() && verbose,
        );
        if _debug_single_iter
            unflatten!(S.F, res.zero);
            return
        end

        if ! res.f_converged
            # If the calculation did not converge, reduce mixing and retry.
            mixing /= 2.0
            iter -= 1
            bubbles!(S)  # Reset bubbles to the unmixed value

            mpi_ismain() && println("Calculation did not converge in $(res.iterations) iterations. Reduce mixing to $mixing")
            continue
        else
            # If the calculation converged, increase mixing.
            mixing = min(1.0, mixing * 1.2)
            if mpi_ismain() && verbose
                println("Calculation converged, increase mixing to $mixing")
            end
        end
        unflatten!(S.F, res.zero);

        # Restore unmixed bubble for SDE
        bubbles_real_space!(S.Π0pp, S.Π0ph, S.G0)
        bubbles_real_space!(S.Πpp, S.Πph, S.G)

        # Update self-energy
        # SDE!(S; strategy = :fdPA)
        SDE!(S; strategy = :scPA)
        Σ_err = absmax(S.Σ - S.Σ0) / mixing


        # Update reference Green function, bubble, and self-energy
        set!(S.Π0pp, Πpp_mixed)
        set!(S.Π0ph, Πph_mixed)
        set!(S.G0, S.G)
        set!(S.Σ0, S.Σ)

        # Update reference vertex and reset target vertex
        add!(S.F0, S.F)
        set!(S.F, 0)

        # Update target Green function and bubble
        if occ_target !== nothing
            # Update chemical potential to fix the occupation
            mpi_ismain() && println("Current occupation $(compute_occupation(S.G)), target $occ_target")
            μ = compute_hubbard_chemical_potential(occ_target, S.Σ, hubbard_params)
            set!(S.Gbare, hubbard_bare_Green(meshes(S.Gbare)...; μ, hubbard_params...))
        end

        Dyson!(S)
        bubbles!(S)

        flush(stdout)
        flush(stderr)

        if filename_log !== nothing
            if mpi_ismain()
                h5open(filename_log * ".iter$iter.h5", "w") do f
                    save!(f, "S", S)
                    f["mixing"] = mixing
                end
            end
        end


        if mpi_ismain()
            println("Self-energy error: $Σ_err")

            if Σ_err < tol
                break
            end
        end
    end
end





# -----------------------------------------------------------------------------
# Self-energy at fixed vertex, with a correction term (used by v2)
# -----------------------------------------------------------------------------

"""
    iterate_solver_self_energy_new!(S; strategy, occ_target = nothing, hubbard_params = nothing, Σ_corr)

Like [`iterate_solver_self_energy!`](@ref), but adds a fixed correction after
the SDE: `Σ = SDE[F](G) + Σ_corr`. With `Σ_corr = σ - SDE[f](g)` this is the
finite-difference form of the SDE (SDE.jl). This is not the local Σ correction
of Lihm 2025, Eq. (8).
"""
function iterate_solver_self_energy_new!(S;
    strategy,
    occ_target = nothing,
    hubbard_params = nothing,
    Σ_corr,
    )

    if occ_target !== nothing
        # Update chemical potential to fix the occupation
        μ = compute_hubbard_chemical_potential(occ_target, S.Σ, hubbard_params)
        mpi_ismain() && println("Updated μ = $μ")
        set!(S.Gbare, hubbard_bare_Green(meshes(S.Gbare)...; μ, hubbard_params...))
    end

    Dyson!(S)
    bubbles!(S)

    SDE!(S; strategy)
    add!(S.Σ, Σ_corr)
end

"""
    fixed_point_self_energy_new!(R, x, S; kwargs...) -> nothing

Fixed-point residual for the self-energy alone, using
[`iterate_solver_self_energy_new!`](@ref): `R = Σ_new - x`.
"""
function fixed_point_self_energy_new!(
    R :: Vector{Q},
    x :: Vector{Q},
    S :: AbstractSolver{Q}
    ;
    kwargs_solver...
    ) :: Nothing where {Q}

    # update self-energy from input vector
    unflatten!(S.Σ, x)

    # Iterate the solver
    iterate_solver_self_energy_new!(S; kwargs_solver...)

    # calculate residue
    flatten!(S.Σ, R)
    R .-= x

    return nothing
end

# -----------------------------------------------------------------------------
# Older driver: vertex at fixed bubbles, by stepwise mixing
# -----------------------------------------------------------------------------

"""
    solve_using_mfRG_fixed_bubble!(S; maxiter = 100, verbose = true, mixing_step_init = 1.0, tol = 1e-3, strategy) -> nothing

Older driver (not called anywhere). Converge the vertex for the current
target bubble Π by moving the reference bubble towards it in steps:
`Π_mix = mixing · Π + (1 - mixing) · π`, with `mixing` raised from 0 to 1. After
a failed step the step size is halved; after a converged step it grows by 20%
and the reference absorbs the solution (`F0 += F`, `π ← Π_mix`). At the end the
original bubbles are restored. Σ is not updated.
"""
function solve_using_mfRG_fixed_bubble!(
    S :: AbstractSolver
    ;
    maxiter = 100,
    verbose = true,
    mixing_step_init = 1.0,
    tol = 1e-3,
    strategy,
    )

    # Converge for given Π and Π0 by incremental mixing
    Πpp = copy(S.Πpp)
    Πph = copy(S.Πph)
    Π0pp = copy(S.Π0pp)
    Π0ph = copy(S.Π0ph)

    mixing_prev = 0.0
    mixing_step = mixing_step_init

    for iter in 1:maxiter

        # Mix bubble
        mixing = min(1.0, mixing_prev + mixing_step)
        set!(S.Πpp, Πpp * mixing + Π0pp * (1 - mixing))
        set!(S.Πph, Πph * mixing + Π0ph * (1 - mixing))

        if mpi_ismain() && verbose
            println(" === Iteration $iter, mixing $mixing_prev -> $mixing ===")
            println("norm|Πpp - Π0pp| = $(norm((S.Πpp - S.Π0pp).data))")
            println("norm|Πph - Π0ph| = $(norm((S.Πph - S.Π0ph).data))")
        end

        # Solve vertex
        mpi_ismain() && verbose && println("Solving vertex")

        kwargs_solver_vertex = (; update_Σ = false, strategy)
        @time res = nlsolve((R, x) -> fixed_point_preconditioned!(R, x, S; kwargs_solver_vertex...), flatten(S.F),
            method = :anderson,
            iterations = 40,
            ftol = tol,
            beta = 0.85,
            m = 100,
            show_trace = mpi_ismain() && verbose,
        );

        if ! res.f_converged
            # If the calculation did not converge, reduce mixing and retry.
            mixing = mixing_prev
            mixing_step /= 2.0
            set!(S.F, 0)

            mpi_ismain() && println("Calculation did not converge in $(res.iterations) iterations. Reduce mixing_step to $mixing_step")
            continue

        else
            # If the calculation converged, increase mixing step size.
            mixing_prev = mixing
            mixing_step *= 1.2
            mpi_ismain() && println("Calculation converged, increase mixing_step to $mixing_step")
        end

        unflatten!(S.F, res.zero);

        # Update reference vertex and bubble
        add!(S.F0, S.F)
        set!(S.Π0pp, S.Πpp)
        set!(S.Π0ph, S.Πph)

        # Reset target vertex
        set!(S.F, 0)

        if mixing == 1.0
            break
        end
    end

    # Reset bubbles to the unmixed value
    set!(S.Πpp, Πpp)
    set!(S.Πph, Πph)
    set!(S.Π0pp, Π0pp)
    set!(S.Π0ph, Π0ph)

    return nothing
end

# -----------------------------------------------------------------------------
# Older driver: continuation with an inner self-energy loop (v2)
# -----------------------------------------------------------------------------

"""
    solve_using_mfRG_v2!(S; filename_log = nothing, maxiter = 100, verbose = true, occ_target = nothing, hubbard_params = nothing, mixing_init = 1.0, iter_restart = 0, tol = 1e-4, auto_restart = false, Σ_corr)

Older driver (lattice solvers only), like [`solve_using_mfRG!`](@ref) with two
changes. Σ is converged in an inner Anderson loop at fixed vertex, as
`Σ = SDE[F](G) + Σ_corr` with re-fitted μ ([`iterate_solver_self_energy_new!`](@ref)).
And the new Σ is damped, `Σ ← (Σ + Σ0)/2`, before the reference absorbs it.
"""
function solve_using_mfRG_v2!(
    S :: AbstractSolver
    ;
    filename_log = nothing,
    maxiter = 100,
    verbose = true,
    occ_target = nothing,
    hubbard_params = nothing,
    mixing_init = 1.0,
    iter_restart = 0,
    tol = 1e-4,
    auto_restart = false,
    Σ_corr,
    )

    if auto_restart && filename_log !== nothing
        iter = findlast(i -> isfile("$filename_log.iter$i.h5"), 1:100)
        if iter !== nothing
            iter_restart = iter
        end
    end

    if iter_restart !== 0
        # Restart from file
        iter = iter_restart
        filename = "$filename_log.iter$iter_restart.h5"
        mpi_ismain() && println("Restarting from $filename")
        f = h5open(filename, "r")
        set!(S.Gbare,  load_mesh_function(f, "Gbare"))
        set!(S.G,  load_mesh_function(f, "G"))
        set!(S.Σ,  load_mesh_function(f, "Σ"))
        set!(S.G0, load_mesh_function(f, "G0"))
        set!(S.Σ0, load_mesh_function(f, "Σ0"))
        set!(S.Π0pp, load_mesh_function(f, "Π0pp"))
        set!(S.Π0ph, load_mesh_function(f, "Π0ph"))
        set!(S.Πpp, load_mesh_function(f, "Πpp"))
        set!(S.Πph, load_mesh_function(f, "Πph"))
        set!(S.F,  load_vertex(NL2_Vertex, f, "F"))
        if S.F0 isa NL2_Vertex
            set!(S.F0, load_vertex(NL2_Vertex, f, "F0"))
        elseif S.F0 isa NL2_MBEVertex
            set!(S.F0, load_vertex(NL2_MBEVertex, f, "F0"))
        else
            error("Wrong type of S.F0")
        end
        mixing = read(f, "mixing")
        close(f)
    else
        mixing = mixing_init
        iter = 0
    end


    for iter_ in 1:maxiter
        iter += 1

        # Mix bubble
        # Here, S.Π are the unmixed bubble, and S.Π0 is the mixed bubble from the previous iteration.
        Πpp_mixed = S.Πpp * mixing + S.Π0pp * (1 - mixing)
        Πph_mixed = S.Πph * mixing + S.Π0ph * (1 - mixing)
        set!(S.Πpp, Πpp_mixed)
        set!(S.Πph, Πph_mixed)

        if mpi_ismain() && verbose
            println(" === Iteration $iter (including retry: $iter_) ===")
            println("Mix Π and Π0 by $mixing")
            println("max|Πpp - Π0pp| = $(absmax(S.Πpp - S.Π0pp))")
            println("max|Πph - Π0ph| = $(absmax(S.Πph - S.Π0ph))")
        end

        # Solve vertex
        if mpi_ismain() && verbose
            println("Solving vertex")
        end
        kwargs_solver_vertex = (; update_Σ = false, strategy = :fdPA)
        @time res = nlsolve((R, x) -> fixed_point_preconditioned!(R, x, S; kwargs_solver_vertex...), flatten(S.F),
            method = :anderson,
            iterations = 200,
            ftol = tol,
            beta = 0.85,
            m = 100,
            show_trace = mpi_ismain() && verbose,
        );

        if ! res.f_converged
            # If the calculation did not converge, reduce mixing and retry.
            mixing /= 2.0
            iter -= 1
            bubbles!(S)  # Reset bubbles to the unmixed value

            mpi_ismain() && println("Calculation did not converge in $(res.iterations) iterations. Reduce mixing to $mixing")
            continue
        else
            # If the calculation converged, increase mixing.
            mixing = min(1.0, mixing * 1.2)
            mpi_ismain() && println("Calculation converged, increase mixing to $mixing")
        end
        unflatten!(S.F, res.zero);

        set!(S.Π0pp, S.Πpp)
        set!(S.Π0ph, S.Πph)

        # ---------------------------------------------------------------------
        # Update self-energy: inner Anderson loop for Σ = SDE[F](G) + Σ_corr

        verbose = true

        kwargs_solver_selfen = (; strategy = :scPA, occ_target, hubbard_params, Σ_corr)
        @time res = fdDGAsolver.nlsolve((R, x) -> fixed_point_self_energy_new!(R, x, S; kwargs_solver_selfen...), flatten(S.Σ),
            method = :anderson,
            iterations = 100,
            ftol = 1e-5,
            beta = 0.85,
            m = 10,
            show_trace = mpi_ismain() && verbose,
        );
        unflatten!(S.Σ, res.zero);
        Σ_err = absmax(S.Σ - S.Σ0) / mixing

        # Update reference vertex and reset target vertex
        add!(S.F0, S.F)
        set!(S.F, 0)

        # Update self-energy (damped by 50%)
        set!(S.Σ, S.Σ * 0.5 + S.Σ0 * 0.5)

        set!(S.Σ0, S.Σ)
        set!(S.G0, S.G)

        # Update target Green function and bubble
        if occ_target !== nothing
            # Update chemical potential to fix the occupation
            Dyson!(S)
            mpi_ismain() && println("Current occupation $(compute_occupation(S.G)), target $occ_target")
            μ = compute_hubbard_chemical_potential(occ_target, S.Σ, hubbard_params)
            set!(S.Gbare, hubbard_bare_Green(meshes(S.Gbare)...; μ, hubbard_params...))
        end

        Dyson!(S)
        bubbles!(S)

        flush(stdout)
        flush(stderr)

        if filename_log !== nothing
            if mpi_ismain()
                h5open(filename_log * ".iter$iter.h5", "w") do f
                    save!(f, "S", S)
                    f["mixing"] = mixing
                end
            end
        end

        if mpi_ismain()
            println("Self-energy error: $Σ_err")

            if Σ_err < tol
                break
            end
        end
    end
end





# -----------------------------------------------------------------------------
# Main driver
# -----------------------------------------------------------------------------

"""
    solve_using_mfRG_without_mixing!(S; verbose = true, occ_target = nothing, hubbard_params = nothing, tol = 1e-4, strategy = :fdPA, update_Σ = false, maxiter = 100, memory = 50, filename_log = nothing, use_preconditioner = true, store_trace = false, mixing = 0.85, _debug_outer = 1, compute_Hartree = true, krylov_maxiter = 400, krylov_memory = 200, callback_after_iterate = nothing) -> (; res, history)

Solve the fd parquet equations with the preconditioned iteration (Lihm 2025,
Eq. (S74)) and Anderson acceleration. This is the main driver for the lattice
solvers (used by script/common/runs.jl).

- Unknowns: the fd channels `Φ̃_r`, and also Σ if `update_Σ = true`.
- `occ_target`, `hubbard_params`: re-fit μ in every step so that the density
  stays at `occ_target` (Lihm 2025, text below Eq. (7)).
- Anderson: `maxiter` steps, tolerance `tol`, history length `memory`, mixing
  `mixing` (NLsolve's `beta`).
- `use_preconditioner = false` gives the plain fixed-point iteration (Eq. (S68)).
- `krylov_maxiter`, `krylov_memory`: settings of the dqgmres solver.
- `_debug_outer`: number of Anderson restarts (each starts from the last result
  with a fresh Krylov workspace).
- `compute_Hartree`, `callback_after_iterate`, `filename_log`: passed to
  [`fixed_point_preconditioned!`](@ref).

At the end the solution is loaded into `S` and symmetrized. Returns the last
NLsolve result and the iteration history.
"""
function solve_using_mfRG_without_mixing!(
    S :: AbstractSolver
    ;
    verbose = true,
    occ_target = nothing,
    hubbard_params = nothing,
    tol = 1e-4,
    strategy = :fdPA,
    update_Σ = false,
    maxiter = 100,
    memory = 50,
    filename_log = nothing,
    use_preconditioner = true,
    store_trace = false,
    mixing = 0.85,
    _debug_outer = 1,
    compute_Hartree = true,
    krylov_maxiter = 400,
    krylov_memory = 200,
    callback_after_iterate = nothing,
    )

    if occ_target !== nothing
        if hubbard_params === nothing
            throw(ArgumentError("hubbard_params must be provided if occ_target is given."))
        end
    end

    if mpi_ismain() && verbose
        println("Solve using mfRG preconditioning, strategy $strategy, tol $tol, maxiter $maxiter, update_Σ $update_Σ")
        println("Parallelization mode: $(S.mode), nthreads = $(Threads.nthreads())")
        if occ_target !== nothing
            println("Occupation fixed to $occ_target, hubbard parameters $hubbard_params")
        end
        println("norm|Πpp - Π0pp| = $(norm((S.Πpp - S.Π0pp).data))")
        println("norm|Πph - Π0ph| = $(norm((S.Πph - S.Π0ph).data))")
    end

    # Solve vertex
    if mpi_ismain() && verbose
        println("Solving vertex")
    end

    res = nothing
    history = NLsolve.SolverState[]

    # each pass is one (re)start of Anderson acceleration
    for i in 1:_debug_outer

        # start vector: the current solver state
        if update_Σ
            x0 = flatten(S)
        else
            x0 = flatten(S.F)
        end

        # OLD broken (older Krylov interface)
        # krylov_solver = Krylov.DqgmresSolver(mfRGLinearMap(S, strategy), flatten(S.F), krylov_memory)

        # NEW
        krylov_solver = Krylov.DqgmresWorkspace(mfRGLinearMap(S, strategy), flatten(S.F); memory = krylov_memory)

        res = nlsolve((R, x) -> fixed_point_preconditioned!(R, x, S, krylov_solver; update_Σ, strategy, occ_target, hubbard_params, filename_log, use_preconditioner, compute_Hartree, krylov_maxiter, callback_after_iterate), x0,
            method = :anderson,
            iterations = maxiter,
            ftol = tol,
            beta = mixing,
            m = memory,
            show_trace = mpi_ismain() && verbose,
            store_trace = store_trace,
        );

        if ! res.f_converged
            mpi_ismain() && println("Calculation did not converge in $(res.iterations) iterations.")
        else
            if mpi_ismain() && verbose
                println("Calculation converged")
            end
        end

        # load the solution into the solver
        if update_Σ
            unflatten!(S, res.zero);
        else
            unflatten!(S.F, res.zero);
        end

        symmetrize_solver!(S; verbose = false)
        append!(history, res.trace.states)
    end

    return (; res, history)
end
