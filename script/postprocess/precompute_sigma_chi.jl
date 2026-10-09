"""
precompute_sigma_chi.jl

Precompute and cache the self-energy and susceptibilities, as well as their
differences to the DMFT solution.

For one converged solver at (U, T, nmax, nq, NL_method) the self-energy and
the density/magnetic susceptibilities are evaluated on the dense k-grid and
written to one file, `output/postprocess/<run basename>.h5` (`postprocess_path`),
holding both the raw full-physical quantities and their tilde differences to the
DMFT reference:

  params/        U, U_str, T, nmax, nq, NL_method, strategy, iteration_scheme, nk
  convergence/   a copy of the solver's group (read it with `load_convergence`)
  sigma/nu                                            (nν)
  sigma/{raw,tilde}/{re,im}                           (nν × nk²)
  chi/omega                                           (nΩ)
  chi/{raw,tilde}/{D,M}                               (nΩ × nk²)

Each quantity is stored on the *entire* solver frequency grid (full ± Matsubara
range): `nu` / `omega` are the fermionic / bosonic Matsubara values in mesh
order, and each row of a stored matrix is that frequency's map over the dense
k-grid (nk = 48 by default, so nν = 8·nmax rows of 48² columns for Σ).

params/U_str is U as the file names spell it, which is not always what `string`
makes of params/U.

A run is addressed by the same two knobs `run_hubbard_model.jl` writes it with —
`strategy` (:scPA | :fdPA) and `iteration_scheme` (:anderson | :mfRG). Only
`strategy` affects what is computed: whether χ is stored relative to the DMFT
reference. `iteration_scheme` only picks which file is read and written.

Usage (from the repo root):
  julia --project=script script/postprocess/precompute_sigma_chi.jl --U=<U> [options]

Run with `--help` for the full option list.

  --U            interaction strength, as spelled in the DMFT file name (required)
  --nmax --nq --NL   the box the run was solved on
  --strategies   comma-separated scPA | fdPA      (default: both)
  --schemes      comma-separated anderson | mfRG  (default: both)
  --force        recompute runs whose cache already exists

With `--strategies` and `--schemes` left out, every (strategy, iteration_scheme)
pair is attempted and the ones with no data on disk are skipped quietly, so a
sweep picks up whichever runs exist. Naming either one makes missing data an
explicit warning instead.

  julia --project=script script/postprocess/precompute_sigma_chi.jl --U=2.9 --nmax=12 --nq=6
  julia --project=script script/postprocess/precompute_sigma_chi.jl --U=2.9 --strategies=fdPA --force

Can also be `include`d to use `precompute(...)` directly. The cache path it
writes comes from `postprocess_path(...)` in `common/paths.jl`.
"""

using MPI
MPI.Init()

using fdDGAsolver
using MatsubaraFunctions
using HDF5
using Printf

@isdefined(ScriptCommon) || include(joinpath(@__DIR__, "..", "common", "ScriptCommon.jl"))
using .ScriptCommon
@isdefined(ScriptRuns) || include(joinpath(@__DIR__, "..", "common", "ScriptRuns.jl"))
using .ScriptRuns

include(joinpath(@__DIR__, "load_hubbard_data.jl"))   # load_hubbard_solver
include(joinpath(@__DIR__, "post_process.jl"))        # chi_K1_all, chi_{D,M}_from_K1

# -----------------------------------------------------------------------
# Cache location, run naming and the DMFT reference all come from
# common/paths.jl: `postprocess_path` gives the cache for one run, and
# `dmft_file` / `dmft_temperature` address the reference input.
# -----------------------------------------------------------------------

"""
    split_U(U) -> (U_str, U_num)

The two forms of an interaction strength: the string that NAMES its files, and
the number the physics is done with (χ ~ 1/U²).

Every caller here takes U out of a file name and must put it back unchanged — a
run saved as `U2.10` is not the run `U2.1` names — while `precompute` also has to
divide by it. Taking `U` as given and splitting it once, here, keeps the two
apart: pass the string and both forms are exact.
"""
split_U(U) = U isa AbstractString ? (U, parse(Float64, U)) : (string(U), float(U))

"""
    dmft_reference(U) -> (; T, Σ_DMFT_ν0, chi_D_DMFT, chi_M_DMFT)

Load the DMFT reference and return the local self-energy at ν = iπT and the local
density / magnetic susceptibilities at ω = 0, alongside the reference temperature T.

`U` is taken as spelled (see [`split_U`](@ref)): the string picks the file, the
number divides into χ.
"""
function dmft_reference(U)
    U_str, U_num = split_U(U)
    dfile = dmft_file(U_str)
    isfile(dfile) || error("DMFT reference data not found:\n  $dfile")
    data_triqs = load_triqs_data(dfile)
    T  = data_triqs.params.T

    Ω0 = MatsubaraFrequency(T, 0, Boson)     # iω = 0
    ν0 = MatsubaraFrequency(T, 0, Fermion)   # iν = iπT

    Σ_DMFT_ν0  = data_triqs.Σ(value(ν0))
    chi_M_DMFT = chi_M_from_K1(data_triqs.Γ.γa.K1[Ω0], U_num)
    chi_D_DMFT = chi_D_from_K1(data_triqs.Γ.γa.K1[Ω0], data_triqs.Γ.γt.K1[Ω0], U_num)

    return (; T, Σ_DMFT_ν0, chi_D_DMFT, chi_M_DMFT)
end

# -----------------------------------------------------------------------
# Precompute
# -----------------------------------------------------------------------
"""
    precompute(U, nmax, nq, NL_method, strategy, iteration_scheme; force = false)
        -> path

Load the converged solver for one `(strategy, iteration_scheme)` run, evaluate Σ
and χ_{D,M} on the dense k-grid, and write both the raw full-physical quantities
and their tilde differences to the DMFT reference into the run's cache. When the
cache already exists the (slow) solver load is skipped unless `force = true`.

Pass `U` as the string it is spelled with in the file names; a number is accepted
and stringified, which is exact for every U currently on disk but not a property
to rely on (see [`split_U`](@ref)).

Returns the cache path.
"""
function precompute(U, nmax, nq, NL_method, strategy, iteration_scheme; force = false)
    # The name of every file touched below is built from `U_str`, never from the
    # number — including the solver this reads and the DMFT reference it subtracts.
    U_str, U_num = split_U(U)
    # Naming the cache needs only T, and `dmft_temperature` reads just that one
    # field — so an already-cached run never touches the (large) local vertex.
    T     = dmft_temperature(U_str)
    path  = postprocess_path(U_str, T, nmax, nq, NL_method, strategy, iteration_scheme)
    tag   = "$(ALGO_FAMILY[strategy])_$(iteration_scheme)"   # for messages only

    if !force && isfile(path)
        mpi_ismain() && println("Cache present for $(tag), U=$(U_str); nothing to do.")
        return path
    end

    mpi_ismain() && @printf "Precomputing %s data for U = %s (T = %s) …\n" tag U_str T

    # Full DMFT payload — needed from here on for the tilde differences, and handed
    # to the loader below, which would otherwise read the same file a second time.
    data_triqs = load_triqs_data(dmft_file(U_str))

    # Load the converged solver.
    S = load_hubbard_solver(U_str, T, nmax, nq, NL_method, strategy, iteration_scheme; data_triqs)

    # Convergence verdict recorded with the solver (run_hubbard_model.jl /
    # save_convergence!), copied into the cache below so the plot scripts can
    # filter without the solver file.
    conv = load_convergence(U_str, T, nmax, nq, NL_method, strategy, iteration_scheme)

    mν_G = meshes(S.Σ, Val(1))     # ν-mesh (G, Σ)
    mk_G = meshes(S.Σ, Val(2))     # dense k-mesh (G, Σ)
    nk   = Int(sqrt(length(mk_G))) # linear size of the dense k-grid

    # -------------------------------------------------------------------
    # Full-frequency-grid quantities, stored over the whole solver frequency grid. 
    # -------------------------------------------------------------------
    # Σ(iν, k): already stored on the entire (ν, k) grid — no slicing.  νvals is
    # the fermionic Matsubara axis in mesh order (full ± range).
    νvals    = Float64[ plain_value(ν) for ν in mν_G ]
    Σ_DMFT_ν = ComplexF64[ data_triqs.Σ(value(ν)) for ν in mν_G ]   # local DMFT Σ per ν

    # Σ is stored as im*Σ: the physical parts are (Re, Im) = (imag, -real) of the
    # stored value, for the solver and the DMFT reference alike.
    Σ_raw   = S.Σ.data
    Σ_tilde = Σ_raw .- Σ_DMFT_ν                                     # subtract per ν across k

    ReΣ_raw,   ImΣ_raw   = imag.(Σ_raw),   -real.(Σ_raw)
    ReΣ_tilde, ImΣ_tilde = imag.(Σ_tilde), -real.(Σ_tilde)

    # χ_{D,M}(iΩ, q) for every bosonic Ω on the solver's K1 mesh.
    Ωvals, χD_all, χM_all = chi_K1_all(S, U_num)                    # (nΩ,), (nΩ, nk²)×2
    mΩ_G    = meshes(S.F.γa.K1, Val(1))                             # matches Ωvals order
    mΩ_ref  = meshes(data_triqs.Γ.γa.K1, Val(1))                    # local DMFT K1 mesh
    K1a_ref = data_triqs.Γ.γa.K1
    K1t_ref = data_triqs.Γ.γt.K1
    # Local DMFT reference χ_{D,M}(Ω); zero outside the reference K1 mesh, matching
    # the high-frequency decay convention of γ_r (see Channel evaluation).
    chiD_DMFT_all = Float64[ (Ωf = value(Ω); is_inbounds(Ωf, mΩ_ref) ?
        chi_D_from_K1(K1a_ref[Ωf], K1t_ref[Ωf], U_num) : 0.0) for Ω in mΩ_G ]
    chiM_DMFT_all = Float64[ (Ωf = value(Ω); is_inbounds(Ωf, mΩ_ref) ?
        chi_M_from_K1(K1a_ref[Ωf], U_num) : 0.0) for Ω in mΩ_G ]
    if strategy === :fdPA
        # The fd solver stores K1 as the difference to the reference, so what
        # `chi_K1_all` returned *is* the tilde quantity. Adding the reference back
        # and subtracting it again would round-trip through (a+b)-b, costing
        # accuracy and two full (nΩ × nk²) allocations per channel.
        chiD_tilde, chiM_tilde = χD_all, χM_all
        chiD_raw = χD_all .+ chiD_DMFT_all                      # add reference back per Ω
        chiM_raw = χM_all .+ chiM_DMFT_all
    else
        chiD_raw, chiM_raw = χD_all, χM_all
        chiD_tilde = chiD_raw .- chiD_DMFT_all                  # subtract reference per Ω
        chiM_tilde = chiM_raw .- chiM_DMFT_all
    end

    if mpi_ismain()
        mkpath(dirname(path))

        # Written under a temporary name and moved into place, so an interrupted
        # write never leaves a file that `isfile(path)` would count as complete.
        tmp = path * ".part"
        h5open(tmp, "w") do f
            f["params/U"]         = U_num     # numeric; `params/U_str` names the files
            f["params/T"]         = T
            f["params/nmax"]      = nmax
            f["params/nq"]        = nq
            f["params/NL_method"] = NL_method
            # Stored as strings: HDF5 has no symbol type, and these are read back
            # by h5py / TRIQS as well. `Symbol(read(...))` recovers the knobs.
            f["params/strategy"]         = String(strategy)
            f["params/iteration_scheme"] = String(iteration_scheme)
            f["params/U_str"]     = U_str
            f["params/nk"]        = nk

            write_convergence!(f, conv)

            f["sigma/nu"]        = νvals
            f["sigma/raw/re"]    = ReΣ_raw
            f["sigma/raw/im"]    = ImΣ_raw
            f["sigma/tilde/re"]  = ReΣ_tilde
            f["sigma/tilde/im"]  = ImΣ_tilde

            f["chi/omega"]       = Ωvals
            f["chi/raw/D"]       = chiD_raw
            f["chi/raw/M"]       = chiM_raw
            f["chi/tilde/D"]     = chiD_tilde
            f["chi/tilde/M"]     = chiM_tilde
        end
        mv(tmp, path; force = true)
        println("  wrote $path")
    end

    return path
end
# ---------------------------------------------------------------------------
# CLI driver (skipped when this file is `include`d)
# ---------------------------------------------------------------------------
# The defaults sit INSIDE this block rather than in a file-scope REPL block like
# every other script's. This file is `include`d by others, and a name defined at
# file scope lands in theirs: `U` and `nmax` here would arrive in the middle of
# a plot script that defines its own. There is nothing to edit-and-rerun anyway —
# the REPL entry point is calling `precompute(...)` directly.
if abspath(PROGRAM_FILE) == @__FILE__
    nmax       = 10   # half-box size for the vertex grids
    nq         = 4    # linear k-grid size for the vertex mesh
    NL_method  = 1    # vertex momentum structure: 1 (K2(k,q)≈K2(q) ...
    strategies = ""   # comma-separated scPA | fdPA (empty: both)
    schemes    = ""   # comma-separated anderson | mfRG (empty: both)

    settings = cli_settings(
        "Precompute and cache the self-energy and susceptibilities for saved solver runs.")
    add_arg_table!(settings,
        ["--U"], Dict(:help => "interaction strength, as spelled in the DMFT file name",
                      :arg_type => String, :required => true))
    add_box_args!(settings; nmax, nq, NL_method)
    add_arg_table!(settings,
        ["--strategies"], Dict(:help => "comma-separated scPA | fdPA (default: both)",
                               :arg_type => String, :default => strategies),
        ["--schemes", "--iteration_schemes"],
                          Dict(:help => "comma-separated anderson | mfRG (default: both)",
                               :arg_type => String, :default => schemes))
    add_flags!(settings,
        "force" => "recompute runs whose cache already exists")

    parsed = parse_args(ARGS, settings)

    U         = parsed["U"]
    nmax      = parsed["nmax"]
    nq        = parsed["nq"]
    NL_method = parsed["NL_method"]
    force     = parsed["force"]

    # A named run is a request for that one thing, so missing data there is worth
    # reporting. Left at the default, every (strategy, iteration_scheme) pair is
    # attempted and absent runs are the normal case, so those are skipped quietly.
    named_strategies = isempty(parsed["strategies"]) ? nothing : parse_strategies(parsed["strategies"])
    named_schemes    = isempty(parsed["schemes"])    ? nothing : parse_schemes(parsed["schemes"])
    explicit         = named_strategies !== nothing || named_schemes !== nothing

    runs = [(s, i) for s in something(named_strategies, STRATEGIES)
                   for i in something(named_schemes,    ITERATION_SCHEMES)]

    # `U` stays the string it is spelled with — that is what names the files.
    # `precompute` splits off the number it needs for the arithmetic itself.
    T = dmft_temperature(U)

    for (strategy, iteration_scheme) in runs
        tag = "$(ALGO_FAMILY[strategy])_$(iteration_scheme)"

        # A missing solver file is fine once the cache exists — `precompute` skips
        # the (slow) solver load in that case — so only treat a run as absent when
        # neither the solver nor the cache is on disk. Under `--force` the cache
        # is rewritten from the solver, so then the solver has to be there.
        have_solver = isfile(solver_path(U, T, nmax, nq, NL_method, strategy, iteration_scheme))
        have_cache  = !force && isfile(postprocess_path(U, T, nmax, nq, NL_method, strategy, iteration_scheme))
        if !have_solver && !have_cache
            if explicit
                @warn "no saved solver and no complete cache" strategy iteration_scheme U T nmax nq NL_method
            else
                println("No data for $(tag); skipping.")
            end
            continue
        end

        try
            precompute(U, nmax, nq, NL_method, strategy, iteration_scheme; force)
        catch err
            # A sweep should survive one bad run, but not swallow Ctrl-C — without
            # this an interrupt is logged as a failure and the sweep marches on.
            err isa InterruptException && rethrow()
            @warn "precompute failed" strategy iteration_scheme exception = err
        end
    end
end
