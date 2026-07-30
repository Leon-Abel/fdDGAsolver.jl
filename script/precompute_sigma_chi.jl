"""
precompute_sigma_chi.jl

Precompute and cache the self-energy and susceptibilities, aswell as their
differences to the DMFT solution.

For a converged `algo` solver at (U, T, nmax, nq, NL_method) the self-energy and
the density/magnetic susceptibilities are evaluated on the dense k-grid and
written to four subfolders of `output/`: the tilde differences to the DMFT
reference go into the *_tilde folders, the raw full-physical quantities into the
plain folders (file basenames are identical across all four):

Each quantity is stored on the *entire* solver frequency grid (full ± Matsubara
range) alongside its axis:

  self_energy_tilde:    nu,    ReSigma_tilde, ImSigma_tilde   (8*nmax × 48²)
  self_energy:          nu,    ReSigma_raw,   ImSigma_raw     (8*nmax × 48²)
  susceptibility_tilde: omega, chiD_tilde,    chiM_tilde      (nΩ × nk²)
  susceptibility:       omega, chiD_raw,      chiM_raw        (nΩ × nk²)

where `nu` (length nν) / `omega` (length nΩ) are the fermionic / bosonic Matsubara
values in mesh order, and each row of a stored matrix is that frequency's map over
the dense k-grid. Consumers wanting a single frequency (e.g. ν = iπT, Ω = 0) locate
the row on the `nu` / `omega` axis — see `nu0_row` / `omega0_row`.

All four files also store params/{U,T,nmax,nq,NL_method,algo,nk}.

Usage (from the repo root):
  julia script/precompute_sigma_chi.jl <U> <nmax> <nq> <NL_method> [algo]
    algo defaults to running both "sc" and "fd".

Can also be `include`d to use `precompute(...)`, `dmft_temperature(...)`,
`sigma_tilde_path(...)`, `sigma_raw_path(...)`, `chi_tilde_path(...)` and
`chi_raw_path(...)` directly.
"""

using MPI
MPI.Init()

using fdDGAsolver
using MatsubaraFunctions
using StaticArrays
using HDF5
using Printf

include(joinpath(@__DIR__, "load_hubbard_data.jl"))   # load_hubbard_solver, load_triqs_data
include(joinpath(@__DIR__, "post_process.jl"))        # chi_K1

# -----------------------------------------------------------------------
# Cache locations.  The tilde differences to the DMFT reference live in the
# *_tilde subfolders; the raw (full physical) quantities live in the plain ones.
# -----------------------------------------------------------------------
_output_dir()     = normpath(joinpath(dirname(pathof(fdDGAsolver)), "..", "output"))
sigma_tilde_dir() = joinpath(_output_dir(), "self_energy_tilde")
sigma_raw_dir()   = joinpath(_output_dir(), "self_energy")
chi_tilde_dir()   = joinpath(_output_dir(), "susceptibility_tilde")
chi_raw_dir()     = joinpath(_output_dir(), "susceptibility")

_cache_name(algo, U, T, nmax, nq, NL_method) =
    "hubbard_$(algo)_U$(U)_T$(T)_nmax$(nmax)_nq$(nq)_NL$(NL_method).h5"

sigma_tilde_path(U, T, nmax, nq, NL_method, algo) =
    joinpath(sigma_tilde_dir(), _cache_name(algo, U, T, nmax, nq, NL_method))
sigma_raw_path(U, T, nmax, nq, NL_method, algo) =
    joinpath(sigma_raw_dir(),   _cache_name(algo, U, T, nmax, nq, NL_method))
chi_tilde_path(U, T, nmax, nq, NL_method, algo) =
    joinpath(chi_tilde_dir(),   _cache_name(algo, U, T, nmax, nq, NL_method))
chi_raw_path(U, T, nmax, nq, NL_method, algo) =
    joinpath(chi_raw_dir(),     _cache_name(algo, U, T, nmax, nq, NL_method))

# -----------------------------------------------------------------------
# Convergence flag propagated from the solver into the caches
# -----------------------------------------------------------------------
# The plot scripts read the caches, not the (often deleted) solver files, so the
# convergence verdict recorded by `run_hubbard_model.jl` / `save_convergence!` is
# copied into each cache under params/converged (1 = converged, 0 = not converged).

"""
    cache_converged(path) -> Bool

Whether the solver behind a precompute cache converged, from the `params/converged`
flag written by `precompute`. Returns `false` if the flag (or file) is absent.
"""
function cache_converged(path)
    isfile(path) || return false
    return h5open(path, "r") do f
        haskey(f, "params") && haskey(f["params"], "converged") &&
            read(f["params"]["converged"]) == 1
    end
end

# -----------------------------------------------------------------------
# Fixed-frequency slice from the caches
# -----------------------------------------------------------------------
# precompute stores the full-frequency arrays; consumers that want a
# single frequency locate its row on the cached `nu` / `omega` axis.

"Row index of ν = iπT (lowest positive fermionic Matsubara) on a cached `nu` axis."
nu0_row(nu, T) = argmin(abs.(nu .- π * T))

"Row index of Ω = 0 on a cached `omega` axis."
omega0_row(omega) = argmin(abs.(omega))

# -----------------------------------------------------------------------
# DMFT reference
# -----------------------------------------------------------------------
_dmft_file(U) = normpath(joinpath(dirname(pathof(fdDGAsolver)), "..", "data",
                                  "high_temperature_U$(U).h5"))

# `U` may arrive as a String — the plot scripts parse it out of the cache/solver
# filenames and must keep that exact spelling for path interpolation ("2.10" and
# 2.1 name different files). Use this for arithmetic only.
_Uval(U) = U isa AbstractString ? parse(Float64, U) : float(U)

"""
    dmft_temperature(U)

Temperature T stored with the DMFT reference data for interaction U.
"""
function dmft_temperature(U)
    dfile = _dmft_file(U)
    isfile(dfile) || error("DMFT reference data not found:\n  $dfile")
    return h5read(dfile, "params/T")
end

"""
    dmft_reference(U) -> (; T, Σ_DMFT_ν0, chi_D_DMFT, chi_M_DMFT)

Load the DMFT reference and return the local self-energy at ν = iπT and the local
density / magnetic susceptibilities at ω = 0, alongside the reference temperature T.

These are single-frequency (ν = iπT, Ω = 0) reference values used by the
fixed-frequency plots (`plot_sigma_chi_vs_{U,q}.jl`) to draw the DMFT point and to
shift the detQMC benchmark. The frequency-resolved tilde differences stored in the
caches are formed per-frequency inside `precompute`, not from this function.
"""
function dmft_reference(U)
    dfile = _dmft_file(U)
    isfile(dfile) || error("DMFT reference data not found:\n  $dfile")
    data_triqs = load_triqs_data(dfile)
    T  = data_triqs.params.T
    Uv = _Uval(U)

    Ω0 = MatsubaraFrequency(T, 0, Boson)     # iω = 0
    ν0 = MatsubaraFrequency(T, 0, Fermion)   # iν = iπT

    Σ_DMFT_ν0  = data_triqs.Σ(value(ν0))
    chi_M_DMFT = real(data_triqs.Γ.γa.K1[Ω0]) / Uv^2 / 2
    chi_D_DMFT = (-2real(data_triqs.Γ.γt.K1[Ω0]) + real(data_triqs.Γ.γa.K1[Ω0])) / Uv^2 / 2

    return (; T, Σ_DMFT_ν0, chi_D_DMFT, chi_M_DMFT)
end

# -----------------------------------------------------------------------
# Precompute
# -----------------------------------------------------------------------
"""
    precompute(U, nmax, nq, NL_method, algo; force = false)
        -> (; sigma_tilde, sigma_raw, chi_tilde, chi_raw)

Load the converged `algo` ("sc" or "fd") solver, evaluate Σ and χ_{D,M} on the
dense k-grid, and write both the tilde differences to the DMFT reference (the
*_tilde caches) and the raw full-physical quantities (the plain caches). The
temperature is taken from the DMFT reference (fixed by U). When all four cache
files already exist the (slow) solver load is skipped unless `force = true`.
Returns a NamedTuple of the four file paths.
"""
function precompute(U, nmax, nq, NL_method, algo; force = false)
    # Load the DMFT reference; used to form the full-grid tilde differences below.
    dfile = _dmft_file(U)
    isfile(dfile) || error("DMFT reference data not found:\n  $dfile")
    data_triqs = load_triqs_data(dfile)
    T  = data_triqs.params.T
    Uv = _Uval(U)

    files = (; sigma_tilde = sigma_tilde_path(U, T, nmax, nq, NL_method, algo),
               sigma_raw   = sigma_raw_path(  U, T, nmax, nq, NL_method, algo),
               chi_tilde   = chi_tilde_path(  U, T, nmax, nq, NL_method, algo),
               chi_raw     = chi_raw_path(    U, T, nmax, nq, NL_method, algo))

    if !force && all(isfile, files)
        mpi_ismain() && println("Cache present for algo=$(algo), U=$(U); nothing to do.")
        return files
    end

    mpi_ismain() && @printf "Precomputing %s data for U = %s (T = %s) …\n" algo U T

    # Load the converged solver.
    S = load_hubbard_solver(U, T, nmax, nq, NL_method, algo)

    # Convergence verdict recorded with the solver (run_hubbard_model.jl /
    # save_convergence!), stamped into the caches below so the plot scripts can filter
    # without the solver file.
    conv          = load_convergence(hubbard_solver_filename(U, T, nmax, nq, NL_method, algo))
    conv_flag     = Int(conv.converged)
    conv_residual = conv.residual_norm

    mν_G = meshes(S.Σ, Val(1))     # ν-mesh (G, Σ)
    mK_G = meshes(S.Σ, Val(2))     # dense k-mesh (G, Σ)
    nk   = Int(sqrt(length(mK_G))) # linear size of the dense k-grid

    # -------------------------------------------------------------------
    # Full-frequency-grid quantities, stored over the whole solver frequency grid. 
    # A fixed-frequency slice (e.g. ν = iπT, Ω = 0) is recovered 
    # from the `nu` / `omega` axis (see nu0_row / omega0_row).
    # -------------------------------------------------------------------
    # Σ(iν, k): already stored on the entire (ν, k) grid — no slicing.  νvals is
    # the fermionic Matsubara axis in mesh order (full ± range).
    νvals    = Float64[ plain_value(ν) for ν in mν_G ]
    Σ_DMFT_ν = ComplexF64[ data_triqs.Σ(value(ν)) for ν in mν_G ]   # local DMFT Σ per ν
    ReΣ_raw =  imag.(S.Σ.data)                                  
    ImΣ_raw = -real.(S.Σ.data)
    ReΣ_tilde = ReΣ_raw .- imag.(Σ_DMFT_ν)                  # subtract per ν across k
    ImΣ_tilde = ImΣ_raw .+ real.(Σ_DMFT_ν)

    # χ_{D,M}(iΩ, q) for every bosonic Ω on the solver's K1 mesh.
    Ωvals, χD_all, χM_all = chi_K1_all(S, Uv)                       # (nΩ,), (nΩ, nk²)×2
    mΩ_G    = meshes(S.F.γa.K1, Val(1))                             # matches Ωvals order
    mΩ_ref  = meshes(data_triqs.Γ.γa.K1, Val(1))                    # local DMFT K1 mesh
    K1a_ref = data_triqs.Γ.γa.K1
    K1t_ref = data_triqs.Γ.γt.K1
    # Local DMFT reference χ_{D,M}(Ω); zero outside the reference K1 mesh, matching
    # the high-frequency decay convention of γ_r (see Channel evaluation).
    chiD_DMFT_all = Float64[ (Ωf = value(Ω); is_inbounds(Ωf, mΩ_ref) ?
        (-2real(K1t_ref[Ωf]) + real(K1a_ref[Ωf])) / Uv^2 / 2 : 0.0) for Ω in mΩ_G ]
    chiM_DMFT_all = Float64[ (Ωf = value(Ω); is_inbounds(Ωf, mΩ_ref) ?
        real(K1a_ref[Ωf]) / Uv^2 / 2 : 0.0) for Ω in mΩ_G ]
    if algo == "fd"
        chiD_raw = χD_all .+ chiD_DMFT_all                      # add reference back per Ω
        chiM_raw = χM_all .+ chiM_DMFT_all
    else
        chiD_raw, chiM_raw = χD_all, χM_all
    end
    chiD_tilde = chiD_raw .- chiD_DMFT_all
    chiM_tilde = chiM_raw .- chiM_DMFT_all

    if mpi_ismain()
        for d in (sigma_tilde_dir(), sigma_raw_dir(), chi_tilde_dir(), chi_raw_dir())
            mkpath(d)
        end

        write_params!(f) = begin
            f["params/U"]         = Uv   # numeric, whatever form the caller passed
            f["params/T"]         = T
            f["params/nmax"]      = nmax
            f["params/nq"]        = nq
            f["params/NL_method"] = NL_method
            f["params/algo"]      = algo
            f["params/nk"]        = nk
            f["params/converged"]     = conv_flag       # 1 converged, 0 not
            f["params/residual_norm"] = conv_residual
        end

        h5open(files.sigma_tilde, "w") do f
            write_params!(f)
            f["nu"]                = νvals              # fermionic axis (length nν)
            f["ReSigma_tilde"] = ReΣ_tilde      # full grid (nν × nk²)
            f["ImSigma_tilde"] = ImΣ_tilde
        end
        h5open(files.sigma_raw, "w") do f
            write_params!(f)
            f["nu"]              = νvals                # fermionic axis (length nν)
            f["ReSigma_raw"] = ReΣ_raw          # full grid (nν × nk²)
            f["ImSigma_raw"] = ImΣ_raw
        end
        h5open(files.chi_tilde, "w") do f
            write_params!(f)
            f["omega"]          = Ωvals                 # bosonic axis (length nΩ)
            f["chiD_tilde"] = chiD_tilde        # full grid (nΩ × nk²)
            f["chiM_tilde"] = chiM_tilde
        end
        h5open(files.chi_raw, "w") do f
            write_params!(f)
            f["omega"]        = Ωvals                   # bosonic axis (length nΩ)
            f["chiD_raw"] = chiD_raw            # full grid (nΩ × nk²)
            f["chiM_raw"] = chiM_raw
        end
        for p in files
            println("  wrote $p")
        end
    end

    return files
end

# ---------------------------------------------------------------------------
# CLI driver (skipped when this file is `include`d)
# ---------------------------------------------------------------------------
if abspath(PROGRAM_FILE) == @__FILE__
    length(ARGS) >= 4 || error(
        "Usage: julia script/precompute_sigma_chi.jl <U> <nmax> <nq> <NL_method> [algo]")

    U         = parse(Float64, ARGS[1])
    nmax      = parse(Int,     ARGS[2])
    nq        = parse(Int,     ARGS[3])
    NL_method = parse(Int,     ARGS[4])
    algos     = length(ARGS) >= 5 ? [ARGS[5]] : ["sc", "fd"]

    for algo in algos
        try
            precompute(U, nmax, nq, NL_method, algo)
        catch err
            @warn "precompute failed" algo exception = err
        end
    end
end
