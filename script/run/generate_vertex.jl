"""
Parquet approximation on top of a DMFT bath propagator.

The reference system is trivial (G0 = Σ0 = 0, F0 = bare U), so `:scPA` reduces to the
self-consistent parquet approximation for the impurity problem defined by the bath
propagator `G0` stored in the DMFT file.

Usage (from the repo root, multi-threaded — `mode = :threads` needs the threads):
  julia --project=script -t <nthreads> script/run/generate_vertex.jl [options]

Run with `--help` for the full option list and the current defaults. Like every
other script here the options are named, so order never matters:

  --dmft           DMFT reference: a path relative to the repo root, or just the U
                   (shorthand for data/high_temperature_U<U>.h5)
  --tol --maxiter  solver stopping criteria

  julia --project=script -t 8 script/run/generate_vertex.jl --dmft=2.089
  julia --project=script -t 8 script/run/generate_vertex.jl --dmft=0.5 --tol=1e-10

Output — one HDF5 file under `output/parquet_vertex/`:
  parquet_U<U>_T<T>_nK1<nK1>_nK2<nK2a>-<nK2b>_nK3<nK3a>-<nK3b>.h5
holding the full solver state written by `save!` (Gbare, G0, Σ0, F0, Π0pp, Π0ph, G,
Σ, F, Πpp, Πph) plus the same `convergence` group every other run writes, so
`load_convergence` (common/convergence.jl) reads it. Reload the state with
`load_solver!` into a solver shell built with the same box sizes.
"""

@isdefined(ScriptCommon) || include(joinpath(@__DIR__, "..", "common", "ScriptCommon.jl"))
using .ScriptCommon

# ---------------------------------------------------------------------------
# What executing this file in the REPL does. Edit these and execute it again.
#
# Running the file as a script overrides them from the command line - and these
# are the defaults `--help` reports, so the two entry points cannot drift apart.
# ---------------------------------------------------------------------------
dmft    = "0.5"    # DMFT reference: a path relative to the repo root, or just ...
tol     = 1.0e-12  # residual-norm convergence tolerance
maxiter = 200      # maximum parquet iterations

# The impurity box. Not a CLI option: unlike the lattice runs these are not one
# `nmax` apart but three independently chosen boxes, and `box_sizes` (common/runs.jl)
# does not describe them.
const NK1 = 255
const NK2 = (147, 100)
const NK3 = (47, 32)

# ---------------------------------------------------------------------------
# CLI driver (skipped when this file is executed in the REPL or `include`d)
# ---------------------------------------------------------------------------
# Parsed before the heavy imports below, so `--help` and a mistyped option come
# back immediately instead of after MPI and fdDGAsolver have loaded.
if abspath(PROGRAM_FILE) == @__FILE__
    settings = cli_settings(
        "Converge the parquet approximation from the bare propagator and save the vertex.")
    add_dmft_arg!(settings; default = dmft)
    add_convergence_args!(settings; tol, maxiter)

    parsed = parse_args(ARGS, settings)

    dmft    = parsed["dmft"]
    tol     = parsed["tol"]
    maxiter = parsed["maxiter"]
end

# A missing input is the most common mistake, and saying so before the heavy
# imports costs nothing. `resolve_dmft` (common/cli.jl) is the one place that
# turns what `--dmft` accepts into a path.
dmft_path = resolve_dmft(dmft)

# ---------------------------------------------------------------------------
# Heavy imports
# ---------------------------------------------------------------------------
using MPI
MPI.Init()

using fdDGAsolver
using MatsubaraFunctions
using HDF5

const OUTPUT_DIR = joinpath(output_dir(), "parquet_vertex")

# Write the converged solver plus its convergence verdict. `save!` for a solver ignores
# the label and writes Gbare, G0, Σ0, F0, Π0pp, Π0ph, G, Σ, F, Πpp, Πph at the file root,
# so `convergence` sits alongside them. The group itself is written by the shared
# `save_convergence!` (common/convergence.jl), so `load_convergence` reads it back.
# Returns the output path.
function save_solver!(S, result; U, T, nK1, nK2, nK3, tol, maxiter)
    mkpath(OUTPUT_DIR)
    fname = joinpath(OUTPUT_DIR,
        "parquet_U$(U)_T$(T)_nK1$(nK1)_nK2$(nK2[1])-$(nK2[2])_nK3$(nK3[1])-$(nK3[2]).h5")

    h5open(fname, "w") do f
        save!(f, "S", S)
        save_convergence!(f, result; tol, maxiter)
    end

    return fname
end

function solve_with_bare_propagator(path::String; tol = tol, maxiter = maxiter)
    # load 2P data
    data = load_triqs_data(path)

    # T and U from DMFT data
    (; T, U) = data.params

    Gbare = data.G0                       # physical impurity bare propagator (bath)
    G0    = copy(Gbare)                   # trivial reference: G0 = 0
    set!(G0, 0)
    Σ0    = copy(Gbare)                   # trivial reference: Σ0 = 0
    set!(Σ0, 0)
    F0    = RefVertex(T, U, ComplexF64)   # bare vertex, +U pSp convention

    S = ParquetSolver(NK1, NK2, NK3, Gbare, G0, Σ0, F0; mode = :threads)
    init_sym_grp!(S)
    result = fdDGAsolver.solve!(S; strategy = :scPA, maxiter, tol, verbose = mpi_ismain())

    # every rank holds a full solver copy, so only the main rank writes
    return mpi_ismain() ?
        save_solver!(S, result; U, T, nK1 = NK1, nK2 = NK2, nK3 = NK3, tol, maxiter) :
        nothing
end

if abspath(PROGRAM_FILE) == @__FILE__
    fname = solve_with_bare_propagator(dmft_path; tol, maxiter)
    mpi_ismain() && println("Saved solver to $fname")
end
