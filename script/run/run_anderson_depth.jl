"""
Solve the 2D Hubbard model with the finite-difference parquet solver for a given
Anderson acceleration DEPTH and save the converged solver object.

This is run_hubbard_model.jl with the two algorithm knobs fixed and one new one
opened up. `strategy` is always :fdPA (the DMFT vertex as the reference F0) and
`iteration_scheme` is always :anderson (`solve!`), so the options simplify to

  `--mem`   — the Anderson history depth handed to NLsolve (0 = damped Picard)
  `--delta` — the Anderson mixing β, because the two interact: a depth that stalls
              at β = 0.85 may converge at a smaller β, so a scan at fixed β answers
              "which depths converge at this β", not "which depth is best"

Everything else — meshes, box sizes, solver construction, `update_Σ` and the saved
solver state — comes from the same `common/runs.jl` that run_hubbard_model.jl
uses, so a run here is directly comparable with a production fd_anderson run at
the same (nmax, nq, NL_method). What differs is only the output directory and the
file name, which carry `mem` and `delta`, so nothing under output/hubbard_model/
is touched or shadowed.

The DMFT reference (temperature T, chemical potential, hopping and local vertex)
is read from `--dmft`; U and T are taken from it.

Usage (from the repo root, multi-threaded):
  julia --project=script -t <nthreads> script/run/run_anderson_depth.jl [options]

Run with `--help` for the full option list and the current defaults:

  julia --project=script -t 8 script/run/run_anderson_depth.jl --mem=16
  julia --project=script -t 8 script/run/run_anderson_depth.jl --mem=4 --delta=0.5 --dmft=2.06

Output — one HDF5 file under `output/anderson_depth/`:
  hubbard_fd_anderson_mem<mem>_delta<delta>_U<U>_T<T>_nmax<nmax>_nq<nq>_NL<NL>.h5

The file holds the full solver state written by `save!` plus a `convergence`
group — the same layout run_hubbard_model.jl writes, so `load_convergence`
(common/convergence.jl) reads these files unchanged.
"""

@isdefined(ScriptCommon) || include(joinpath(@__DIR__, "..", "common", "ScriptCommon.jl"))
using .ScriptCommon

# -----------------------------------------------------------------------
# Arguments
# -----------------------------------------------------------------------
# Parsed before the heavy imports below, so `--help` and a mistyped option come
# back immediately instead of after MPI and fdDGAsolver have loaded.
# ---------------------------------------------------------------------------
# What executing this file in the REPL does. Edit these and execute it again.
#
# Running the file as a script overrides them from the command line - and these
# are the defaults `--help` reports, so the two entry points cannot drift apart.
# ---------------------------------------------------------------------------
mem       = 8       # Anderson history depth (0 = damped Picard)
delta     = 0.85    # Anderson mixing β handed to NLsolve
nmax      = 10      # half-box size for the vertex grids
nq        = 4       # linear k-grid size for the vertex mesh
NL_method = 1       # vertex momentum structure: 1 (K2(k,q)≈K2(q) ...
dmft      = "2.9"   # DMFT reference: a path relative to the repo root, or just ...
tol       = 1.0e-8  # residual-norm convergence tolerance
maxiter   = 200     # maximum parquet iterations

# ---------------------------------------------------------------------------
# CLI driver (skipped when this file is executed in the REPL or `include`d)
# ---------------------------------------------------------------------------
if abspath(PROGRAM_FILE) == @__FILE__
    settings = cli_settings(
        "Solve the 2D Hubbard model at a given Anderson acceleration depth (:fdPA, :anderson).")
    add_anderson_args!(settings; mem, delta)
    add_box_args!(settings; nmax, nq, NL_method)
    add_dmft_arg!(settings; default = dmft)
    add_convergence_args!(settings; tol, maxiter)

    parsed = parse_args(ARGS, settings)

    mem       = parsed["mem"]
    delta     = parsed["delta"]
    nmax      = parsed["nmax"]
    nq        = parsed["nq"]
    NL_method = parsed["NL_method"]
    dmft      = parsed["dmft"]
    tol       = parsed["tol"]
    maxiter   = parsed["maxiter"]
end

δ         = delta

# Fixed by design — this script exists to vary `mem`, nothing else about the
# algorithm: the reference vertex F0 is always the DMFT one, and the fixed point
# is always reached by the Anderson-accelerated parquet iteration.
const STRATEGY         = :fdPA
const ITERATION_SCHEME = :anderson

# Iterate Σ together with the vertex, so that `mem` is the only thing varying
# across a depth scan.
update_Σ = true

# The DMFT path is resolved before the heavy imports too: a missing input is the
# most common mistake, and it costs nothing to say so now. `resolve_dmft`
# (common/cli.jl) is the one place that turns what `--dmft` accepts into a path.
dmft_path = resolve_dmft(dmft)

# -----------------------------------------------------------------------
# Heavy imports
# -----------------------------------------------------------------------
using MPI
MPI.Init()

using fdDGAsolver
using MatsubaraFunctions
using Printf

@isdefined(ScriptRuns) || include(joinpath(@__DIR__, "..", "common", "ScriptRuns.jl"))
using .ScriptRuns

# -----------------------------------------------------------------------
# Load DMFT reference data (PHS)
# -----------------------------------------------------------------------
data_triqs = load_triqs_data(dmft_path)
(; T, U, μ, t1) = data_triqs.params

@mainprintf("=== PHS Hubbard model — Anderson depth scan ===\n")
@mainprintf("  mem   = %-8d  delta = %.4f\n", mem, δ)
@mainprintf("  T     = %-8.4f  U  = %-8.4f  μ  = %-8.4f  t1 = %.4f\n", T, U, μ, t1)
@mainprintf("  nmax  = %-8d  nq = %-8d  NL = %d\n", nmax, nq, NL_method)
@mainprintf("  tol   = %-8.1e  maxiter = %d\n", tol, maxiter)
@mainprintf("  strategy = :%s,  iteration_scheme = :%s\n", STRATEGY, ITERATION_SCHEME)

# -----------------------------------------------------------------------
# Build, converge, save
# -----------------------------------------------------------------------
@mainprintf("\n=== Running fdpDΓA with anderson (mem = %d, delta = %.4f) ===\n", mem, δ)
@mainprintf("Create solver.\n")

S = build_solver_from_dmft(data_triqs, nmax, nq, NL_method; strategy = STRATEGY)
data_triqs = nothing

@mainprintf("Initialize symmetry group.\n")
t_sym = time()
init_sym_grp!(S)
@mainprintf("Symmetry group initialized in %.3f seconds.\n", time() - t_sym)

# `mem` and `δ` go straight through to NLsolve's `m` / `beta`.
@mainprintf("Start solver.\n")
t_solve = time()
result = solve_run!(S; strategy = STRATEGY, iteration_scheme = ITERATION_SCHEME,
                    maxiter, tol, update_Σ, δ, mem, verbose = mpi_ismain())
@mainprintf("Solver finished in %.3f seconds.\n", time() - t_solve)

if mpi_ismain()
    @printf("\n=== Saving fd-data (mem = %d) ===\n", mem)
    path  = anderson_depth_path(mem, δ, U, T, nmax, nq, NL_method)
    fname = save_run!(path, S, result; tol, maxiter)
    @printf("fd-data saved to %s\n", fname)
end
