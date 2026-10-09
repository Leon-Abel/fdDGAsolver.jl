"""
Solve the 2D Hubbard model with the finite-difference parquet solver and save the
converged solver object for post-processing.

One run per process, selected by two independent settings:

  `--strategy` — what the reference vertex F0 is:
    scPA  parquet approximation — no DMFT input, bare vertex as reference
    fdPA  finite-difference parquet DΓA — DMFT vertex as the reference

  `--iteration-scheme` (or `--scheme`) — how the fixed point is reached:
    anderson  Anderson-accelerated parquet iteration (`solve!`)
    mfRG      preconditioned iteration (`solve_using_mfRG_without_mixing!`)

The DMFT reference (temperature T, chemical potential, hopping and local vertex)
is read from `--dmft`; U and T are taken from it for both strategies.

Usage (from the repo root, multi-threaded):
  julia --project=script -t <nthreads> script/run/run_hubbard_model.jl [options]

Run with `--help` for the full option list and the current defaults. All options
are named, so order never matters and only what you set needs to be passed:

  julia --project=script -t 8 script/run/run_hubbard_model.jl --nmax=12 --nq=6
  julia --project=script -t 8 script/run/run_hubbard_model.jl --strategy=scPA --scheme=anderson
  julia --project=script -t 8 script/run/run_hubbard_model.jl --dmft=2.06 --NL=2

`--dmft` takes a path relative to the repo root, or just the U as shorthand for
`data/high_temperature_U<U>.h5`.

Output — one HDF5 file under `output/hubbard_model/`:
  hubbard_<algo>_U<U>_T<T>_nmax<nmax>_nq<nq>_NL<NL_method>.h5
with <algo> one of sc_anderson, sc_mfRG, fd_anderson, fd_mfRG.

The file holds the full solver state written by `save!` (Gbare, G0, Σ0, F0, Π0pp,
Π0ph, G, Σ, F, Πpp, Πph) plus a `convergence` group (converged flag,
residual_norm, iterations, tol, maxiter). Reload the solver with
`load_hubbard_solver` (load_hubbard_data.jl); read the convergence verdict with
`load_convergence` (common/convergence.jl).
"""

@isdefined(ScriptCommon) || include(joinpath(@__DIR__, "..", "common", "ScriptCommon.jl"))
using .ScriptCommon

# -----------------------------------------------------------------------
# Arguments
# -----------------------------------------------------------------------
# Parsed before the heavy imports below, so `--help` and a mistyped option come
# back immediately instead of after MPI and fdDGAsolver have loaded.
#
# Help text and validation come from the shared spec (common/cli.jl); the
# defaults are the REPL block below.
# ---------------------------------------------------------------------------
# What executing this file in the REPL does. Edit these and execute it again.
#
# Running the file as a script overrides them from the command line - and these
# are the defaults `--help` reports, so the two entry points cannot drift apart.
# ---------------------------------------------------------------------------
nmax             = 10       # half-box size for the vertex grids
nq               = 4        # linear k-grid size for the vertex mesh
NL_method        = 1        # vertex momentum structure: 1 (K2(k,q)≈K2(q) ...
strategy         = :fdPA    # reference vertex F0: scPA (bare, parquet approximation) or ...
iteration_scheme = :mfRG    # how the fixed point is reached: anderson or mfRG
dmft             = "2.089"  # DMFT reference: a path relative to the repo root, or just ...
tol              = 1.0e-8   # residual-norm convergence tolerance
maxiter          = 200      # maximum parquet iterations

# ---------------------------------------------------------------------------
# CLI driver (skipped when this file is executed in the REPL or `include`d)
# ---------------------------------------------------------------------------
if abspath(PROGRAM_FILE) == @__FILE__
    settings = cli_settings("Solve the 2D Hubbard model with the finite-difference parquet solver.")
    add_box_args!(settings; nmax, nq, NL_method)
    add_run_args!(settings; strategy, iteration_scheme)
    add_dmft_arg!(settings; default = dmft)
    add_convergence_args!(settings; tol, maxiter)

    parsed = parse_args(ARGS, settings)

    nmax             = parsed["nmax"]
    nq               = parsed["nq"]
    NL_method        = parsed["NL_method"]
    strategy         = parsed["strategy"]
    iteration_scheme = parsed["iteration_scheme"]
    dmft             = parsed["dmft"]
    tol              = parsed["tol"]
    maxiter          = parsed["maxiter"]
end


# Iterate Σ together with the vertex. Set explicitly rather than left to the
# defaults, which disagree between the two schemes (`solve!` updates Σ, the mfRG
# entry point does not), so that `iteration_scheme` only changes how the fixed
# point is reached and not which fixed point it is.
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

@mainprintf("=== PHS Hubbard model ===\n")
@mainprintf("  T     = %-8.4f  U  = %-8.4f  μ  = %-8.4f  t1 = %.4f\n", T, U, μ, t1)
@mainprintf("  nmax  = %-8d  nq = %-8d  NL = %d\n", nmax, nq, NL_method)
@mainprintf("  tol   = %-8.1e  maxiter = %d\n", tol, maxiter)
@mainprintf("  strategy = :%s,  iteration_scheme = :%s\n", strategy, iteration_scheme)

# -----------------------------------------------------------------------
# Build the solver — `strategy` picks the reference vertex F0, `NL_method` the
# momentum structure of K2/K3. Both live in common/runs.jl.
# -----------------------------------------------------------------------
@mainprintf("\n=== Running %s with %s ===\n", strategy_label(strategy), iteration_scheme)
@mainprintf("Create solver.\n")

S = build_solver_from_dmft(data_triqs, nmax, nq, NL_method; strategy)

# The solver keeps everything it needs (Gbare, G0, Σ0 and the local vertex Γ that
# F0 wraps), so the rest of the DMFT payload — G, G0, Σ, occ — can go.
data_triqs = nothing

@mainprintf("Initialize symmetry group.\n")
t_sym = time()
init_sym_grp!(S)
@mainprintf("Symmetry group initialized in %.3f seconds.\n", time() - t_sym)

# -----------------------------------------------------------------------
# Converge it
# -----------------------------------------------------------------------
@mainprintf("Start solver.\n")
t_solve = time()
result = solve_run!(S; strategy, iteration_scheme, maxiter, tol, update_Σ,
                    verbose = mpi_ismain())
@mainprintf("Solver finished in %.3f seconds.\n", time() - t_solve)

# -----------------------------------------------------------------------
# Save
# -----------------------------------------------------------------------
if mpi_ismain()
    @printf("\n=== Saving %s-data ===\n", ALGO_FAMILY[strategy])
    path  = solver_path(U, T, nmax, nq, NL_method, strategy, iteration_scheme)
    fname = save_run!(path, S, result; tol, maxiter)
    @printf("%s-data saved to %s\n", ALGO_FAMILY[strategy], fname)
end
