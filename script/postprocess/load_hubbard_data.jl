"""
load_hubbard_data.jl

Reload a converged 2D-Hubbard-model solver previously saved by `run_hubbard_model.jl`
(via `save!(f, "S", S)`): reconstruct the matching solver shell and fill it with the
stored data using `load_solver!`. The loaded solver holds data only — call
`init_sym_grp!` before iterating it. The stored convergence verdict is printed on load.

A run is addressed by the same two knobs `run_hubbard_model.jl` writes it with:

  `strategy`         :scPA | :fdPA      — which reference vertex the solver uses
  `iteration_scheme` :anderson | :mfRG  — how the fixed point was reached

Usage (from the repo root):
  julia --project=script script/postprocess/load_hubbard_data.jl --U=<U> --T=<T> [options]

Run with `--help` for the full option list.

  --U, --T         : interaction and temperature in code units. These must reproduce
                     the numbers in the saved filename exactly (e.g. 2.089, 0.5), so
                     they are taken as the strings they are spelled with.
  --nmax, --nq     : vertex box half-size and linear k-grid size.
  --NL             : 1, 2 or 3 — momentum structure of the vertex (see --help).
  --strategy       : scPA (parquet approximation) or fdPA (finite-difference
                     parquet DΓA with DMFT reference). Decides which solver shell is
                     rebuilt here — the two store different things.
  --scheme         : anderson or mfRG. Names the file only: both converge the same
                     fixed point, so the loaded object is structurally identical.
  --dmft           : (fdPA only) path to the DMFT reference data, relative to the
                     repo root. Defaults to data/high_temperature_U<U>.h5.

The solver file is expected at
  output/hubbard_model/hubbard_<algo>_U<U>_T<T>_nmax<nmax>_nq<nq>_NL<NL_method>.h5
where <algo> is the solver family (ALGO_FAMILY[strategy], "sc" or "fd") joined to
the iteration scheme by an underscore — e.g. fd_mfRG.

When run as a script the loaded solver is left in the global `S`. The file can also be
`include`d to use `load_hubbard_solver(...)` directly (the CLI driver is skipped then).
"""

@isdefined(ScriptCommon) || include(joinpath(@__DIR__, "..", "common", "ScriptCommon.jl"))
using .ScriptCommon

using MPI
MPI.Init()

using fdDGAsolver
using MatsubaraFunctions
using HDF5
using Printf

@isdefined(ScriptRuns) || include(joinpath(@__DIR__, "..", "common", "ScriptRuns.jl"))
using .ScriptRuns

# The run vocabulary and `solver_path` come from common/paths.jl, the solver
# shell from `build_solver` (common/runs.jl), and `load_convergence` /
# `report_convergence` from common/convergence.jl.

"""
    load_hubbard_solver(U, T, nmax, nq, NL_method, strategy, iteration_scheme;
                        dmft_path = nothing, data_triqs = nothing,
                        nk_G = 48, mode = :hybrid)

Rebuild the solver shell matching a saved run and load the stored data into it.
Returns the loaded solver (data only — call `init_sym_grp!` before iterating) and
prints its stored convergence verdict.

Only `strategy` decides which shell is built; `iteration_scheme` merely picks which
file is read, since both schemes converge the same fixed point of the same solver.

For `:fdPA` the reference vertex shell is rebuilt from the DMFT data. Callers that
already hold that payload (`precompute_sigma_chi.jl`) pass it as `data_triqs` so the
file is not read a second time; otherwise it is loaded from `dmft_path`
(defaulting to the `data/high_temperature_U<U>.h5` that `dmft_file(U)` names).

The shell itself comes from `build_solver` (common/runs.jl) — the same function
`run_hubbard_model.jl` built the saved run with. That matters because a shell
that does not match the file it loads is not an error `load_solver!` can raise.

`U` and `T` are used TWICE over, and the two uses want different things: the
saved file is addressed by them exactly as its name spells them (they may be
strings), while the constructors want numbers. Both are derived here, so a caller
can pass either.
"""
function load_hubbard_solver(U, T, nmax, nq, NL_method, strategy, iteration_scheme;
                             dmft_path = nothing, data_triqs = nothing,
                             nk_G = 48, mode = :hybrid)
    check_run(strategy, iteration_scheme)

    filename = solver_path(U, T, nmax, nq, NL_method, strategy, iteration_scheme)
    isfile(filename) || error("Saved solver not found:\n  $filename")

    _num(x) = x isa AbstractString ? parse(Float64, x) : float(x)
    U_num, T_num = _num(U), _num(T)

    if strategy === :scPA
        # For the parquet approximation, U/μ/hopping only seed Gbare and F0, which
        # `load_solver!` overwrites — so `build_solver`'s placeholder hopping is
        # fine and only T has to match the saved run.
        S = build_solver(nmax, nq, NL_method; strategy, T = T_num, U = U_num, nk_G, mode)
    else
        # The fd solver's reference vertex wraps the local DMFT vertex, so its
        # structure must be rebuilt from the same DMFT data before loading —
        # reusing the caller's copy when there is one.
        if data_triqs === nothing
            dfile = dmft_path === nothing ? dmft_file(U) : repo_path(dmft_path)
            isfile(dfile) || error("DMFT reference data not found:\n  $dfile")
            data_triqs = load_triqs_data(dfile)
        end
        (; μ, t1, t2, t3) = data_triqs.params

        # G0/Σ0 are overwritten by `load_solver!`, so the zeroed placeholders
        # `build_solver` makes when handed no G_dmft/Σ_dmft are what is wanted.
        S = build_solver(nmax, nq, NL_method; strategy, T = T_num, U = U_num,
                         μ, t1, t2, t3, Γ_dmft = data_triqs.Γ, nk_G, mode)
    end

    load_solver!(S, filename)

    if mpi_ismain()
        println("Loaded $(ALGO_FAMILY[strategy])_$(iteration_scheme)-solver from\n  $filename")
        # `load_convergence` is strict on purpose: a missing group or field means
        # a save that did not finish, and it says so rather than handing back a
        # neutral value. Here that verdict is a REPORT about a load that already
        # succeeded, so it is caught — a solver worth several GB and several
        # minutes must not be discarded because the line describing it could not
        # be printed. The callers that actually act on the verdict
        # (precompute_sigma_chi.jl, run_all_anderson_depth.jl) still let it throw.
        try
            report_convergence(load_convergence(filename))
        catch err
            println("Convergence metadata: unreadable ($(sprint(showerror, err))).")
        end
    end
    return S
end

# ---------------------------------------------------------------------------
# CLI driver (skipped when this file is `include`d)
# ---------------------------------------------------------------------------
# The defaults sit INSIDE this block rather than in a file-scope REPL block like
# every other script's. This file is `include`d by others, and a name defined at
# file scope lands in theirs: `U` and `nmax` here would arrive in the middle of
# a plot script that defines its own. There is nothing to edit-and-rerun anyway —
# the REPL entry point is calling `load_hubbard_solver(...)` directly.
if abspath(PROGRAM_FILE) == @__FILE__
    nmax             = 10      # half-box size for the vertex grids
    nq               = 4       # linear k-grid size for the vertex mesh
    NL_method        = 1       # vertex momentum structure: 1 (K2(k,q)≈K2(q) ...
    strategy         = :fdPA   # reference vertex F0: scPA (bare, parquet approximation) or ...
    iteration_scheme = :mfRG   # how the fixed point is reached: anderson or mfRG
    dmft             = ""      # (fdPA only) DMFT reference path; empty: derive it from U

    settings = cli_settings("Reload a converged 2D-Hubbard-model solver saved by run_hubbard_model.jl.")
    add_arg_table!(settings,
        ["--U"], Dict(:help => "interaction strength, exactly as the saved file name spells it",
                      :arg_type => String, :required => true),
        ["--T"], Dict(:help => "temperature, exactly as the saved file name spells it",
                      :arg_type => String, :required => true))
    add_box_args!(settings; nmax, nq, NL_method)
    add_run_args!(settings; strategy, iteration_scheme)
    add_arg_table!(settings,
        ["--dmft"], Dict(:help => "(fdPA only) DMFT reference path relative to the repo root; " *
                                  "defaults to data/high_temperature_U<U>.h5",
                         :arg_type => String, :default => dmft))

    parsed = parse_args(ARGS, settings)

    U                = parsed["U"]
    T                = parsed["T"]
    nmax             = parsed["nmax"]
    nq               = parsed["nq"]
    NL_method        = parsed["NL_method"]
    strategy         = parsed["strategy"]
    iteration_scheme = parsed["iteration_scheme"]
    dmft_path        = isempty(parsed["dmft"]) ? nothing : parsed["dmft"]

    S = load_hubbard_solver(U, T, nmax, nq, NL_method, strategy, iteration_scheme; dmft_path)

    mpi_ismain() && begin
        @printf "U = %s, T = %s, nmax = %d, nq = %d, NL = %d, strategy = :%s, iteration_scheme = :%s\n" U T nmax nq NL_method strategy iteration_scheme
        println(S)
    end
end
