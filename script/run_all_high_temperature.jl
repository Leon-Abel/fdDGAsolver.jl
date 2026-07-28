"""
run_all_high_temperature.jl

Batch driver: run `run_hubbard_model.jl` for every DMFT reference file in
`data/high_temperature_*.h5`.

Each input is run in its own fresh Julia process (run_hubbard_model.jl calls
`MPI.Init()` and reads its parameters from `ARGS`, so it is designed to be
launched as a standalone script). A failing/non-converging input is reported
but does not abort the batch.

Usage (from the repo root):
  julia -t <nthreads> script/run_all_high_temperature.jl [nmax] [nq] [NL_method] [do_PA] [do_fd]

  nmax      : half-box size for vertex grids       (default 10)
  nq        : linear k-grid size for vertex mesh   (default 4)
  NL_method : 1 (NL), 2 (NL2), 3 (NL3)             (default 1)
  do_PA     : run the parquet approximation        (default true)
  do_fd     : run the finite-difference pDΓA        (default true)

The number of threads used for each run matches the threads of this driver
(set via `julia -t <nthreads>`). Outputs land in `output/` as usual.
"""

# ---------------------------------------------------------------------------
# Parameters (forwarded verbatim to run_hubbard_model.jl)
# ---------------------------------------------------------------------------
nmax      = length(ARGS) >= 1 ? ARGS[1] : "10"
nq        = length(ARGS) >= 2 ? ARGS[2] : "4"
NL_method = length(ARGS) >= 3 ? ARGS[3] : "1"
do_PA     = length(ARGS) >= 4 ? ARGS[4] : "false"
do_fd     = length(ARGS) >= 5 ? ARGS[5] : "true"

repo_root = normpath(joinpath(@__DIR__, ".."))
data_dir  = joinpath(repo_root, "data")
script    = joinpath(@__DIR__, "run_hubbard_model.jl")

# ---------------------------------------------------------------------------
# Collect all high-temperature DMFT inputs
# ---------------------------------------------------------------------------
files = sort(filter(readdir(data_dir)) do f
    startswith(f, "high_temperature_") && endswith(f, ".h5")
end)

isempty(files) && error("No high_temperature_*.h5 files found in $data_dir")

nthreads = Threads.nthreads()

println("Running run_hubbard_model.jl for $(length(files)) high_temperature input(s)")
println("  nmax = $nmax, nq = $nq, NL_method = $NL_method, do_PA = $do_PA, do_fd = $do_fd")
println("  threads per run = $nthreads\n")

# ---------------------------------------------------------------------------
# Run each input in a fresh Julia process
# ---------------------------------------------------------------------------
failures = String[]

for (i, f) in enumerate(files)
    # relative path (with forward slashes) understood by run_hubbard_model.jl
    dmft_partial = "data/" * f
    println("[$i/$(length(files))] === $f ===")

    cmd = `$(Base.julia_cmd()) --project=$repo_root -t $nthreads $script $nmax $nq $NL_method $dmft_partial $do_PA $do_fd`
    try
        run(cmd)
        println("[$i/$(length(files))] done: $f\n")
    catch err
        @warn "run_hubbard_model.jl failed for $f" exception = err
        push!(failures, f)
    end
end

# ---------------------------------------------------------------------------
# Summary
# ---------------------------------------------------------------------------
println("\n=== Batch complete: $(length(files) - length(failures))/$(length(files)) succeeded ===")
if !isempty(failures)
    println("Failed inputs:")
    foreach(f -> println("  - $f"), failures)
    exit(1)
end
