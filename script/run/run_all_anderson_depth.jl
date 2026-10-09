"""
run_all_anderson_depth.jl

Batch driver for the Anderson-depth scan: runs `run_anderson_depth.jl` once per
Anderson history depth `mem` (optionally crossed with the mixing β and with
several DMFT inputs), then reads the saved runs back into one comparison table.

`run_anderson_depth.jl` performs exactly one run per invocation — it is
run_hubbard_model.jl with `strategy` fixed to :fdPA and `iteration_scheme` fixed
to :anderson, and with `mem`/`delta` opened up instead. This driver sweeps the
cross product

    depths  ×  mixings  ×  DMFT inputs

launching one fresh Julia process per combination. A failing / non-converging /
out-of-memory combination is reported but does not abort the batch.

What the table shows
--------------------
`conv`, `iters` and `residual` come from the `convergence` group of each saved
run; `mem`, `delta` and `U` are read off the file name. `t_wall` is this driver's
own wall-clock measurement of the child process, so it includes Julia startup and
JIT on top of the solve — treat it as a coarse cost, and compare `iters` when you
want the machine-independent number. Wall times for runs launched in earlier
invocations are carried over from the existing `summary.tsv`, so a scan assembled
in stages with `--reuse` keeps its timings.

Nothing here measures memory: the depth's memory cost is exact arithmetic rather
than something to measure — NLsolve's `AndersonCache` holds `2·mem + 4` vectors
of length n = length(flatten(S)), i.e. 2·mem·n·16 bytes of depth-dependent
working set.

Two caveats worth keeping in mind when reading a scan:

  * `mem` and `delta` interact — a depth that stalls at β = 0.85 may converge at a
    smaller β, so a scan at fixed β says which depths converge at THAT β. Cross a
    couple of β values if the ranking looks marginal.
  * A depth that diverges only at large `mem` may be an ill-conditioning artifact
    of NLsolve's least-squares update rather than physics: `solve!` leaves
    NLsolve's `droptol` at its default (Inf, no column dropping) and does not
    expose it.

Disk note: each run saves the full solver state, which is multi-GB at production
box sizes. A nine-depth scan therefore writes nine of them — size the depth ladder
and the box accordingly, or sweep in stages with `--reuse`.

Usage (from the repo root):
  julia --project=script -t <nthreads> script/run/run_all_anderson_depth.jl [options]

Run with `--help` for the full option list.

  --mems       comma-separated depths / ranges      (default 0,1,2,4,8,16,32,64)
  --deltas     comma-separated Anderson mixing β    (default 0.85)
  --dmfts      comma-separated DMFT inputs — a path, or just the U  (default 2.9)
  --dry-run    list the runs and exit without launching anything
  --reuse      skip combinations whose output file already exists

`--mems` accepts single values and `lo:hi` / `lo:step:hi` ranges, so `0,1:4,8,16`
and `0:8:64` both work.

Examples:
  julia --project=script -t 8 script/run/run_all_anderson_depth.jl
  julia --project=script -t 8 script/run/run_all_anderson_depth.jl --mems=0,2,4,8,16,32 --dmfts=0.5,2.06
  julia --project=script -t 8 script/run/run_all_anderson_depth.jl --mems=4,8,16 --dmfts=2.06 --deltas=0.5,0.85
  julia --project=script -t 8 script/run/run_all_anderson_depth.jl --mems=0:8 --nmax=8 --reuse

Outputs land in `output/anderson_depth/` — one solver file per run (see
run_anderson_depth.jl) plus `summary.tsv`, a tab-separated copy of the table
printed at the end. The table is rebuilt from every matching file present, so
`--reuse` sweeps accumulate into it.
"""

using Printf

@isdefined(ScriptCommon) || include(joinpath(@__DIR__, "..", "common", "ScriptCommon.jl"))
using .ScriptCommon

# ---------------------------------------------------------------------------
# Arguments
# ---------------------------------------------------------------------------
# ---------------------------------------------------------------------------
# What executing this file in the REPL does. Edit these and execute it again.
#
# Running the file as a script overrides them from the command line - and these
# are the defaults `--help` reports, so the two entry points cannot drift apart.
# ---------------------------------------------------------------------------
nmax      = 10                    # half-box size for the vertex grids
nq        = 4                     # linear k-grid size for the vertex mesh
NL_method = 1                     # vertex momentum structure: 1 (K2(k,q)≈K2(q) ...
threads   = Threads.nthreads()    # threads per spawned run (default: this driver's own)
mems      = "0,1,2,4,8,16,32,64"  # comma-separated depths, with `lo:hi` / `lo:step:hi` ranges
deltas    = "0.85"                # comma-separated Anderson mixing β
dmfts     = "2.9"                 # comma-separated DMFT inputs — a path, or just the U
dry_run   = false                 # list the runs and exit without launching anything
reuse     = false                 # skip combinations whose output file already exists

# ---------------------------------------------------------------------------
# CLI driver (skipped when this file is executed in the REPL or `include`d)
# ---------------------------------------------------------------------------
if abspath(PROGRAM_FILE) == @__FILE__
    settings = cli_settings("Sweep run_anderson_depth.jl over Anderson history depths.")
    add_box_args!(settings; nmax, nq, NL_method)
    add_threads_arg!(settings; threads)
    add_arg_table!(settings,
        ["--mems"],   Dict(:help => "comma-separated depths, with `lo:hi` / `lo:step:hi` ranges",
                           :arg_type => String, :default => mems),
        ["--deltas"], Dict(:help => "comma-separated Anderson mixing β",
                           :arg_type => String, :default => deltas),
        ["--dmfts"],  Dict(:help => "comma-separated DMFT inputs — a path, or just the U",
                           :arg_type => String, :default => dmfts))
    add_flags!(settings,
        "dry-run" => "list the runs and exit without launching anything",
        "reuse"   => "skip combinations whose output file already exists")

    parsed = parse_args(ARGS, settings)

    nmax      = parsed["nmax"]
    nq        = parsed["nq"]
    NL_method = parsed["NL_method"]
    threads   = parsed["threads"]
    mems      = parsed["mems"]
    deltas    = parsed["deltas"]
    dmfts     = parsed["dmfts"]
    dry_run   = parsed["dry_run"]
    reuse     = parsed["reuse"]
end

nthreads  = threads
# Validated here as well as in the child, so a typo in a sweep list fails before
# the earlier (long) runs rather than after them.
mems      = parse_ints(mems)
deltas    = parse_floats(deltas)
dmfts     = parse_dmft_inputs(dmfts)

all(>=(0), mems) || error("mem must be >= 0, got $(filter(<(0), mems)).")
all(d -> 0 < d <= 1, deltas) ||
    error("deltas must lie in (0, 1], got $(filter(d -> !(0 < d <= 1), deltas)).")

script     = joinpath(@__DIR__, "run_anderson_depth.jl")
out_dir    = anderson_depth_dir()

missing_inputs = [f for f in dmfts if !isfile(repo_path(f))]
isempty(missing_inputs) || error("DMFT input(s) not found: $(join(missing_inputs, ", ")).")

mkpath(out_dir)
const SUMMARY_PATH = joinpath(out_dir, "summary.tsv")

# ---------------------------------------------------------------------------
# Runs to launch. `mem` varies fastest so one input/β block stays together in
# the log and in the table.
# ---------------------------------------------------------------------------
runs = [(f, d, m) for f in dmfts for d in deltas for m in mems]

println("Anderson-depth scan: $(length(runs)) run(s)")
println("  mems     = $(join(mems, ", "))")
println("  deltas   = $(join(deltas, ", "))")
println("  inputs   = $(join(dmfts, ", "))")
println("  nmax = $nmax, nq = $nq, NL_method = $NL_method")
println("  threads per run = $nthreads")
println("  output = $out_dir")
println()

if dry_run
    for (i, (f, d, m)) in enumerate(runs)
        println("[$i/$(length(runs))] mem = $m, delta = $d, $f")
    end
    println("\n--dry-run: nothing launched.")
    exit(0)
end

# ---------------------------------------------------------------------------
# Wall times, keyed by the (mem, delta, U) a table row is identified by.
# Seeded from the previous summary.tsv so timings measured in earlier `--reuse`
# invocations survive; runs launched below overwrite their own entry.
# ---------------------------------------------------------------------------
wall_times = Dict{Tuple{String, String, String}, Float64}()

if isfile(SUMMARY_PATH)
    try
        for (i, line) in enumerate(eachline(SUMMARY_PATH))
            i == 1 && continue                      # header
            cols = split(line, '\t')
            length(cols) >= 7 || continue           # pre-t_wall summary: nothing to carry
            t = tryparse(Float64, cols[7])
            t === nothing && continue               # "-" for a row never timed here
            wall_times[(cols[1], cols[2], cols[3])] = t
        end
    catch err
        @warn "Could not read wall times from $SUMMARY_PATH; they will be re-measured." exception = err
    end
end

# ---------------------------------------------------------------------------
# Launch, one fresh process per combination
# ---------------------------------------------------------------------------
# `--reuse` and the summary table both need to recognise the child's output
# before/without opening it. `anderson_depth_re` (common/paths.jl) is built from
# the same rule `anderson_depth_basename` writes the name with, so the two cannot
# drift. U and T come from the DMFT file, hence their wildcards.
function existing_run(m, d)
    pat  = anderson_depth_re(nmax, nq, NL_method; mem = string(m),
                             delta = replace(string(d), "." => "\\."))
    hits = filter(f -> occursin(pat, f), readdir(out_dir))
    return isempty(hits) ? nothing : joinpath(out_dir, first(hits))
end

failures = Tuple{Int, Float64, String}[]
skipped  = 0

for (i, (f, d, m)) in enumerate(runs)
    label = "mem = $m, delta = $d, $(basename(f))"

    if reuse
        hit = existing_run(m, d)
        if hit !== nothing
            println("[$i/$(length(runs))] skip (exists): $label -> $(basename(hit))")
            global skipped += 1
            continue
        end
    end

    println("[$i/$(length(runs))] === $label ===")
    args = child_args(; mem = m, delta = d, nmax, nq, NL_method, dmft = f)
    t0 = time()
    try
        run(julia_cmd(script, args; threads = nthreads))
        elapsed = time() - t0
        @printf("[%d/%d] done in %.1f s: %s\n\n", i, length(runs), elapsed, label)

        # Key off the file the child actually wrote, so the U in the key matches
        # the one the table reads back out of the name.
        written = existing_run(m, d)
        if written !== nothing
            mu = match(r"_U([0-9.]+)_T", basename(written))
            mu === nothing || (wall_times[(string(m), @sprintf("%.2f", d), mu.captures[1])] = elapsed)
        end
    catch err
        # A depth that exhausts memory or produces non-finite residuals takes
        # down its own process only; the remaining depths are still informative.
        @warn "run_anderson_depth.jl failed for $label" exception = err
        push!(failures, (m, d, f))
    end
end

# ---------------------------------------------------------------------------
# Comparison table, read back from every saved run in the output directory whose
# box size matches this sweep (so --reuse runs accumulate).
#
# Only the small `convergence` group is read, never the solver state, so this
# stays fast even though the files are multi-GB.
# ---------------------------------------------------------------------------
"""
    read_run(path) -> NamedTuple

One row of the table: the convergence verdict from the file, and mem / delta / U
from its name. `U` is kept as the string it is spelled with, so rows group by
input without opening the DMFT file.

`load_convergence` (common/convergence.jl) is the reader that matches the
`save_convergence!` the child wrote with. It lives in the light half precisely so
this driver can use it — reading the group needs HDF5 and nothing else, and this
process exists only to spawn others.

It throws for a file whose save did not complete; the loop below catches that and
names the file, so one unreadable run costs its row and not the table.
"""
function read_run(path)
    name = basename(path)
    grab(re, default) = (m = match(re, name); m === nothing ? default : m.captures[1])

    conv = load_convergence(path)

    return (; path,
        mem       = grab(r"_mem(\d+)_", "?"),
        delta     = grab(r"_delta([0-9.]+)_", "?"),
        U         = grab(r"_U([0-9.]+)_T", "?"),
        converged = conv.converged,
        iters     = conv.iterations,
        residual  = conv.residual_norm,
        maxiter   = conv.maxiter)
end

const ROW_FILE = anderson_depth_re(nmax, nq, NL_method)

rows = Any[]
for file in sort(readdir(out_dir))
    occursin(ROW_FILE, file) || continue
    try
        push!(rows, read_run(joinpath(out_dir, file)))
    catch err
        @warn "Could not read $file" exception = err
    end
end

# Depth varies fastest within an (input, β) block, matching the launch order.
# mem and delta are strings off the file name, so sort them numerically.
numeric(s, default) = (v = tryparse(Float64, s); v === nothing ? default : v)
sort!(rows; by = r -> (r.U, numeric(r.delta, Inf), numeric(r.mem, Inf)))

header = ("mem", "delta", "U", "conv", "iters", "residual", "t_wall")

function row_cells(r)
    # delta is normalised to two decimals so the key matches what the launch loop
    # stored, whatever the file name happens to spell (0.85 / 0.850).
    dkey = (v = tryparse(Float64, r.delta); v === nothing ? r.delta : @sprintf("%.2f", v))
    t    = get(wall_times, (r.mem, dkey, r.U), nothing)
    return (r.mem,
            dkey,
            r.U,
            r.converged ? "yes" : "NO",
            "$(r.iters)/$(r.maxiter)",
            @sprintf("%.2e", r.residual),
            t === nothing ? "-" : @sprintf("%.1f", t))
end

println("\n=== Anderson depth scan: nmax = $nmax, nq = $nq, NL = $NL_method ===")

if isempty(rows)
    println("No saved runs to summarise.")
else
    cells = [row_cells(r) for r in rows]
    print_table(header, cells)

    println("\n  t_wall = whole child process in seconds (Julia startup and JIT included),")
    println("  `-` where this driver did not launch the run itself; compare `iters`")
    println("  for the machine-independent number.")

    write_tsv(SUMMARY_PATH, header, cells)
    println("\nTable written to $SUMMARY_PATH")
end

# ---------------------------------------------------------------------------
# Summary
# ---------------------------------------------------------------------------
attempted = length(runs) - skipped
println("\n=== Scan complete: " *
        (attempted == 0 ? "nothing to launch" :
         "$(attempted - length(failures))/$attempted launched run(s) succeeded") *
        (skipped > 0 ? ", $skipped skipped (--reuse)" : "") * " ===")
if !isempty(failures)
    println("Failed runs:")
    for (m, d, f) in failures
        println("  - mem = $m, delta = $d, $(basename(f))")
    end
    exit(1)
end
