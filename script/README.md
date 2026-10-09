# `script/`

Scripts are grouped by what they **produce**, which is also the order the pipeline
runs in.

```
(TRIQS, under WSL)    ->  <prefix>_{DCA,chi3_fromDCA,F_fromDCA}.h5   intermediates
fdDGAsolver.parse_triqs_data
                      ->  data/*.h5           DMFT reference
run/                  ->  output/hubbard_model/, output/anderson_depth/,
                          output/parquet_vertex/, output/Chi_{sc,fd,HA}_*.h5
postprocess/          ->  output/postprocess/
plot/                 ->  figures/
```

The first stage is not Julia and not in this folder: TRIQS + CT-HYB write three
intermediate files per parameter set, and `fdDGAsolver.parse_triqs_data` converts
them into the `data/*.h5` everything below reads. TRIQS has no Windows build, so
that stage runs under WSL while everything else runs natively.

> The script that regenerates the intermediates, `script/triqs_dmft_input.py`, is
> **not currently in the checkout** — only a stale `script/__pycache__/*.pyc` of it
> is left, and it is in no commit. The `data/*.h5` already on disk are unaffected;
> regenerating or adding a U needs that script back.

Everything runs under the script environment, from the repo root:

```
julia --project=script -t <nthreads> script/<folder>/<name>.jl [options]
```

Every script takes **named** options and answers `--help` with its full option list
and current defaults. Nothing is positional.

## Folders

| Folder | Holds | Writes |
|---|---|---|
| `run/` | converge the solver; batch drivers that sweep it | saved solver state |
| `postprocess/` | turn saved solvers into cached Σ / χ on the dense grid | precompute caches |
| `plot/` | read caches, draw figures | `figures/*.svg` |
| `analytic/` | analytic Hubbard-atom vertex — a reference, not an entry point | — |
| `common/` | shared modules every script loads | — |
| `upstream/` | unmodified from [jaemolihm/fdDGAsolver.jl](https://github.com/jaemolihm/fdDGAsolver.jl) | — |

Two entries in the table are approximations worth knowing about:

* `run/` is mostly the lattice Hubbard runs, but `generate_vertex.jl` (impurity
  parquet vertex) and `calculate_susceptibilities.jl` (AIM χ(T) scan) also live
  there and write elsewhere under `output/` — see the diagram above.
* `plot/` writes SVGs except for `plot_DMFT_data.jl`, which is an interactive
  GLMakie session, and `plot_vertex_old.jl`, which is legacy PyPlot code kept for
  reference and does not run under `--project=script` (PyPlot is deliberately not
  a dependency). The same holds for `upstream/check_triqs.jl`.

`upstream/` is kept separate so it is obvious which scripts are not ours; note
that moving them does mean their paths differ from the original repo's.

## `common/`

Split by load cost, so a batch driver that only spawns other processes never pays
for loading `fdDGAsolver` (and, through it, MPI):

| File | Holds | Loads `fdDGAsolver`? |
|---|---|---|
| `paths.jl` | repo layout, run vocabulary, file-naming rules, cache paths | no |
| `cli.jl` | the ArgParse spec, sweep-list parsers, child-process launching | no |
| `logging.jl` | `@mainprintf`, table and report formatting | no |
| `convergence.jl` | the `convergence` group: write it, read it, report it | no |
| `plotting.jl` | momentum indexing, labels, the bare bubble, cache discovery | no |
| `runs.jl` | build / converge / save one run | **yes** |

`convergence.jl` is light for a reason: reading a solve's verdict needs HDF5 and
nothing else, so a batch driver can summarise a sweep without loading the package.
`save_run!` — the one step that needs `save!` — stays in `runs.jl`.

Exposed as two modules — `ScriptCommon` (the light five) and `ScriptRuns` (the
heavy one):

```julia
@isdefined(ScriptCommon) || include(joinpath(@__DIR__, "..", "common", "ScriptCommon.jl"))
using .ScriptCommon

# only if the script actually builds or loads a solver:
@isdefined(ScriptRuns) || include(joinpath(@__DIR__, "..", "common", "ScriptRuns.jl"))
using .ScriptRuns
```

The `@isdefined` guard matters because scripts include one another (a plot script
pulls in `postprocess/precompute_sigma_chi.jl`), and a module must not be defined
twice in one session.

## The run vocabulary

A run is addressed by two orthogonal knobs, spelled the same way everywhere —
`--strategy` and `--iteration-scheme` on the command line, `:scPA`/`:fdPA` and
`:anderson`/`:mfRG` in code:

| Knob | Values | Means |
|---|---|---|
| `strategy` | `scPA` / `fdPA` | which reference vertex F0: bare, or the DMFT vertex |
| `iteration_scheme` | `anderson` / `mfRG` | how the fixed point is reached |

`ALGO_FAMILY` in `common/paths.jl` is the only place the short `sc`/`fd` form used
inside file names is produced, and `parse_run_basename` the only place it is read
back — so a name that is written and a name that is matched cannot drift apart.

## Option spelling

`ArgParseSettings(autofix_names = true)` makes **dashes** canonical on the command
line. An option registered as `--NL_method` is typed `--NL-method` and arrives
under the key `"NL_method"`; the underscore spelling is *not* accepted. Short
aliases exist where they read better:

| Canonical | Alias |
|---|---|
| `--NL-method` | `--NL` |
| `--iteration-scheme` | `--scheme` |

Sweep options take comma-separated lists, and the integer ones accept `lo:hi` and
`lo:step:hi` ranges (`--mems=0,1:4,8,16`). `--dmft` takes a path relative to the
repo root, or just the U as shorthand for `data/high_temperature_U<U>.h5`;
`--dmfts` is its comma-separated form.

A singular name always takes one value and a plural name always takes a list.
A script that *sweeps* or draws several values of a knob spells it in the plural:
`--mems`, `--deltas`, `--dmfts`, `--strategies`, `--schemes`, `--sizes`, and
`--nls` (in the plot scripts that draw several momentum structures at once). So `--NL` is always
one NL method and `--nls` always a list, and no script takes both.
