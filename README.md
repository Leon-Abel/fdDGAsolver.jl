# fdDGAsolver


## Installation

Requires **Julia ≥ 1.11**. Older versions don't support the `[sources]` section, which this project uses to pin its unregistered dependency.

If you don't have Julia yet, the easiest way is [juliaup](https://github.com/JuliaLang/juliaup):

```bash
# Linux / macOS
curl -fsSL https://install.julialang.org | sh
```

```powershell
# Windows
winget install --name Julia --id 9NJNWW8PVKMN -e -s msstore
```

Check your version with `julia --version`.

### As a package

`MatsubaraFunctions.jl` is not registered, and the code needs its `mesh_generalization_v3` branch. Add it first, then add this package. Run the following in the Julia REPL (same on all platforms):

```julia
using Pkg
Pkg.add(url="https://github.com/dominikkiese/MatsubaraFunctions.jl", rev="mesh_generalization_v3")
Pkg.add(url="https://github.com/Leon-Abel/fdDGAsolver.jl")
```

### From a clone (to run the scripts or develop)

The repo has two Julia environments:

| Environment | Path | Purpose |
|---|---|---|
| package | `Project.toml` | the solver library itself |
| scripts | `script/Project.toml` | run, postprocessing and plotting scripts (adds Makie, ArgParse, …) |

Both pick up `MatsubaraFunctions` from their `[sources]` entry. The script environment links to the local checkout of `fdDGAsolver` (`path = ".."`), so your edits to `src/` take effect right away.

**Linux / macOS (bash, zsh):**

```bash
git clone https://github.com/Leon-Abel/fdDGAsolver.jl
cd fdDGAsolver.jl

julia --project=.      -e 'using Pkg; Pkg.instantiate()'   # package
julia --project=script -e 'using Pkg; Pkg.instantiate()'   # scripts
```

**Windows (PowerShell or cmd):**

```powershell
git clone https://github.com/Leon-Abel/fdDGAsolver.jl
cd fdDGAsolver.jl

julia --project=.      -e "using Pkg; Pkg.instantiate()"   # package
julia --project=script -e "using Pkg; Pkg.instantiate()"   # scripts
```

Run scripts from the repo root, inside the script environment:

```bash
# Linux / macOS
julia --project=script -t <nthreads> script/<folder>/<name>.jl [options]
```

```powershell
# Windows
julia --project=script -t <nthreads> script\<folder>\<name>.jl [options]
```

Forward slashes in paths also work on Windows. To use all available cores, pass `-t auto`.

Every script lists its options with `--help`. See [`script/README.md`](script/README.md) for the full pipeline.
