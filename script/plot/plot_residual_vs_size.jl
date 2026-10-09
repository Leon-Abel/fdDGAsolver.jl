"""
plot_residual_vs_size.jl

Convergence diagnostic: the final parquet residual norm ‖ΔS‖_∞ versus *system size*
(the (nmax, nq) pair), with one curve per interaction strength U — one panel per
run (strategy × iteration scheme) sharing a common log y-axis.

The question this answers is whether the solver's failure to converge at large U is
a box-size artifact that a bigger box would cure. Read horizontally: a curve that
slopes down to the right means enlarging the box helps; a flat curve means the box
is not what is limiting the solve. Read vertically against the dashed `tol` line:
everything above it is a run that exhausted `maxiter` without reaching the solver's
own stopping tolerance. Each marker is annotated with the number of iterations the
run took (from the same `convergence` group), in the hue of its curve.

One curve per U in Makie's default categorical palette (Wong's colorblind-safe
seven), every curve carrying the same default marker, so the only thing separating
the curves is hue. The palette holds seven distinct hues and then wraps, so asking
for many more U than that will repeat colors; prefer a representative handful.

The y-axis is labelled every *second* decade (… 10², 10⁰, 10⁻², 10⁻⁴ …), anchored
on even exponents so the same decades are named no matter what range the data
happens to cover. The unlabelled decades are still on the axis — it is continuous —
they simply carry no tick.

Unlike the other plot scripts this one reads *only* convergence metadata, never
vertex data, so it never loads (or precomputes) a solver: the residual is taken
from the `convergence` group of a precompute cache when one exists, and otherwise
from the same group in the saved solver itself. Nothing here is slow.

Usage (from the repo root):
  julia --project=script script/plot/plot_residual_vs_size.jl [options]

Run with `--help` for the full option list and the current defaults.

  --NL           NL method: 1 = K23 s-wave, 2 = K3 s-wave, 3 = full q
  --Us           "all", or U values as spelled in the file names, e.g.
                 "0.5,1.25,2.089" (default: a 6-U spread across the weak /
                 critical / strong regimes). Note U/t = 4·U.
  --strategies   comma-separated scPA | fdPA
  --schemes      comma-separated anderson | mfRG — one panel per
                 (strategy, scheme); add both to compare the two routes
  --sizes        "all", or an explicit list "nmax:nq,nmax:nq,…". "all" drops
                 sizes carrying fewer than 2 of the selected U (a lone marker,
                 not a curve); naming such a size explicitly forces it in.
  --tol          solver stopping tolerance, drawn as the reference line

  NL1 is the default because it is the only method run at every box size; NL2 stops
  at nmax=14 and NL3 at nmax=10, so those give a narrower x-axis.

  Examples:
    julia --project=script script/plot/plot_residual_vs_size.jl
    julia --project=script script/plot/plot_residual_vs_size.jl --Us=all --strategies=fdPA
    julia --project=script script/plot/plot_residual_vs_size.jl --NL=2 --Us=0.5,1.25,2.06,2.089 --strategies=fdPA
    julia --project=script script/plot/plot_residual_vs_size.jl --Us=0.5,2.089,2.85 --sizes=10:4,14:8,20:14
    julia --project=script script/plot/plot_residual_vs_size.jl --schemes=anderson,mfRG   # compare the routes
"""

using HDF5
using CairoMakie
using Printf
using LaTeXStrings

# Only the layout and naming rules are needed here — no solver is ever loaded, so
# this script stays clear of fdDGAsolver (and MPI) entirely.
@isdefined(ScriptCommon) || include(joinpath(@__DIR__, "..", "common", "ScriptCommon.jl"))
using .ScriptCommon

# ---------------------------------------------------------------------------
# What executing this file in the REPL does. Edit these and execute it again.
#
# Running the file as a script overrides them from the command line - and these
# are the defaults `--help` reports, so the two entry points cannot drift apart.
# ---------------------------------------------------------------------------
NL_method  = 1            # NL method: 1 = K23 s-wave, 2 = K3 s-wave, 3 = full q
Us_arg     = "default"    # "all", or U values as spelled in the file names (default: a ...
strategies = "fdPA"       # comma-separated scPA | fdPA
schemes    = "anderson,mfRG"  # comma-separated anderson | mfRG — one panel per (strategy ...
sizes_arg  = "all"        # "all", or an explicit list "nmax:nq,nmax:nq,…"
tol        = 1.0e-8       # solver stopping tolerance, drawn as the reference line

# ---------------------------------------------------------------------------
# CLI driver (skipped when this file is executed in the REPL or `include`d)
# ---------------------------------------------------------------------------
if abspath(PROGRAM_FILE) == @__FILE__
    settings = cli_settings("Plot the solver residual against vertex box size.")
    add_arg_table!(settings,
        ["--NL_method", "--NL"],
                  Dict(:help => "NL method: 1 = K23 s-wave, 2 = K3 s-wave, 3 = full q",
                       :arg_type => Int, :default => NL_method, :range_tester => in(1:3)),
        ["--Us"], Dict(:help => "\"all\", or U values as spelled in the file names " *
                                "(default: a 6-U spread across the weak / critical / strong regimes)",
                       :arg_type => String, :default => Us_arg),
        ["--strategies"], Dict(:help => "comma-separated scPA | fdPA",
                               :arg_type => String, :default => strategies),
        ["--schemes"],    Dict(:help => "comma-separated anderson | mfRG — one panel per " *
                                        "(strategy, scheme); add both to compare the two routes",
                               :arg_type => String, :default => schemes),
        ["--sizes"],      Dict(:help => "\"all\", or an explicit list \"nmax:nq,nmax:nq,…\"",
                               :arg_type => String, :default => sizes_arg),
        ["--tol"],        Dict(:help => "solver stopping tolerance, drawn as the reference line",
                               :arg_type => Float64, :default => tol))

    parsed = parse_args(ARGS, settings)

    NL_method  = parsed["NL_method"]
    Us_arg     = parsed["Us"]
    strategies = parsed["strategies"]
    schemes    = parsed["schemes"]
    sizes_arg  = parsed["sizes"]
    tol        = parsed["tol"]
end

# One panel per (strategy, iteration_scheme), identified by the same tag the file
# names carry (e.g. "fd_mfRG").
run_tag(strategy, scheme) = "$(ALGO_FAMILY[strategy])_$(scheme)"

SCHEME_LABEL = Dict(:anderson => "Anderson", :mfRG => "mfRG preconditioning")

# A representative spread when the caller does not name U explicitly: two well inside
# the convergent regime, one just below the wall, two in the critical window around
# the detQMC benchmark (U/t = 8.356), one deep in the divergent regime.
default_Us = ["0.5", "1.25", "1.75", "2.06", "2.089", "2.85"]

# -----------------------------------------------------------------------
# Scan the residual metadata
# -----------------------------------------------------------------------
# A run is discoverable either through a precompute cache (output/postprocess/) or
# through the saved solver itself (output/hubbard_model/) — the large solvers are
# often deleted once cached, and the small ones often never get a cache. Both carry
# the same `convergence` group (the cache a copy of its solver's), so
# `load_convergence` reads either; a file missing it throws rather than being
# plotted as a converged point.

# Every (run, U, nmax, nq) on NL method `NL_method` that has a residual on disk, keyed
# (run_tag, Ustr, size) => residual. The cache dir is scanned first so it wins over
# the solver file when both exist; they agree by construction (precompute copies it).
# `parse_run_basename` (common/paths.jl) is the inverse of the rule that wrote the
# names, so the two cannot drift apart.
#
# The iteration count of each run is collected alongside, under the same key, for the
# per-marker annotations.
function scan_residuals(NL_method)
    res   = Dict{Tuple{String, String, Tuple{Int, Int}}, Float64}()
    iters = Dict{Tuple{String, String, Tuple{Int, Int}}, Int}()
    for dir in (postprocess_dir(), solver_dir())
        isdir(dir) || continue
        for f in readdir(dir)
            r = parse_run_basename(f)
            r === nothing && continue
            r.NL == NL_method || continue
            (r.strategy === nothing || r.iteration_scheme === nothing) && continue
            key = (run_tag(r.strategy, r.iteration_scheme), r.U_str, (r.nmax, r.nq))
            haskey(res, key) && continue                 # cache dir already supplied it
            try
                conv = load_convergence(joinpath(dir, f))
                res[key]   = conv.residual_norm
                iters[key] = conv.iterations
            catch err
                @warn "Could not read the residual from $f; skipping it." exception = err
            end
        end
    end
    return res, iters
end

"""
    load_data(; NL_method, Us_arg, strategies, schemes, sizes_arg) -> NamedTuple

Scan the residual metadata and select the U and the system sizes to draw. Each
keyword defaults to the parameter block above, so `load_data()` loads exactly what
executing this file would. The tolerance is NOT one of them: it only draws the
reference line, so `plot_data` can move it without this running again.
"""
function load_data(; NL_method  = NL_method,
                     Us_arg     = Us_arg,
                     strategies = strategies,
                     schemes    = schemes,
                     sizes_arg  = sizes_arg)

    strats_sel  = parse_strategies(strategies)
    schemes_sel = parse_schemes(schemes)

    runs_sel = [(s, i) for s in strats_sel for i in schemes_sel]
    algos    = [run_tag(s, i) for (s, i) in runs_sel]

    # Panel label for a run tag. It names only what differs between the panels: the
    # scheme when a single strategy is compared across schemes ("fd_mfRG" -> "mfRG
    # preconditioning"), the strategy when a single scheme is ("pDΓA"), and both
    # otherwise ("pDΓA, Anderson"). What is common to every panel goes in the figure
    # subtitle instead (`common_tag`).
    algo_label = Dict(run_tag(s, i) =>
            (length(strats_sel) == 1 && length(schemes_sel) > 1 ? SCHEME_LABEL[i] :
             length(schemes_sel) == 1                             ? strategy_label(s) :
                                                                    "$(strategy_label(s)), $(SCHEME_LABEL[i])")
            for (s, i) in runs_sel)
    common_tag = length(strats_sel) == 1 && length(schemes_sel) > 1 ? strategy_label(only(strats_sel)) :
                 length(schemes_sel) == 1                             ? SCHEME_LABEL[only(schemes_sel)] : nothing

    nl_name = Dict(1 => "K23 s-wave", 2 => "K3 s-wave", 3 => "full q")[NL_method]

    residuals, iterations = scan_residuals(NL_method)
    isempty(residuals) && error("No NL$(NL_method) runs with recorded residuals found in\n  $(output_dir())")

    # -----------------------------------------------------------------------
    # Select the U and the sizes to draw
    # -----------------------------------------------------------------------
    all_U = Set(k[2] for k in keys(residuals) if k[1] in algos)

    Us = if Us_arg == "all"
        sort!(collect(all_U); by = u -> parse(Float64, u))
    else
        want = Us_arg == "default" ? default_Us : [String(strip(u)) for u in split(Us_arg, ",") if !isempty(strip(u))]
        keep = String[]
        for u in want
            if u in all_U
                push!(keep, u)
            else
                @warn "No NL$(NL_method) run at U=$u for the selected algorithm(s); skipping it."
            end
        end
        sort!(keep; by = u -> parse(Float64, u))
    end
    isempty(Us) && error("None of the requested U has NL$(NL_method) data.")

    # The unit U and T are quoted in. Read off the reference files rather than written
    # here as `t = 0.25`: only the scalars are read, no vertex load, and a constant is
    # only right for as long as every input agrees with it. (t1 is stored negative;
    # the axis unit is its magnitude — see `dmft_hopping`.)
    _mp = dmft_model_params(Us)
    t   = _mp === nothing ? 0.25 : abs(_mp.t1)
    T   = _mp === nothing ? nothing : _mp.T

    if length(Us) > 7
        @warn "$(length(Us)) U selected; the default palette holds 7 hues and then repeats. Consider naming fewer."
    end

    # Sizes present for the selected algorithms *and* the selected U.
    Uset = Set(Us)
    found_sizes = sort!(unique(k[3] for k in keys(residuals) if k[1] in algos && k[2] in Uset))

    # Number of selected U carrying data at a size, over any selected algorithm.
    n_at(s) = count(u -> any(haskey(residuals, (a, u, s)) for a in algos), Us)

    sizes = if sizes_arg == "all"
        keep = Tuple{Int, Int}[]
        for s in found_sizes
            if n_at(s) >= 2
                push!(keep, s)
            else
                @warn "Fewer than 2 selected U at nmax=$(s[1]), nq=$(s[2]); dropping it (name it in `sizes` to force it in)."
            end
        end
        keep
    else
        want = Tuple{Int, Int}[]
        for tok in split(sizes_arg, ",")
            isempty(strip(tok)) && continue
            parts = split(strip(tok), ":")
            length(parts) == 2 || error("Bad size \"$tok\": expected nmax:nq (e.g. 10:4).")
            push!(want, (parse(Int, parts[1]), parse(Int, parts[2])))
        end
        keep = Tuple{Int, Int}[]
        for s in want
            if s in found_sizes
                push!(keep, s)
            else
                @warn "No NL$(NL_method) data at nmax=$(s[1]), nq=$(s[2]) for the selection; skipping it."
            end
        end
        keep
    end
    isempty(sizes) && error("None of the requested system sizes has data.")

    return (; residuals, iterations, Us, sizes, algos, algo_label, common_tag,
              NL_method, nl_name, t, T)
end

# -----------------------------------------------------------------------
# Plot helpers
# -----------------------------------------------------------------------
# Makie's default categorical palette, taken explicitly rather than left to the
# axis cycler so the two panels assign the same hue to the same U regardless of
# which curves happen to have data. It wraps after 7 (see the warning in `load_data`).
U_colors(n) = [Makie.wong_colors()[mod1(i, 7)] for i in 1:n]

# Label every second decade, on even exponents (… 10², 10⁰, 10⁻², 10⁻⁴ …), so the
# same decades are named whatever range the data covers. The fallback keeps the axis
# from ending up tickless when the limits are too narrow to contain an even exponent.
function decade_ticks(lo, hi)
    decades = ceil(Int, log10(lo)):floor(Int, log10(hi))
    es = filter(iseven, decades)
    isempty(es) && (es = collect(decades))
    return (10.0 .^ es, [L"10^{%$e}" for e in es])
end

# Vertical positions for labels wanted at `ys` so that no two are closer than `gap`
# (all in log10 units). Overlapping labels are merged into a block, evenly spaced by
# `gap` and centred on the mean of the positions it replaces, and blocks are merged
# until none overlap, so each label moves the minimum needed to clear its neighbours.
function spread_labels(ys, gap)
    order  = sortperm(ys)
    blocks = [[k] for k in order]                       # each block: label indices, ascending y
    centre(b) = sum(ys[b]) / length(b)
    lo(b) = centre(b) - gap * (length(b) - 1) / 2
    hi(b) = centre(b) + gap * (length(b) - 1) / 2
    merged = true
    while merged
        merged = false
        for k in 1:length(blocks) - 1
            if hi(blocks[k]) + gap > lo(blocks[k + 1])
                blocks[k] = vcat(blocks[k], blocks[k + 1])
                deleteat!(blocks, k + 1)
                merged = true
                break
            end
        end
    end
    out = similar(ys, Float64)
    for b in blocks, (n, k) in enumerate(b)
        out[k] = lo(b) + gap * (n - 1)
    end
    return out
end

# A horizontal Makie legend fills its banks *column*-first: entry n lands at row
# mod1(n, nbanks), column cld(n, nbanks). Permuting the entries into that order
# makes the legend *read* row-first — U ascending left to right, then wrapping to
# the next row. The bounds check is a safety net for bank counts that leave a hole
# in the grid (nbanks > ncols), which the column-first fill cannot express.
function rowfirst(idx, nbanks)
    n = length(idx)
    nbanks <= 1 && return idx
    ncols = cld(n, nbanks)
    perm = [(mod1(k, nbanks) - 1) * ncols + cld(k, nbanks) for k in 1:n]
    all(p -> 1 <= p <= n, perm) || return idx
    return idx[perm]
end

"""
    plot_data(d; tol) -> Figure

Report, draw, save and display the figure from `load_data`'s result. Nothing here
re-reads a file, so retuning the layout costs the drawing alone:

    julia> data = load_data()
    julia> plot_data(data; tol = 1e-6)
"""
function plot_data(d; tol = tol)
    residuals, iterations = d.residuals, d.iterations
    Us, sizes, algos      = d.Us, d.sizes, d.algos
    NL_method, nl_name, t = d.NL_method, d.nl_name, d.t
    _algo_name(a)         = d.algo_label[a]

    println("NL$(NL_method) ($nl_name),  algorithms: ", join(_algo_name.(algos), " + "))
    println("System sizes: ", join(["(nmax=$(s[1]), nq=$(s[2]))" for s in sizes], ", "))
    println("U/t: ", join([string(round(parse(Float64, u) / t, digits = 3)) for u in Us], ", "))
    @printf "Reference tolerance: %.1e\n\n" tol

    # Report what is being drawn, so the figure is never the only record.
    for algo in algos
        println("  $(_algo_name(algo)) ($algo):")
        for u in Us
            cells = String[]
            for s in sizes
                r = get(residuals, (algo, u, s), nothing)
                push!(cells, r === nothing ? @sprintf("%18s", "-") :
                    @sprintf("%18s", @sprintf("%.2e%s (%d)", r, r <= tol ? " " : "*", iterations[(algo, u, s)])))
            end
            @printf("    U/t=%-7.3f%s\n", parse(Float64, u) / t, join(cells))
        end
    end
    println("\n  (* = residual above tol, i.e. the run stopped at maxiter;  (n) = iterations)\n")

    # x is ordinal: the sizes are (nmax, nq) *pairs* with no single scalar measure, so
    # they are placed at 1, 2, 3, … and labelled, rather than positioned by nmax.
    xs   = 1:length(sizes)
    cols = U_colors(length(Us))
    Uset = Set(Us)

    # tol as a LaTeX power of ten (e.g. 1e-8 -> "10^{-8}") for the band annotation.
    tol_tex = let e = round(Int, log10(tol))
        isapprox(tol, 10.0^e; rtol = 1e-12) ? "10^{$e}" : @sprintf("%.1e", tol)
    end

    println("Plotting Figure:")
    # font, line and marker sizes tuned for projection on a presentation slide
    figure = with_theme(theme_latexfonts();
        fontsize = 22,
        Axis   = (titlesize = 26, xlabelsize = 26, ylabelsize = 26,
                  xticklabelsize = 22, yticklabelsize = 22),
        Legend = (labelsize = 20,),
        Lines        = (linewidth = 3,),
        ScatterLines = (linewidth = 3, markersize = 12),
    ) do
        fig = Figure(size = (640 * length(algos) + 60, 760))

        # Common y-limits across the panels: the comparison between algorithms is the
        # point of the figure, so both must be read against the same decades. Padded by
        # half a decade either side, and always wide enough to show the tol line.
        allres = [r for ((a, u, s), r) in residuals if a in algos && u in Uset && s in sizes]
        ylo = 10.0^(floor(log10(min(minimum(allres), tol))) - 0.5)
        yhi = 10.0^(ceil(log10(max(maximum(allres), tol))) + 0.5)

        # Minimum vertical distance between iteration labels, in decades: one 15 pt line of
        # text plus a pixel, over the ~490 pt the axis gets in a 760 pt tall figure.
        label_gap = 16 / 490 * log10(yhi / ylo)

        axis_kw = (
            xticksmirrored = true,
            xtickalign     = 1.0,
            yticksmirrored = true,
            ytickalign     = 1.0,
            yscale         = log10,
            xticks         = (collect(xs), [L"(%$(s[1]),\, %$(s[2]))" for s in sizes]),
            yticks         = decade_ticks(ylo, yhi),
        )

        # Legend entries are taken from the curves of the first panel as they are drawn, so
        # they can be reordered before handing them to the Legend (see `rowfirst`).
        leg_plots  = Any[]
        leg_labels = LaTeXString[]

        axes = Axis[]
        for (j, algo) in enumerate(algos)
            ax = Axis(fig[2, j]; axis_kw...,
                title  = L"\text{%$(_algo_name(algo))}",
                xlabel = L"\text{system size } (n_\mathrm{max},\, n_q)",
                ylabel = j == 1 ? L"\text{residual } \|\Delta S\|_\infty" : "")
            push!(axes, ax)
            xlims!(ax, 0.5, length(sizes) + 0.5)
            ylims!(ax, ylo, yhi)
            j > 1 && hideydecorations!(ax; grid = false, minorgrid = false, ticks = false, minorticks = false)

            # Everything above `tol` is a run that hit maxiter without converging. The band
            # is what carries that reading; keep it recessive so the curves stay dominant.
            band!(ax, [0.5, length(sizes) + 0.5], fill(tol, 2), fill(yhi, 2);
                color = (:gray, 0.10))
            hlines!(ax, [tol]; color = :gray, linestyle = :dash, linewidth = 2)

            for (i, u) in enumerate(Us)
                # Sizes at which this (algo, U) actually ran; the line simply spans gaps,
                # and the markers show which boxes carry data.
                pts = [(x, residuals[(algo, u, s)]) for (x, s) in zip(xs, sizes)
                       if haskey(residuals, (algo, u, s))]
                isempty(pts) && continue
                lab = L"%$(round(parse(Float64, u) / t, digits = 3))"
                p = scatterlines!(ax, first.(pts), last.(pts); label = lab, color = cols[i])
                if j == 1
                    push!(leg_plots, p)
                    push!(leg_labels, lab)
                end
            end

            # Iteration count of each run, just right of its marker in the curve's hue. At
            # one size the markers of different U are often closer than a line of text, so
            # the labels there are spread apart vertically (`spread_labels`); the hue says
            # which marker a displaced label belongs to. The white outline keeps the
            # numbers legible where a curve runs underneath them. CairoMakie strokes text
            # *over* its fill, so the outline is a separate white layer drawn first.
            for (x, s) in zip(xs, sizes)
                idx = [i for (i, u) in enumerate(Us) if haskey(residuals, (algo, u, s))]
                isempty(idx) && continue
                ly  = spread_labels([log10(residuals[(algo, Us[i], s)]) for i in idx], label_gap)
                lbl = [string(iterations[(algo, Us[i], s)]) for i in idx]
                kw  = (text = lbl, align = (:left, :center), offset = (8, 0), fontsize = 15)
                text!(ax, fill(x, length(idx)), 10.0 .^ ly; kw...,
                    color = :white, strokecolor = :white, strokewidth = 6)
                text!(ax, fill(x, length(idx)), 10.0 .^ ly; kw..., color = cols[idx])
            end

            # Name the band once, on the leftmost panel only. `tol_tex` is built outside the
            # L"..." because interpolation there takes an expression, not a macro call. Pinned
            # to the top-left corner (the band always reaches the top of the axis) rather than
            # just above the tol line, where the labels of runs that barely converged sit.
            j == 1 && text!(ax, 0.015, 0.985; space = :relative,
                text = L"\text{not converged } (\|\Delta S\|_\infty > %$(tol_tex))",
                align = (:left, :top), color = :gray, fontsize = 18)
        end

        # All runs share one T, fixed by the DMFT reference; the NL method is fixed for
        # the whole figure, so both go in the title. T/t is read off the reference rather
        # than written in as the literal 2 it happens to be for the inputs on disk. Keep
        # the whole LaTeXString on ONE source line: a line break inside L"..." would inject
        # the continuation's indentation as literal whitespace and open a gap in the title.
        # The title is split over two lines: on one it is wider than a single panel, and a
        # Label that is wider than the figure stretches the layout past the figure edges,
        # cropping the y-axis on the left and the last panel on the right. tellwidth = false
        # keeps an over-long title from ever doing that again (it would overhang instead).
        # The hyphen in the NL name is U+2010: MathTeXEngine sets an ASCII "-" as a minus
        # even inside \text, which rendered "K23 s-wave" as "K23 s − wave".
        Tt     = d.T === nothing ? nothing : round(d.T / t, digits = 4)
        nl_tex = replace(nl_name, "-" => "‐")
        d.common_tag === nothing || (nl_tex = "$(d.common_tag), $(nl_tex)")
        title  = fig[0, :] = GridLayout()
        Label(title[1, 1], L"\text{Parquet residual vs. system size}"; fontsize = 26, tellwidth = false)
        Label(title[2, 1], Tt === nothing ? L"\text{PHS, %$(nl_tex)}" : L"$T/t=%$(Tt)$, $\text{PHS, %$(nl_tex)}$";
            fontsize = 22, tellwidth = false)
        rowgap!(title, 4)

        # The first panel carries the full set of U labels, so the shared legend is built
        # from its curves. The labels are bare numbers and need the title to be readable.
        nbanks = length(leg_plots) > 4 ? 2 : 1
        order  = rowfirst(collect(eachindex(leg_plots)), nbanks)
        Legend(fig[1, :], leg_plots[order], leg_labels[order], L"U/t";
            orientation   = :horizontal,
            framevisible  = false,
            titleposition = :left,
            tellheight    = true,
            tellwidth     = false,
            nbanks        = nbanks,
        )

        figdir = figures_dir()
        mkpath(figdir)
        ztag = join(["$(s[1])-$(s[2])" for s in sizes], "_")
        utag = join([replace(u, "." => "p") for u in Us], "-")
        out  = joinpath(figdir,
            "residual_vs_size_NL$(NL_method)_$(join(algos, "-"))_$(ztag)_U$(utag).svg")
        save(out, fig)
        println("Saved figure to\n  $out")

        display(fig)
        return fig
    end

    return figure
end

# ---------------------------------------------------------------------------
# Executing this file does the whole thing. The two halves are also callable on
# their own, so a layout tweak need not re-scan the metadata:
#
#   julia> data = load_data()                # once
#   julia> plot_data(data; tol = 1e-6)       # as often as you like
# ---------------------------------------------------------------------------
data = load_data()
fig  = plot_data(data)
