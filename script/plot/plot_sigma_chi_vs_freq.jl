"""
Plot Re Σ, Im Σ and χ_D, χ_M as a function of Matsubara frequency at a *fixed*
momentum k — the frequency-resolved counterpart of `plot_sigma_chi_vs_q.jl` (which
instead fixes the frequency and sweeps k along Γ-X-M-Γ).

The self-energy panels are drawn versus the fermionic frequency iν, the
susceptibility panels versus the bosonic frequency iω, both over the positive half
of the Matsubara range stored in the precomputed caches. The data come from the
full-frequency-grid datasets written by `precompute_sigma_chi.jl`:

  output/postprocess/hubbard_<algo>_..._NL<NL>.h5
      sigma/nu,  sigma/{raw,tilde}/{re,im}   (nν × nk²)
      chi/omega, chi/{raw,tilde}/{D,M}        (nΩ × nk²)

This script does NOT load the (large) converged solvers. If a cache file is
missing, `precompute` is called to generate it; that is
the only step that touches the solver. A curve is drawn only if its data can be produced.

Multiple NL methods can be drawn together: colour encodes the strategy
(pDΓA / PA), line style the NL method (NL1 solid, NL2 dash, NL3 dot).

In raw mode the local DMFT reference is drawn as its own (gray, dash-dotted) curve;
in tilde mode it is the zero line by construction and is omitted.

Unlike `plot_sigma_chi_vs_q.jl`, no detQMC benchmark is overlaid: the reference
files are k-resolved at a single Matsubara frequency and carry no frequency
dependence to compare against.

Executing this file plots. Run as a script, the options come from the command line:

  julia --project=script script/plot/plot_sigma_chi_vs_freq.jl [options]

Run with `--help` for the full option list and the current defaults.

Executed in the REPL, they come instead from the parameter block just below the
imports. Edit it and execute the file again (VSCode: *Julia: Execute File in REPL*,
or `include("script/plot/plot_sigma_chi_vs_freq.jl")`); a warm session redraws in
about a second, since only the drawing is repeated and not the package loading.

That block is also where the command line takes its defaults from, so the two
entry points cannot drift apart.

  --U            interaction, as spelled in the DMFT file name
                 (T is fixed by U via that reference)
  --nmax --nq    vertex box half-size and coarse k-grid size
  --nls          comma-separated NL methods to draw: 1 | 2 | 3
  --strategies   comma-separated scPA | fdPA
  --schemes      comma-separated anderson | mfRG
  --qx --qy      fixed momentum in units of π (1,1 → M = (π, π))
  --tilde        plot X̃ = X − X_DMFT instead of the raw X
  --wmax         max frequency shown (Inf → whole stored positive range)

  julia --project=script script/plot/plot_sigma_chi_vs_freq.jl --U=2.089 --nmax=12 --nq=6
  julia --project=script script/plot/plot_sigma_chi_vs_freq.jl --U=0.5 --nls=1 --qx=1 --qy=1 --tilde
"""

using HDF5
using CairoMakie

include(joinpath(@__DIR__, "..", "postprocess", "precompute_sigma_chi.jl"))   # caches + precompute + dmft_temperature

# ---------------------------------------------------------------------------
# What executing this file in the REPL plots. Edit these and execute it again.
#
# Running the file as a script overrides them from the command line — and these
# are the defaults `--help` reports, so the two entry points cannot drift apart.
# ---------------------------------------------------------------------------
U          = "0.5"              # as spelled in the DMFT file name
nmax       = 10                 # vertex box half-size
nq         = 4                  # coarse k-grid size
nls        = "1"                # NL methods to draw: "1" | "1,2" | "1,2,3"
strategies = "fdPA"             # "fdPA" | "scPA" | both
schemes    = "mfRG"             # "anderson" | "mfRG" | both
qx_pi      = 0.5                # fixed momentum in units of π
qy_pi      = 0.5                #   (1, 1 -> M = (π, π))
wmax       = 30                 # frequency window; Inf -> whole stored range
tilde      = false              # plot X - X_DMFT instead of the raw X

# ---------------------------------------------------------------------------
# CLI driver (skipped when this file is executed in the REPL or `include`d)
# ---------------------------------------------------------------------------
if abspath(PROGRAM_FILE) == @__FILE__
    settings = cli_settings(
        "Plot Σ and χ versus Matsubara frequency at a fixed momentum.")
    add_arg_table!(settings,
        ["--U"],  Dict(:help => "interaction strength, as spelled in the DMFT file name",
                       :arg_type => String,  :default => U),
        ["--nls"], Dict(:help => "comma-separated NL methods to draw: 1 | 2 | 3",
                       :arg_type => String,  :default => nls),
        ["--strategies"], Dict(:help => "comma-separated scPA | fdPA",
                               :arg_type => String, :default => strategies),
        ["--schemes"],    Dict(:help => "comma-separated anderson | mfRG",
                               :arg_type => String, :default => schemes),
        ["--qx"],   Dict(:help => "q_x in units of π", :arg_type => Float64, :default => qx_pi),
        ["--qy"],   Dict(:help => "q_y in units of π", :arg_type => Float64, :default => qy_pi),
        ["--wmax"], Dict(:help => "frequency window", :arg_type => Float64, :default => wmax))
    # NL_method = nothing: the momentum structures drawn come from `--nls`, so the
    # scalar `--NL` would only be a second option one capital apart that did nothing.
    add_box_args!(settings; nmax, nq, NL_method = nothing)
    add_flags!(settings,
        "tilde" => "plot the difference to DMFT (X̃ = X − X_DMFT) instead of the raw X")

    parsed = parse_args(ARGS, settings)

    U          = parsed["U"]
    nmax       = parsed["nmax"]
    nq         = parsed["nq"]
    nls        = parsed["nls"]
    strategies = parsed["strategies"]
    schemes    = parsed["schemes"]
    qx_pi      = parsed["qx"]
    qy_pi      = parsed["qy"]
    wmax       = parsed["wmax"]
    tilde      = parsed["tilde"]
end

# Curve styling: colour encodes the strategy, line style the NL method (or, with a
# single NL method selected, the strategy again: PA dashed, pDΓA solid).
strat_color = Dict(:fdPA => Makie.wong_colors()[1], :scPA => Makie.wong_colors()[2])
nl_style    = Dict(1 => :solid, 2 => :dash, 3 => :dot)
nl_name     = Dict(1 => "K23 s-wave", 2 => "K3 s-wave", 3 => "full q")

"""
    load_data(; U, nmax, nq, nls, strategies, schemes, tilde) -> NamedTuple

Read the caches and assemble every curve the figure draws. Each keyword defaults to
the parameter block above, so `load_data()` loads exactly what executing this file
would. The momentum is deliberately NOT one of them: it only indexes the dense
k-grid, so `plot_data` can move it without this running again.
"""
function load_data(; U          = U,
                     nmax       = nmax,
                     nq         = nq,
                     nls        = nls,
                     strategies = strategies,
                     schemes    = schemes,
                     tilde      = tilde)

    U = string(U)   # `U = 2.089` is natural to type; the string is what names the files

    nls_sel     = parse_nls(nls)
    strats_sel  = parse_strategies(strategies)
    schemes_sel = parse_schemes(schemes)

    mode = tilde ? "tilde" : "raw"

    # U stays the string it is spelled with — that is what names the cache files, and
    # a U written `2.0875` must not come back as `2.0875000001`. `U_num` is the same
    # value for the arithmetic (axis labels, χ ~ 1/U² inside `precompute`).
    U_num = parse(Float64, U)

    # T is fixed by U via the DMFT reference; used so the cache filenames match.
    T = dmft_temperature(U)

    # The unit the U and T axes are quoted in, read off the same reference rather
    # than written into the title as a bare factor of 4.
    t = dmft_hopping(U)

    multi_nl     = length(nls_sel)     > 1
    multi_scheme = length(schemes_sel) > 1

    # The group read follows the mode: `tilde` holds the difference to the DMFT
    # reference, `raw` the full-physical quantity.
    tag = tilde ? "tilde" : "raw"

    # Load the full-frequency-grid Σ / χ for one run on one NL method, generating from
    # the saved solver on first use. Returns `nothing` (with a warning) if unavailable.
    # `file` is kept so the DMFT curve below can read the other mode from it.
    function load_all(strategy, scheme, NL_method)
        run_tag = "$(ALGO_FAMILY[strategy])_$(scheme)"
        file    = postprocess_path(U, T, nmax, nq, NL_method, strategy, scheme)

        if !isfile(file)
            try
                # U goes over as the string it names the files with; `precompute`
                # splits off the number it needs for the arithmetic (χ ~ 1/U²).
                precompute(U, nmax, nq, NL_method, strategy, scheme)
            catch err
                @warn "Could not obtain $(run_tag)/NL$(NL_method) data; skipping it." exception = err
                return nothing
            end
            isfile(file) || return nothing
        else
            println("Reusing cached $(run_tag)/NL$(NL_method) data:\n  $file")
        end

        return h5open(file, "r") do f
            (; file,
               nu    = read(f, "sigma/nu"),
               ReΣ   = read(f, "sigma/$(tag)/re"),
               ImΣ   = read(f, "sigma/$(tag)/im"),
               omega = read(f, "chi/omega"),
               χD    = read(f, "chi/$(tag)/D"),
               χM    = read(f, "chi/$(tag)/M"),
               nk    = read(f, "params/nk"))
        end
    end

    # One curve per requested (strategy, scheme, NL) triple — the same vocabulary
    # run_hubbard_model.jl writes a run with; drop those without data.
    curves = NamedTuple[]   # each: (; strategy, scheme, nl, data, label, color, linestyle)
    for strategy in strats_sel, scheme in schemes_sel, nl in nls_sel
        d = load_all(strategy, scheme, nl)
        d === nothing && continue
        label = strategy_label(strategy)
        multi_nl     && (label *= " " * nl_name[nl])
        multi_scheme && (label *= " ($scheme)")
        push!(curves, (; strategy, scheme, nl, data = d, label,
            color     = strat_color[strategy],
            linestyle = multi_nl ? nl_style[nl] : (strategy === :scPA ? :dash : :solid)))
    end
    isempty(curves) && error("No data for the requested selection.")

    # All curves share the dense k-grid; take its size from the first.
    nk = curves[1].data.nk

    # -----------------------------------------------------------------------
    # DMFT reference, drawn as its own curve in raw mode. Unlike in
    # plot_sigma_chi_vs_q.jl it is not a constant here: it is local (momentum
    # independent) but carries the full frequency dependence. The raw and tilde groups
    # of a cache come from the same solver run and differ by exactly that reference
    # (tilde = raw − DMFT, the same k-independent shift at every k), so it is recovered
    # as raw − tilde without touching the solver. In tilde mode it is the zero line by
    # construction and is not drawn.
    # -----------------------------------------------------------------------
    dmft_curve = nothing
    if !tilde
        c = curves[1]
        dmft_curve = h5open(c.data.file, "r") do f
            (; data = (nu    = c.data.nu,
                       omega = c.data.omega,
                       ReΣ   = c.data.ReΣ .- read(f, "sigma/tilde/re"),
                       ImΣ   = c.data.ImΣ .- read(f, "sigma/tilde/im"),
                       χD    = c.data.χD  .- read(f, "chi/tilde/D"),
                       χM    = c.data.χM  .- read(f, "chi/tilde/M")),
               label     = "DMFT",
               color     = :gray,
               linestyle = :dashdot)
        end
    end

    return (; curves, dmft_curve, nk, U, U_num, T, nmax, nq, tilde, mode,
              t, nls_sel, strats_sel, schemes_sel, multi_nl, multi_scheme)
end

"""
    plot_data(d; qx_pi, qy_pi, wmax) -> Figure

Draw and save the figure from `load_data`'s result. Nothing here re-reads a cache,
so retuning the layout costs the drawing alone:

    julia> data = load_data()
    julia> plot_data(data; qx_pi = 1, qy_pi = 1)
"""
function plot_data(d; qx_pi = qx_pi, qy_pi = qy_pi, wmax = wmax)

    curves, dmft_curve     = d.curves, d.dmft_curve
    U, U_num, T, nmax, nq  = d.U, d.U_num, d.T, d.nmax, d.nq
    t                      = d.t
    tilde, mode            = d.tilde, d.mode
    nls_sel, strats_sel    = d.nls_sel, d.strats_sel
    multi_nl, multi_scheme = d.multi_nl, d.multi_scheme

    # Fixed-momentum linear index (shared: all curves use the same dense k-grid).
    ik   = k_index(qx_pi, qy_pi, d.nk)
    qtex = q_tex(qx_pi, qy_pi)

    println("\nPlotting Figure (fixed k = $(q_tex(qx_pi, qy_pi)), linear index $ik):")
    # font, line and marker sizes tuned for projection on a presentation slide
    figure = with_theme(theme_latexfonts();
        fontsize = 22,
        Axis   = (titlesize = 26, xlabelsize = 26, ylabelsize = 26,
                  xticklabelsize = 22, yticklabelsize = 22),
        Legend = (labelsize = 20,),
        Lines        = (linewidth = 3,),
        ScatterLines = (linewidth = 3, markersize = 12),
        VLines       = (linewidth = 2,),
    ) do
        fig = Figure(size = (1300, 1000))

        # Attributes shared by all four panels (sizes come from the theme above).
        axis_kw = (
            xticksmirrored = true,
            xtickalign     = 1.0,
            yticksmirrored = true,
            ytickalign     = 1.0,
        )

        # One panel: axis + every selected curve for value field `fld` versus its
        # frequency axis `xfld` (:nu for Σ, :omega for χ), at the fixed momentum `ik`.
        # Only the positive half of each Matsubara axis is shown; the caches hold the full
        # ± range, and the negative half carries no independent information here. `neg` keeps
        # the first negative frequency as well: the fermionic mesh has no point at ν = 0, so
        # without it the Σ curves would start a gap away from the frame (the axis limit stays
        # at 0 either way, clipping the extra segment).
        function freq_mask(x; neg = false)
            negs = filter(<(0), x)
            lo   = (neg && !isempty(negs)) ? maximum(negs) : 0.0
            return (lo .<= x) .& (x .<= wmax)
        end

        # Values of one curve in the shown frequency window, at the fixed momentum `ik`.
        function window(c, fld, xfld; neg = false)
            x = getproperty(c.data, xfld)
            m = freq_mask(x; neg)
            return x[m], getproperty(c.data, fld)[m, ik]
        end

        # The y values of one curve that actually fall inside the frame (ν ≥ 0), including
        # the point where it crosses the left edge. The extra negative frequency lies outside
        # the frame, and scaling the axis to it would leave half the panel empty (Im Σ is odd
        # in ν, so its first negative value is the mirror of the first positive one).
        function visible(x, y)
            keep = x .>= 0
            vals = collect(y[keep])
            i    = findfirst(keep)
            if i !== nothing && i > 1   # a negative point is present: interpolate onto ν = 0
                x0, x1, y0, y1 = x[i-1], x[i], y[i-1], y[i]
                push!(vals, y0 + (y1 - y0) * (0 - x0) / (x1 - x0))
            end
            return vals
        end

        function panel!(row, col, fld, xfld; title, xlabel, ylabel = "", neg = false)
            # y data is rescaled by 10^mag so the tick labels read as small numbers; the
            # factor is called out above the axis instead of on every tick. The exponent and
            # the y limits come from what is visible, DMFT curve included.
            shown = [window(c, fld, xfld; neg) for c in curves]
            dmft_curve === nothing || push!(shown, window(dmft_curve, fld, xfld; neg))
            vis   = [visible(x, y) for (x, y) in shown]
            vmax  = maximum(maximum(abs, v) for v in vis)
            mag   = vmax > 0 ? floor(Int, log10(vmax)) : 0
            scale = 10.0^(-mag)

            ax = Axis(fig[row, col]; axis_kw..., title = title, xlabel = xlabel, ylabel = ylabel)
            # Above the frame rather than in the TopLeft() corner: the corner is where the
            # topmost y tick label sits, and the two collide once that tick is near the top.
            Label(fig[row, col, Top()], L"\times 10^{%$mag}";
                halign = :left, padding = (0, 0, 4, 0))

            # line widths and marker sizes come from the theme above
            for (c, (x, y)) in zip(curves, shown)
                scatterlines!(ax, x, y .* scale; label = c.label,
                    color = c.color, linestyle = c.linestyle)
            end
            # The DMFT reference (raw mode only), drawn last so it stays on top of the solver
            # curves; markerless, to set it apart from them.
            if dmft_curve !== nothing
                x, y = shown[end]
                lines!(ax, x, y .* scale; label = dmft_curve.label,
                    color = dmft_curve.color, linestyle = dmft_curve.linestyle)
            end
            xlims!(ax, 0, isfinite(wmax) ? wmax : nothing)
            ylo, yhi = extrema(scale .* reduce(vcat, vis))
            pad      = 0.05 * max(yhi - ylo, eps())
            ylims!(ax, ylo - pad, yhi + pad)
            return ax
        end

        # Titles / y-labels adapt to raw vs tilde.  Build the LaTeX symbol fragments as
        # plain strings first — nesting a "…" literal inside an L"…" interpolation would
        # close the L-string early.
        Σsym = tilde ? "\\tilde{\\Sigma}" : "\\Sigma"
        χsym = tilde ? "\\tilde{\\chi}"   : "\\chi"
        xΣ = L"\nu"
        xχ = L"\omega"

        # The fermionic mesh has no ν = 0, so the Σ panels also draw the first negative
        # frequency (neg = true) to reach the frame edge.
        ax1 = panel!(2, 1, :ReΣ, :nu;    title = L"\text{Re } %$(Σsym)", xlabel = xΣ,
            ylabel = L"%$(Σsym)(i\nu, %$(qtex))", neg = true)
        ax2 = panel!(2, 2, :ImΣ, :nu;    title = L"\text{Im } %$(Σsym)", xlabel = xΣ,
            neg = true)
        ax3 = panel!(3, 1, :χD,  :omega; title = L"%$(χsym)_D", xlabel = xχ,
            ylabel = L"%$(χsym)(i\omega, %$(qtex))")
        ax4 = panel!(3, 2, :χM,  :omega; title = L"%$(χsym)_M", xlabel = xχ)

        # Title (row 0) and a single shared legend (row 1) above all plots. The momentum is
        # named in the y-labels instead of here.
        nl_title = join((nl_name[nl] for nl in nls_sel), ", ")
        Tt, Ut = round(T / t, digits = 4), round(U_num / t, digits = 4)
        Label(fig[0, :], L"$T/t=%$(Tt)$, $U/t=%$(Ut)$, $\text{PHS}$"; fontsize = 26)

        # ax1 carries the full set of series labels, so the shared legend is built from it.
        Legend(fig[1, :], ax1;
            orientation  = :horizontal,
            framevisible = false,
            tellheight   = true,
            tellwidth    = false,
        )

        figdir = figures_dir()
        mkpath(figdir)
        stag  = join((ALGO_FAMILY[st] for st in strats_sel), "-")   # e.g. "fd-sc"
        nltag = "NL" * join(nls_sel, "")   # e.g. "NL12"
        ktag  = "q$(qx_pi)_$(qy_pi)"       # e.g. "q1.0_1.0"
        out   = joinpath(figdir,
            "hubbard_vsfreq_U$(U)_T$(T)_nmax$(nmax)_nq$(nq)_$(nltag)_$(stag)_$(ktag)_$(mode).svg")
        save(out, fig)
        println("Saved figure to\n  $out")

        return fig
    end


    return figure
end

# ---------------------------------------------------------------------------
# Executing this file does the whole thing. The two halves are also callable on
# their own, so a layout tweak need not re-read the caches:
#
#   julia> data = load_data()                       # once
#   julia> plot_data(data; qx_pi = 1, qy_pi = 1)    # as often as you like
# ---------------------------------------------------------------------------
data = load_data()
fig  = plot_data(data)