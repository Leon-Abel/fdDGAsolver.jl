"""
Plot Re Σ, Im Σ (at ν = iπT) and χ_D, χ_M (at ω = 0) along the Γ-X-M-Γ
high-symmetry path — either the raw (full physical) quantities or, in tilde mode,
their difference to the DMFT reference (X̃ = X − X_DMFT).

A run is addressed by the same two knobs `run_hubbard_model.jl` writes it with:
`strategy` (:scPA | :fdPA — which reference vertex the solver uses) and
`iteration_scheme` (:anderson | :mfRG — how the fixed point was reached). Both
schemes converge the same fixed point, so drawing them together is a convergence
cross-check: a visible gap between an mfRG curve and its Anderson counterpart is
an artefact rather than physics.

Both variants are read from the precomputed caches written by
`precompute_sigma_chi.jl` (tilde mode reads their `tilde` groups):

  output/postprocess/hubbard_<fam>_<scheme>_U<U>_T<T>_nmax<nmax>_nq<nq>_NL<NL>.h5

with <fam> the solver family (ALGO_FAMILY[strategy], "sc" or "fd") and <scheme>
the iteration scheme — e.g. fd_mfRG. This script does NOT load the (large)
converged solvers. If a cache file is missing, `precompute` is called to generate
it — that is the only step that touches the solver. A curve is drawn only if its
data can be produced, so selecting a run that was never performed is harmless.

Several runs can be drawn together. Color and line style both encode the iteration
scheme — :anderson blue and dashed, :mfRG orange and solid — with the strategy
shading the hue (pDΓA/PA as variants of the same color) and the NL method varying
the dash pattern within its scheme's family (NL1 plain, NL2/NL3 finer).

Usage (from the repo root):
  julia --project=script script/plot/plot_sigma_chi_vs_q.jl [options]

Run with `--help` for the full option list and the current defaults. Every option
is named, so order never matters and only what you change needs to be passed.

  --U            interaction, as spelled in the DMFT file name. T is fixed by U
                 through that reference. The detQMC benchmark overlays only at
                 U = 2.089.
  --nmax --nq    vertex box half-size and coarse k-grid size
  --nls          comma-separated NL methods to draw: 1 | 2 | 3
  --strategies   comma-separated scPA | fdPA
  --schemes      comma-separated anderson | mfRG; both (the default) plots the
                 two routes side by side
  --tilde        plot X̃ = X − X_DMFT instead of the raw X. The benchmark overlay
                 follows the same convention.

  julia --project=script script/plot/plot_sigma_chi_vs_q.jl --U=2.089 --nmax=12 --nq=6
  julia --project=script script/plot/plot_sigma_chi_vs_q.jl --U=0.5 --nls=1,2 --schemes=anderson --tilde
"""

using HDF5
using CairoMakie
using DelimitedFiles

include(joinpath(@__DIR__, "..", "postprocess", "precompute_sigma_chi.jl"))   # caches + precompute + dmft_temperature

# ---------------------------------------------------------------------------
# What executing this file in the REPL does. Edit these and execute it again.
#
# Running the file as a script overrides them from the command line - and these
# are the defaults `--help` reports, so the two entry points cannot drift apart.
# ---------------------------------------------------------------------------
U          = "1.0"            # interaction strength, as spelled in the DMFT file name
nls        = "1"              # comma-separated NL methods to draw: 1 | 2 | 3
strategies = "fdPA"           # comma-separated scPA | fdPA
schemes    = "mfRG,anderson"  # comma-separated anderson | mfRG
nmax       = 10               # half-box size for the vertex grids
nq         = 4                # linear k-grid size for the vertex mesh
tilde      = false            # plot X̃ = X − X_DMFT (the difference to DMFT) instead of ...

# ---------------------------------------------------------------------------
# CLI driver (skipped when this file is executed in the REPL or `include`d)
# ---------------------------------------------------------------------------
if abspath(PROGRAM_FILE) == @__FILE__
    settings = cli_settings(
        "Plot Σ and χ along Γ-X-M-Γ at fixed frequency, for one interaction strength.")
    add_arg_table!(settings,
        ["--U"],  Dict(:help => "interaction strength, as spelled in the DMFT file name",
                       :arg_type => String, :default => U),
        ["--nls"], Dict(:help => "comma-separated NL methods to draw: 1 | 2 | 3",
                       :arg_type => String, :default => nls),
        ["--strategies"], Dict(:help => "comma-separated scPA | fdPA",
                               :arg_type => String, :default => strategies),
        ["--schemes"],    Dict(:help => "comma-separated anderson | mfRG",
                               :arg_type => String, :default => schemes))
    # NL_method = nothing: the momentum structures drawn come from `--nls`, so the
    # scalar `--NL` would only be a second option one capital apart that did nothing.
    add_box_args!(settings; nmax, nq, NL_method = nothing)
    add_flags!(settings,
        "tilde" => "plot X̃ = X − X_DMFT (the difference to DMFT) instead of the raw X")

    parsed = parse_args(ARGS, settings)

    U          = parsed["U"]
    nls        = parsed["nls"]
    strategies = parsed["strategies"]
    schemes    = parsed["schemes"]
    nmax       = parsed["nmax"]
    nq         = parsed["nq"]
    tilde      = parsed["tilde"]
end

# Curve styling. The iteration scheme picks the hue family — Anderson blue, mfRG
# orange — so the two routes are never mistaken for one series where they nearly
# coincide, which, being the same fixed point, is most of the path. Within a family
# the strategy picks the shade, so pDΓA and PA read as variants of the same color.
# Line style reinforces the scheme: :anderson dashed, :mfRG solid.
wong        = Makie.wong_colors()
algo_color  = Dict((:fdPA, :anderson) => wong[1], (:scPA, :anderson) => wong[5],   # blues
                   (:fdPA, :mfRG)     => wong[2], (:scPA, :mfRG)     => wong[6])   # oranges
algo_name   = Dict(:fdPA => "pDΓA",     :scPA => "PA")
scheme_name = Dict(:anderson => "Anderson", :mfRG => "mfRG")
nl_name     = Dict(1 => "K23 s-wave", 2 => "K3 s-wave", 3 => "full q")

# The NL method varies the pattern *within* its scheme's family, so "Anderson is
# the dashed one" stays true even with several NL methods on screen.
nl_style    = Dict(
    :anderson => Dict(1 => :dash,  2 => :dashdot, 3 => :dashdotdot),
    :mfRG     => Dict(1 => :solid, 2 => :dot,     3 => (:dot, :dense)))

# -----------------------------------------------------------------------
# Γ-X-M-Γ path over the dense k-grid
# -----------------------------------------------------------------------
# The dense BZ runs 0..2π with nk points per axis; the linear k index is
#   idx = (iy-1)*nk + ix          (kx is the fast index),
# idx=1 is Γ=(0,0), and k=π sits at index h+1 = nk/2+1.
# Returns the linear k indices of the connected path and their arc length s, as
# two parallel vectors: the indices slice a stored k-map, s is the abscissa.
function gamma_x_m_path(nk)
    h = nk ÷ 2                            # k = π at index h+1
    k_idx(ix, iy) = (iy - 1) * nk + ix    # linear k index
    #    Γ→X : (kx, 0), kx = 0..π       X→M : (π, ky), ky = 0..π
    #    M→Γ : (k, k),  k  = π..0  (length π√2)
    idx = [[k_idx(ix, 1)   for ix in 1:h+1];
           [k_idx(h+1, iy) for iy in 2:h+1];
           [k_idx(i, i)    for i  in h:-1:1]]
    s   = [[(ix-1)/h * π              for ix in 1:h+1];
           [π + (iy-1)/h * π          for iy in 2:h+1];
           [2π + (π - (i-1)/h*π) * √2 for i  in h:-1:1]]
    return idx, s
end

# Cumulative Euclidean arc length along the k-path (reproduces gamma_x_m_path).
path_arclength(kx, ky) = [0.0; cumsum(hypot.(diff(kx), diff(ky)))]

"""
    load_data(; U, nls, strategies, schemes, nmax, nq, tilde) -> NamedTuple

Read the caches (generating any that are missing), the DMFT reference and the
detQMC benchmark, and assemble every curve the figure draws. Each keyword defaults
to the parameter block above, so `load_data()` loads exactly what executing this
file would.
"""
function load_data(; U          = U,
                     nls        = nls,
                     strategies = strategies,
                     schemes    = schemes,
                     nmax       = nmax,
                     nq         = nq,
                     tilde      = tilde)

    U = string(U)   # `U = 2.089` is natural to type; the string is what names the files

    nls_sel     = parse_nls(nls)
    strats_sel  = parse_strategies(strategies)
    # mfRG first by default, so the Anderson curves end up on top where the two nearly
    # coincide — which, being the same fixed point, is most of the path.
    schemes_sel = parse_schemes(schemes)

    # U stays the string it is spelled with — that is what names the cache files.
    # `U_num` is the same value for the arithmetic (axis labels, χ ~ 1/U²).
    U_num = parse(Float64, U)

    # T is fixed by U via the DMFT reference; used so the cache filenames match.
    T = dmft_temperature(U)

    # The unit the U and T axes are quoted in, read off the same reference rather
    # than written here as a bare factor of 4.
    t = dmft_hopping(U)

    multi_scheme = length(schemes_sel) > 1
    multi_nl     = length(nls_sel)     > 1

    # -----------------------------------------------------------------------
    # External detQMC benchmark along Γ-X-M-Γ (only available for U = 2.089)
    # -----------------------------------------------------------------------
    # Three whitespace-separated files, all sampled on the Γ-X-M-Γ path:
    #   SlfEngkIwnPathGXMG_L16.txt  [path, kx, ky, ReΣ, err, ImΣ, err]  (self-energy)
    #   DenDenkIwnPathGXMG_L16.txt  [path, kx, ky, χ_D, err]            (density χ)
    #   SpinZZkIwnPathGXMG_L16.txt  [path, kx, ky, χ_M, err]            (spin χ)
    # The files hold the *raw* (full physical) quantities. In raw mode they are used as
    # they are; in tilde mode each benchmark is brought into the panels' frame (X − DMFT)
    # by subtracting the (momentum-independent) DMFT reference at ν = iπT / ω = 0. Re Σ is
    # additionally negated (`ref_sign`) to match the solver's sign convention; error
    # bars are invariant under the constant shift and the sign flip. The abscissa (arc
    # length) is rebuilt from (kx, ky) so it lands on the same Γ-X-M-Γ axis as the
    # solver curves. NOTE: these are fixed-U datasets (match the default U = 2.089);
    # they are not re-derived for other U, so only overlay them at that U.

    # -----------------------------------------------------------------------
    # DMFT reference at ν = iπT / ω = 0 (momentum independent). Conventions follow
    # precompute_sigma_chi.jl: ReΣ_DMFT = imag Σ (≡ 0 at PHS), ImΣ_DMFT = −real Σ, and the
    # local density / magnetic susceptibilities. It is needed to draw the DMFT line in the
    # raw panels and to shift the detQMC benchmark into the tilde frame, so it is loaded
    # only when one of the two applies.
    # -----------------------------------------------------------------------
    # `dmft_reference` takes U as spelled: the string picks the file, the number it
    # parses out of it divides into χ.
    dref = (!tilde || U_num == 2.089) ? dmft_reference(U) : nothing

    # Keyed by the same field symbols `panel!` uses.
    dmft_val = dref === nothing ? Dict{Symbol, Float64}() : Dict(
        :ReΣ => imag(dref.Σ_DMFT_ν0),
        :ImΣ => -real(dref.Σ_DMFT_ν0),
        :χD  => dref.chi_D_DMFT,
        :χM  => dref.chi_M_DMFT)

    # The DMFT line is drawn only in the raw panels — in tilde mode it is the zero line
    # by construction (X̃_DMFT = X_DMFT − X_DMFT = 0) and carries no information.
    dmft = tilde ? Dict{Symbol, Float64}() : dmft_val

    # -----------------------------------------------------------------------
    refs = Dict{Symbol, Any}()   # per-panel benchmark overlays (empty unless U = 2.089)
    if U_num == 2.089
        ref_dir  = data_dir()
        ref_sign = -1   # benchmark Re Σ is sign-flipped vs the solver; set +1 to keep as-is

        # Read a value column `vcol` (with error column `ecol`) from a Γ-X-M-Γ path-file in
        # data/, subtract the DMFT reference `sub` (zero in raw mode), optionally flip its
        # sign, and attach the arc length. Returns `nothing` (with a warning) if the file is
        # absent.
        function load_path_ref(fname, vcol, ecol; sub = 0.0, sign = 1)
            path = joinpath(ref_dir, fname)
            isfile(path) || (@warn "No benchmark reference at\n  $path"; return nothing)
            d = readdlm(path)
            return (s   = path_arclength(Float64.(d[:, 2]), Float64.(d[:, 3])),
                    val = sign .* Float64.(d[:, vcol]) .- sub,
                    err = Float64.(d[:, ecol]))
        end

        # Per-panel benchmark overlays, keyed by the same field symbols `panel!` uses. In
        # tilde mode the DMFT reference is subtracted; in raw mode nothing is, so the
        # benchmark stays in the full-physical frame.
        for (key, fname, vcol, ecol, sgn) in (
                (:ReΣ, "SlfEngkIwnPathGXMG_L16.txt", 4, 5, ref_sign),
                (:ImΣ, "SlfEngkIwnPathGXMG_L16.txt", 6, 7, 1),
                (:χD,  "DenDenkIwnPathGXMG_L16.txt", 4, 5, 1),
                (:χM,  "SpinZZkIwnPathGXMG_L16.txt", 4, 5, 1),
            )
            r = load_path_ref(fname, vcol, ecol;
                sub = tilde ? dmft_val[key] : 0.0, sign = sgn)
            r === nothing || (refs[key] = r)
        end
    end

    # -----------------------------------------------------------------------
    # Load the precomputed quantities (compute-on-first-use)
    # -----------------------------------------------------------------------
    println("Loading precomputed $(tilde ? "tilde (X − DMFT)" : "raw") data:")

    # `tilde` picks the group read from the cache; `precompute` always writes both, so
    # a missing cache is regenerated the same way in either mode.
    tag = tilde ? "tilde" : "raw"

    # The caches store every frequency of the solver grid; these panels want one row of
    # each. `nu0_row` / `omega0_row` live in common/plotting.jl rather than in
    # precompute_sigma_chi.jl, which stores the axes and leaves their interpretation
    # to the consumer.

    # Load the cached Σ / χ for one run on one NL method, generating them from the
    # saved solver on first use. Returns `nothing` (with a warning) if unavailable —
    # a run that was never performed simply drops out of the figure.
    function load_curve(strategy, iteration_scheme, NL_method)
        label = "$(ALGO_FAMILY[strategy])_$(iteration_scheme)/NL$(NL_method)"
        file  = postprocess_path(U, T, nmax, nq, NL_method, strategy, iteration_scheme)

        if isfile(file)
            println("Reusing cached $(label) data:\n  $file")
        else
            # Check the solver file first: `precompute` reads the (large) DMFT payload
            # before it ever looks for the solver, so an unperformed run would cost that
            # load only to fail. This keeps "select everything, plot what exists" cheap.
            solver = solver_path(U, T, nmax, nq, NL_method, strategy, iteration_scheme)
            if !isfile(solver)
                @warn "No cache and no saved solver for $(label); skipping it.\n  $solver"
                return nothing
            end
            try
                precompute(U, nmax, nq, NL_method, strategy, iteration_scheme)
            catch err
                @warn "Could not obtain $(label) data; skipping it." exception = err
                return nothing
            end
            isfile(file) ||
                (@warn "precompute wrote no cache for $(label); skipping it."; return nothing)
        end

        # One open, and `dset[i, :]` is a hyperslab read: only the ν = iπT / Ω = 0 row
        # crosses into memory, not the whole (nν × nk²) matrix behind it.
        return h5open(file, "r") do f
            iν = nu0_row(read(f, "sigma/nu"), T)
            iΩ = omega0_row(read(f, "chi/omega"))
            (ReΣ = f["sigma/$(tag)/re"][iν, :], ImΣ = f["sigma/$(tag)/im"][iν, :],
             χD  = f["chi/$(tag)/D"][iΩ, :],    χM  = f["chi/$(tag)/M"][iΩ, :],
             nk  = read(f, "params/nk"))
        end
    end

    # One curve per requested (strategy, scheme, NL) triple; drop those without data.
    curves = NamedTuple[]   # each: (; data, label, color, linestyle)
    for strategy in strats_sel, scheme in schemes_sel, nl in nls_sel
        d = load_curve(strategy, scheme, nl)
        d === nothing && continue
        # The scheme only needs spelling out when both are on screen; likewise the NL
        # method, which keeps a single-selection legend short.
        label = algo_name[strategy] *
            (multi_scheme ? " $(scheme_name[scheme])" : "") *
            (multi_nl     ? " $(nl_name[nl])"         : "")
        push!(curves, (; data = d, label,
            color     = algo_color[(strategy, scheme)],
            linestyle = nl_style[scheme][nl]))
    end
    isempty(curves) && error("No data for the requested selection.")

    # All curves share the dense k-grid; take it from the first available curve.
    nk = curves[1].data.nk

    return (; curves, refs, dmft, nk, U, U_num, T, t, nmax, nq, tilde, tag,
              nls_sel, strats_sel, schemes_sel, multi_scheme)
end

"""
    plot_data(d) -> Figure

Draw, save and display the figure from `load_data`'s result. Nothing here re-reads
a cache, so retuning the layout costs the drawing alone:

    julia> data = load_data()
    julia> plot_data(data)
"""
function plot_data(d)
    curves, refs, dmft          = d.curves, d.refs, d.dmft
    U, U_num, T, t, nmax, nq    = d.U, d.U_num, d.T, d.t, d.nmax, d.nq
    tilde, tag                  = d.tilde, d.tag
    nls_sel, strats_sel         = d.nls_sel, d.strats_sel
    schemes_sel, multi_scheme   = d.schemes_sel, d.multi_scheme

    k_path, s_path = gamma_x_m_path(d.nk)

    println("\nPlotting Figure:")
    # font, line and marker sizes tuned for projection on a presentation slide
    figure = with_theme(theme_latexfonts();
        fontsize = 22,
        Axis   = (titlesize = 26, xlabelsize = 26, ylabelsize = 26,
                  xticklabelsize = 22, yticklabelsize = 22),
        Legend = (labelsize = 20,),
        Lines  = (linewidth = 3,),
        HLines = (linewidth = 2,),
    ) do
        fig = Figure(size = (1300, 1000))

        tick_pos = [0,π,2π,(2+sqrt(2))*π]
        tick_lab = [L"\Gamma", L"\text{X}", L"\text{M}", L"\Gamma"]

        # Attributes shared by all four panels (sizes come from the theme above).
        axis_kw = (
            xticks         = (tick_pos, tick_lab),
            xticksmirrored = true,
            xtickalign     = 1.0,
            yticksmirrored = true,
            ytickalign     = 1.0,
            limits         = (0, 2pi + pi*sqrt(2), nothing, nothing),
        )

        # Values of field `fld` along the Γ-X-M-Γ path, unscaled.
        path_vals(data, fld) = getproperty(data, fld)[k_path]

        # One panel: axis + ×10^mag annotation + every selected curve for field `fld`.
        # `mag = nothing` takes the exponent from the data — used in raw mode, whose
        # magnitudes differ from the hand-tuned tilde ones.
        function panel!(row, col, fld; title, mag = nothing, ylabel = "")
            vals = [path_vals(c.data, fld) for c in curves]
            if mag === nothing
                vmax = maximum(maximum(abs, v) for v in vals)
                haskey(refs, fld) && (vmax = max(vmax, maximum(abs, refs[fld].val)))
                haskey(dmft, fld) && (vmax = max(vmax, abs(dmft[fld])))
                mag  = vmax > 0 ? floor(Int, log10(vmax)) : 0
            end
            scale = 10.0^(-mag)
            ax = Axis(fig[row, col]; axis_kw..., title, ylabel)
            # Above the frame rather than in the TopLeft() corner: the corner is where the
            # topmost y tick label sits, and the two collide once that tick is near the top.
            Label(fig[row, col, Top()], L"\times 10^{%$mag}";
                halign = :left, padding = (0, 0, 4, 0))
            tightlimits!(ax, Left(), Right())
            # The DMFT reference is momentum independent, hence a horizontal line (raw mode
            # only). Drawn first so the solver curves stay on top of it.
            if haskey(dmft, fld)
                hlines!(ax, [dmft[fld] * scale]; label = "DMFT", color = :gray,
                    linestyle = :dashdot)
            end
            # line widths come from the theme above
            for (c, v) in zip(curves, vals)
                lines!(ax, s_path, v .* scale; label = c.label,
                    color = c.color, linestyle = c.linestyle)
            end
            # Overlay the benchmark for this quantity (same ×10^(-mag) scale and the same
            # tilde/raw frame), with error bars. Labeling every panel is harmless — the
            # legend is shared.
            if haskey(refs, fld)
                r = refs[fld]
                errorbars!(ax, r.s, r.val .* scale, r.err .* scale; color = :black, whiskerwidth = 8)
                scatter!(  ax, r.s, r.val .* scale; label = "detQMC", marker = :circle, color = :transparent, strokewidth = 1.5, strokecolor = :black, markersize = 12)
            end
            return ax
        end

        # Panel titles / axis labels and the fixed tilde magnitudes; in raw mode the
        # exponents are derived from the data instead (mag = nothing).
        lab = tilde ?
            (ReΣ = L"\text{Re } \tilde{\Sigma}", ImΣ = L"\text{Im } \tilde{\Sigma}",
             χD  = L"\tilde{\chi}_D",            χM  = L"\tilde{\chi}_M",
             yΣ  = L"\tilde{\Sigma}(i\nu = \pi T, k)",
             yχ  = L"\tilde{\chi}(i\omega = 0, k)") :
            (ReΣ = L"\text{Re } \Sigma",         ImΣ = L"\text{Im } \Sigma",
             χD  = L"\chi_D",                    χM  = L"\chi_M",
             yΣ  = L"\Sigma(i\nu = \pi T, k)",
             yχ  = L"\chi(i\omega = 0, k)")
        mags = tilde ? (ReΣ = -2, ImΣ = -3, χD = -3, χM = -2) :
                       (ReΣ = nothing, ImΣ = nothing, χD = nothing, χM = nothing)

        # ax1 carries the full set of series labels, so the shared legend is built from it.
        ax1 = panel!(2, 1, :ReΣ; title = lab.ReΣ, mag = mags.ReΣ, ylabel = lab.yΣ)
              panel!(2, 2, :ImΣ; title = lab.ImΣ, mag = mags.ImΣ)
              panel!(3, 1, :χD;  title = lab.χD,  mag = mags.χD, ylabel = lab.yχ)
              panel!(3, 2, :χM;  title = lab.χM,  mag = mags.χM)

        # Title (row 0) and a single shared legend (row 1) above all plots.
        # With both schemes on screen the legend already names them, so the title only
        # carries the scheme when a single one is selected.
        scheme_title = multi_scheme ? "" : ", \\text{$(scheme_name[schemes_sel[1]])}"
        Tt, Ut = round(T / t, digits = 4), round(U_num / t, digits = 4)
        Label(fig[0, :], L"$T/t=%$(Tt)$, $U/t=%$(Ut)$, $\text{PHS}%$(scheme_title)$"; fontsize = 26)
        # nl_title = join((nl_name[nl] for nl in nls_sel), ", ")
        #  \text{, } t'=0 \text{ ; } \mathrm{nmax}=%$nmax \text{, } \mathrm{nq}=%$nq \text{ ; } %$(nl_title)

        Legend(fig[1, :], ax1;
            orientation  = :horizontal,
            framevisible = false,
            tellheight   = true,
            tellwidth    = false,
        )

        figdir = figures_dir()
        mkpath(figdir)
        stag  = join((ALGO_FAMILY[s] for s in strats_sel), "-")   # e.g. "fd-sc"
        itag  = join(schemes_sel, "-")                            # e.g. "anderson-mfRG"
        nltag = "NL" * join(nls_sel, "")                          # e.g. "NL12"
        out   = joinpath(figdir,
            "hubbard_U$(U)_T$(T)_nmax$(nmax)_nq$(nq)_$(nltag)_$(stag)_$(itag)_$(tag).svg")
        # Name the backend explicitly: `save` otherwise goes through whichever backend
        # is active, and in a session where GLMakie was activated after CairoMakie was
        # loaded (`using CairoMakie` only activates on first load) that is GLMakie,
        # which cannot write SVG. `display` below still uses the active backend, so an
        # interactive GLMakie window keeps working.
        save(out, fig; backend = CairoMakie)
        println("Saved figure to\n  $out")

        display(fig)
        return fig
    end

    return figure
end

# ---------------------------------------------------------------------------
# Executing this file does the whole thing. The two halves are also callable on
# their own, so a layout tweak need not re-read the caches:
#
#   julia> data = load_data()    # once
#   julia> plot_data(data)       # as often as you like
# ---------------------------------------------------------------------------
data = load_data()
fig  = plot_data(data)
