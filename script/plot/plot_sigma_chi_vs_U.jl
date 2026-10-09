"""
plot_sigma_chi_vs_U.jl

Plot the self-energy and susceptibilities at a *fixed momentum* q as a function
of the interaction strength U — either the tilde differences to the DMFT
reference or the raw full-physical quantities (default):

  Re Σ(q, iν = iπT),  Im Σ(q, iν = iπT),  χ_D(q, iω = 0),  χ_M(q, iω = 0)   vs U

Any combination of the data sources

  * pDΓA  (:fdPA strategy)
  * PA    (:scPA strategy)
  * DMFT  (local reference)

each converged by either iteration scheme

  * :anderson  (parquet iteration with Anderson acceleration)
  * :mfRG      (multiloop functional RG)

and any combination of the three momentum-structure approximations

  * NL1   (K23 s-wave approximation)
  * NL2   (K3 s-wave approximation)
  * NL3   (full momentum dependence)

can be plotted together. Each (strategy, scheme, NL) triple is drawn as its own
curve, with the same encoding as plot_sigma_chi_vs_q.jl: the iteration scheme sets
the hue family and the dashing (:anderson blue and dashed, :mfRG orange and solid),
the strategy shades that hue (pDΓA/PA as variants of one color), and the NL method
picks the marker plus the dash pattern within its scheme's family. The DMFT
reference is momentum-, scheme- and NL-independent, so it is drawn once regardless
of how many are selected (gray, dash-dotted), and is only meaningful in raw mode.

The U values are discovered by scanning `output/` for saved solvers / caches on
the requested (nmax, nq, NL) grid. For each U the quantities are read from the
precomputed caches written by `precompute_sigma_chi.jl`; any missing cache is
generated on first use (the only step that reloads a solver).

The momentum q is given in units of π and snapped to the nearest dense-k-grid
point. Common points: Γ = "qx=0 qy=0", X = "qx=1 qy=0", M = "qx=1 qy=1".

Usage (from the repo root):
  julia --project=script script/plot/plot_sigma_chi_vs_U.jl [options]

Run with `--help` for the full option list and the current defaults. Every option
is named, so order never matters and only what you change needs to be passed.

  --nls          comma-separated NL methods to draw: 1 | 2 | 3
  --strategies   comma-separated scPA | fdPA (empty for a DMFT-only figure)
  --schemes      comma-separated anderson | mfRG
  --qx --qy      fixed momentum in units of π (1,1 → M = (π, π))
  --nmax --nq    vertex box half-size and coarse k-grid size
  --Umin --Umax  U-axis window
  --tilde        plot X̃ = X − X_DMFT instead of the raw X
  --no-dmft      omit the DMFT reference curve
  --resmax       largest admitted residual norm

  A (U, source, NL) point is plotted only if the residual norm stored with its cache
  is <= resmax; the solver's converged flag is reported in the drop warning but does
  not filter. Every run that converged sits below 1e-8 (median 7.5e-9); loosen it
  (e.g. 1e-6) to also plot the pDΓA points that stalled near the vertex divergence
  at U ≳ 2.04.

  Examples:
    julia --project=script script/plot/plot_sigma_chi_vs_U.jl
    julia --project=script script/plot/plot_sigma_chi_vs_U.jl --strategies=scPA --nls=1,2
    julia --project=script script/plot/plot_sigma_chi_vs_U.jl --qx=1 --qy=1 --nls=1
    julia --project=script script/plot/plot_sigma_chi_vs_U.jl --qx=1 --qy=1 --nls=1 --tilde --resmax=1e-6
    julia --project=script script/plot/plot_sigma_chi_vs_U.jl --schemes=anderson
"""

using HDF5
using CairoMakie
using DelimitedFiles
using Printf

include(joinpath(@__DIR__, "..", "postprocess", "precompute_sigma_chi.jl"))   # caches + precompute + dmft_temperature

# ---------------------------------------------------------------------------
# What executing this file in the REPL does. Edit these and execute it again.
#
# Running the file as a script overrides them from the command line - and these
# are the defaults `--help` reports, so the two entry points cannot drift apart.
# ---------------------------------------------------------------------------
nls        = "1,2,3"          # comma-separated NL methods to draw: 1 | 2 | 3
strategies = "fdPA"           # comma-separated scPA | fdPA (empty for a DMFT-only figure)
schemes    = "mfRG,anderson"  # comma-separated anderson | mfRG
qx         = 0.5              # q_x in units of π
qy         = 0.5              # q_y in units of π
Umin       = -Inf             # lower U-axis window bound
Umax       = Inf              # upper U-axis window bound
resmax     = 0.0001           # largest admitted residual norm
nmax       = 12               # half-box size for the vertex grids
nq         = 6                # linear k-grid size for the vertex mesh
tilde      = false            # plot X̃ = X − X_DMFT (the difference to DMFT) instead of ...
no_dmft    = false            # omit the DMFT reference curve

# ---------------------------------------------------------------------------
# CLI driver (skipped when this file is executed in the REPL or `include`d)
# ---------------------------------------------------------------------------
if abspath(PROGRAM_FILE) == @__FILE__
    settings = cli_settings("Plot Σ and χ versus U at a fixed momentum.")
    add_arg_table!(settings,
        ["--nls"], Dict(:help => "comma-separated NL methods to draw: 1 | 2 | 3",
                       :arg_type => String, :default => nls),
        ["--strategies"], Dict(:help => "comma-separated scPA | fdPA (empty for a DMFT-only figure)",
                               :arg_type => String, :default => strategies),
        ["--schemes"],    Dict(:help => "comma-separated anderson | mfRG",
                               :arg_type => String, :default => schemes),
        ["--qx"],     Dict(:help => "q_x in units of π", :arg_type => Float64, :default => qx),
        ["--qy"],     Dict(:help => "q_y in units of π", :arg_type => Float64, :default => qy),
        ["--Umin"],   Dict(:help => "lower U-axis window bound", :arg_type => Float64, :default => Umin),
        ["--Umax"],   Dict(:help => "upper U-axis window bound", :arg_type => Float64, :default => Umax),
        ["--resmax"], Dict(:help => "largest admitted residual norm", :arg_type => Float64, :default => resmax))
    # NL_method = nothing: the momentum structures drawn come from `--nls`, so the
    # scalar `--NL` would only be a second option one capital apart that did nothing.
    add_box_args!(settings; nmax, nq, NL_method = nothing)
    add_flags!(settings,
        "tilde"   => "plot X̃ = X − X_DMFT (the difference to DMFT) instead of the raw X",
        "no-dmft" => "omit the DMFT reference curve")

    parsed = parse_args(ARGS, settings)

    nls        = parsed["nls"]
    strategies = parsed["strategies"]
    schemes    = parsed["schemes"]
    qx         = parsed["qx"]
    qy         = parsed["qy"]
    Umin       = parsed["Umin"]
    Umax       = parsed["Umax"]
    resmax     = parsed["resmax"]
    nmax       = parsed["nmax"]
    nq         = parsed["nq"]
    tilde      = parsed["tilde"]
    no_dmft    = parsed["no_dmft"]
end

# -----------------------------------------------------------------------
# Curve styling. Same encoding as plot_sigma_chi_vs_q.jl: the iteration scheme picks
# the hue family — Anderson blue, mfRG orange — so the two routes are never mistaken
# for one series, and within a family the strategy picks the shade, so pDΓA and PA
# read as variants of the same color. Line style reinforces the scheme (:anderson
# dashed, :mfRG solid) with the NL method varying the pattern inside that family; the
# marker follows the NL method too, so a curve stays identifiable in print.
# -----------------------------------------------------------------------
wong        = Makie.wong_colors()
algo_color  = Dict((:fdPA, :anderson) => wong[1], (:scPA, :anderson) => wong[5],   # blues
                   (:fdPA, :mfRG)     => wong[2], (:scPA, :mfRG)     => wong[6])   # oranges
algo_name   = Dict(:fdPA => "pDΓA", :scPA => "PA")
scheme_name = Dict(:anderson => "Anderson", :mfRG => "mfRG")
nl_marker   = Dict(1 => :circle, 2 => :diamond, 3 => :rect)
nl_name     = Dict(1 => "K23 s-wave", 2 => "K3 s-wave", 3 => "full q")

# The NL method varies the pattern *within* its scheme's family, so "Anderson is
# the dashed one" stays true even with several NL methods on screen.
nl_style    = Dict(
    :anderson => Dict(1 => :dash,  2 => :dashdot, 3 => :dashdotdot),
    :mfRG     => Dict(1 => :solid, 2 => :dot,     3 => (:dot, :dense)))

# -----------------------------------------------------------------------
# detQMC L16 reference point at U/t = 8.356 (internal U = 2.089), at the fixed q
# -----------------------------------------------------------------------
# The Γ-X-M-Γ detQMC files in data/ are a single-U dataset; here we pull out only
# the point at the plotted momentum q and show it at U/t = 4·U on every panel. As
# in plot_sigma_chi_vs_q.jl the files hold the raw quantities: Re Σ is negated to
# match the solver's sign convention, and in tilde mode the (momentum-independent)
# DMFT reference is subtracted — in raw mode the values are used as-is. Only added
# when U = 2.089 falls inside the [Umin, Umax] window. Keyed by the field symbols
# panel! uses. Absent files / a q not on the path silently skip.
qmc_U_str = "2.089"                                 # as the DMFT file name spells it
qmc_U    = parse(Float64, qmc_U_str)                # U/t = qmc_U/t = 8.356
qmc_sign = -1                                       # Re Σ sign flip vs the solver
qmc_dir  = data_dir()

# Row of a Γ-X-M-Γ path-file matching the plotted q (in units of π), C4 partner too.
function qmc_row(d, qx_pi, qy_pi)
    kx, ky = qx_pi * π, qy_pi * π
    for i in axes(d, 1)
        (isapprox(d[i, 2], kx; atol = 1e-5) && isapprox(d[i, 3], ky; atol = 1e-5)) && return i
        (isapprox(d[i, 2], ky; atol = 1e-5) && isapprox(d[i, 3], kx; atol = 1e-5)) && return i
    end
    return nothing
end

# (value, error) at the fixed q from one path-file, tilde-ized by `sub`, sign `sgn`.
function qmc_point(fname, vcol, ecol, qx_pi, qy_pi; sub = 0.0, sgn = 1)
    path = joinpath(qmc_dir, fname)
    isfile(path) || (@warn "No detQMC reference at\n  $path"; return nothing)
    d = readdlm(path)
    i = qmc_row(d, qx_pi, qy_pi)
    i === nothing && return nothing
    return (val = sgn * Float64(d[i, vcol]) - sub, err = Float64(d[i, ecol]))
end

"""
    load_data(; nls, strategies, schemes, qx, qy, Umin, Umax, resmax, nmax, nq,
                tilde, no_dmft) -> NamedTuple

Discover the U on disk, read the caches at the fixed momentum (generating any that
are missing), the DMFT reference and the detQMC point, and assemble every curve the
figure draws. Each keyword defaults to the parameter block above, so `load_data()`
loads exactly what executing this file would.
"""
function load_data(; nls        = nls,
                     strategies = strategies,
                     schemes    = schemes,
                     qx         = qx,
                     qy         = qy,
                     Umin       = Umin,
                     Umax       = Umax,
                     resmax     = resmax,
                     nmax       = nmax,
                     nq         = nq,
                     tilde      = tilde,
                     no_dmft    = no_dmft)

    qx_pi       = qx
    qy_pi       = qy
    nls_sel     = parse_nls(nls)
    # mfRG first by default, so the Anderson curves end up on top where the two
    # nearly coincide — which, being the same fixed point, is most of the range.
    schemes_sel = parse_schemes(schemes)
    # May legitimately be empty — a DMFT-only figure is a valid request.
    strats_sel  = isempty(strategies) ? Symbol[] : parse_strategies(strategies)
    show_dmft   = !no_dmft

    # Accuracy cut: a (U, run, NL) point is plotted iff the residual norm recorded
    # with its cache is <= resmax. The solver's binary converged flag is NOT a filter
    # — it only records whether the run crossed its own tol (1e-8) inside maxiter,
    # which throws away runs that stalled just above it. Every run that converged
    # sits below 1e-8 (median 7.5e-9); loosen it (e.g. 1e-6) to also plot the pDΓA
    # points that stalled near the vertex divergence at U ≳ 2.04.

    multi_scheme = length(schemes_sel) > 1
    multi_nl     = length(nls_sel)     > 1

    # `tilde` picks both the cache folder and the dataset names, and names the figure.
    tag = tilde ? "tilde" : "raw"

    # The DMFT curve referenced to itself is identically zero in tilde mode.
    if show_dmft && tilde
        @warn "DMFT reference is 0 by construction in tilde mode; dropping the DMFT curve."
        show_dmft = false
    end

    # -----------------------------------------------------------------------
    # Shared helpers
    # -----------------------------------------------------------------------
    # `k_index`, `q_label`, `q_tex`, `scan_available`, `scan_dmft_U`, `windowed_U`,
    # the bare-bubble block and `dmft_model_params` all come from
    # common/plotting.jl.

    avail = Dict(nl => scan_available(nmax, nq, nl) for nl in nls_sel)   # NL => (U => runs)

    # The U axis: every U appearing for a selected NL, plus (when DMFT is drawn) every
    # U with a DMFT reference file, restricted to the window. U stays the string it is
    # spelled with, since that is what names the files.
    Us = windowed_U(avail, nls_sel; Umin, Umax,
                    extra = show_dmft ? scan_dmft_U() : String[])
    isempty(Us) && error(
        "No cached / solver / DMFT data for nmax=$nmax nq=$nq NL=$(join(nls_sel, ",")) " *
        "in U ∈ [$Umin, $Umax] under\n  $(output_dir())")

    window = isfinite(Umin) || isfinite(Umax) ? " in U ∈ [$Umin, $Umax]" : ""
    println("Sources: ", join(strategy_label.(strats_sel), ", "),
            show_dmft ? (isempty(strats_sel) ? "DMFT" : ", DMFT") : "")
    println("Iteration schemes: ", join(schemes_sel, ", "))
    println("NL methods: ", join(["NL$nl" for nl in nls_sel], ", "))
    println("Found $(length(Us)) interaction strength(s) at nmax=$nmax, nq=$nq$window")
    @printf "Accuracy cut: residual norm <= %.1e\n" resmax
    println("Fixed momentum q = $(q_label(qx_pi, qy_pi))   (mode = $tag)\n")

    # -----------------------------------------------------------------------
    # Non-interacting (U = 0) susceptibility — the bare particle-hole bubble
    # -----------------------------------------------------------------------
    # At U = 0 the vertex vanishes, so χ_D = χ_M = χ₀/2 with χ₀ the static (iω = 0)
    # Lindhard bubble of the bare band — a cheap lattice sum, not a solver run. The
    # lattice curves (pDΓA / PA) take the q-resolved bubble χ₀(q)/2; the
    # momentum-local DMFT curve takes the q-averaged bubble χ₀,loc/2.
    # `bare_chi_q`, `bare_chi_local` and `dmft_model_params` live in
    # common/plotting.jl.
    _mp        = dmft_model_params(Us)
    # The unit the U and T axes are quoted in. Read off the reference files rather
    # than written here as `t = 0.25`: `dmft_model_params` has already loaded them,
    # and a constant is only right for as long as every input agrees with it. (t1 is
    # stored negative; the axis unit is its magnitude — see `dmft_hopping`.)
    t          = _mp === nothing ? 0.25 : abs(_mp.t1)
    chi_U0_q   = _mp === nothing ? NaN : bare_chi_q(qx_pi, qy_pi; _mp...)
    chi_U0_loc = _mp === nothing ? NaN : bare_chi_local(; _mp...)
    if _mp !== nothing && Umin <= 0 <= Umax
        println("Bare bubble at U=0:  χ(q) = $(round(chi_U0_q, digits=4)),  χ_loc = $(round(chi_U0_loc, digits=4))\n")
    end

    # -----------------------------------------------------------------------
    # Collect the selected (tilde or raw) quantities at the fixed q, versus U
    # -----------------------------------------------------------------------
    # Prepend the exact non-interacting point at U = 0: the self-energy vanishes
    # identically (Re Σ = Im Σ = 0, in both raw and tilde mode since Σ_DMFT = 0 at U = 0),
    # and the susceptibilities take the bare bubble `χ0` supplied by the caller (q-resolved
    # for the lattice curves, q-averaged for DMFT; in tilde mode with its U = 0 DMFT
    # reference already removed). Only added when the curve has real data and U = 0 lies
    # inside the window.
    function add_noninteracting_point!(us, ReΣ, ImΣ, χD, χM, χ0)
        (isempty(us) || !(Umin <= 0 <= Umax)) && return
        pushfirst!(us,  0.0)
        pushfirst!(ReΣ, 0.0); pushfirst!(ImΣ, 0.0)
        pushfirst!(χD,  χ0);  pushfirst!(χM,  χ0)
        return
    end

    # The caches store every frequency of the solver grid; these panels want one row of
    # each. `nu0_row` / `omega0_row` live in common/plotting.jl rather than in
    # precompute_sigma_chi.jl, which stores the axes and leaves their interpretation
    # to the consumer.

    # Read (generating on first use) the cached Σ / χ at the fixed q for one run
    # (strategy, scheme) on one NL method, over every windowed U that has that run.
    # Reads the tilde or raw caches per `tilde`. Returns parallel vectors.
    function collect_vs_U(strategy, scheme, NL_method, available)
        us  = Float64[]
        ReΣ = Float64[]; ImΣ = Float64[]
        χD  = Float64[]; χM  = Float64[]
        idx = 0

        run_tag = "$(ALGO_FAMILY[strategy])_$(scheme)"

        Ulist = sort!(collect(keys(available)); by = u -> parse(Float64, u))
        filter!(u -> Umin <= parse(Float64, u) <= Umax, Ulist)

        for U in Ulist
            (strategy, scheme) in available[U] || continue   # this run not available at this U

            T     = dmft_temperature(U)
            file = postprocess_path(U, T, nmax, nq, NL_method, strategy, scheme)
            if !isfile(file)
                try
                    # U goes over as the string it was discovered as: that is what names
                    # both the cache being written and the solver being read. `precompute`
                    # splits off the number it needs for the arithmetic (χ ~ 1/U²) itself.
                    precompute(U, nmax, nq, NL_method, strategy, scheme)
                catch err
                    @warn "Could not obtain $(run_tag)/NL$(NL_method) data for U=$U; skipping it." exception = err
                    continue
                end
                isfile(file) || continue
            end

            # Accuracy filter: keep U only if the solve got its residual below `resmax`.
            # The residual decides on its own — a run that ran out of iterations just
            # short of its own `tol` is still usable data.
            residual = cache_residual(file)
            if !(residual <= resmax)
                @warn @sprintf("Dropping %s/NL%d at U=%s: residual %.3e > resmax %.1e.",
                    run_tag, NL_method, U, residual, resmax)
                continue
            end

            # One open. `dset[i, idx]` is a hyperslab read: only the single
            # (ν = iπT / Ω = 0, q) element is pulled, not the whole (nν × nk²) matrix.
            Σ, χ = h5open(file, "r") do f
                idx = k_index(qx_pi, qy_pi, read(f, "params/nk"))
                iν  = nu0_row(read(f, "sigma/nu"), T)
                iΩ  = omega0_row(read(f, "chi/omega"))
                ((Re = f["sigma/$(tag)/re"][iν, idx], Im = f["sigma/$(tag)/im"][iν, idx]),
                 (D  = f["chi/$(tag)/D"][iΩ, idx],    M  = f["chi/$(tag)/M"][iΩ, idx]))
            end

            push!(us,  parse(Float64, U))
            push!(ReΣ, Σ.Re); push!(ImΣ, Σ.Im)
            push!(χD,  χ.D);  push!(χM,  χ.M)
        end
        # Tilde caches store χ − χ_DMFT, so the U = 0 point must subtract the U = 0 DMFT
        # reference too — which is exactly the local bare bubble chi_U0_loc.
        add_noninteracting_point!(us, ReΣ, ImΣ, χD, χM,
            tilde ? chi_U0_q - chi_U0_loc : chi_U0_q)   # q-resolved bubble (tilde: minus local)
        return (; us, ReΣ, ImΣ, χD, χM, idx)
    end

    # DMFT local reference (momentum- and NL-independent), collected directly from
    # the reference data over every windowed U. Only drawn in raw mode.
    function collect_dmft(Ustrs)
        us  = Float64[]
        ReΣ = Float64[]; ImΣ = Float64[]
        χD  = Float64[]; χM  = Float64[]
        for U in sort!(collect(Ustrs); by = u -> parse(Float64, u))
            Uf = parse(Float64, U)
            Umin <= Uf <= Umax || continue
            try
                # As spelled: `dmft_reference` picks the file with the string and does its
                # arithmetic (…/U²) with the number it parses out of it.
                ref = dmft_reference(U)
                push!(us,  Uf)
                # Same sign convention as precompute's raw Σ (S.Σ stores im·Σ_phys).
                push!(ReΣ,  imag(ref.Σ_DMFT_ν0))
                push!(ImΣ, -real(ref.Σ_DMFT_ν0))
                push!(χD,   ref.chi_D_DMFT)
                push!(χM,   ref.chi_M_DMFT)
            catch err
                @warn "No DMFT reference for U=$U; skipping it." exception = err
            end
        end
        add_noninteracting_point!(us, ReΣ, ImΣ, χD, χM, chi_U0_loc)   # q-averaged bubble
        return (; us, ReΣ, ImΣ, χD, χM, idx = 0)
    end

    # -----------------------------------------------------------------------
    # Build the list of curves to draw (styling above). Curves that yield no data are
    # dropped.
    # -----------------------------------------------------------------------
    curves = NamedTuple[]   # each: (; data, label, color, marker, linestyle)
    for strategy in strats_sel, scheme in schemes_sel, nl in nls_sel
        data = collect_vs_U(strategy, scheme, nl, avail[nl])
        isempty(data.us) && continue
        # The scheme and the NL method are only spelled out when more than one of each
        # is on screen, keeping a single-selection legend short.
        label = algo_name[strategy] *
            (multi_scheme ? " $(scheme_name[scheme])" : "") *
            (multi_nl     ? " $(nl_name[nl])"         : "")
        push!(curves, (; data,
            label,
            color     = algo_color[(strategy, scheme)],
            marker    = nl_marker[nl],
            linestyle = nl_style[scheme][nl]))
    end

    dmft_data = show_dmft ? collect_dmft(Us) : (; us = Float64[])
    has_dmft  = show_dmft && !isempty(dmft_data.us)

    if isempty(curves) && !has_dmft
        error("No cached / derivable data at q = $(q_label(qx_pi, qy_pi)) for the requested selection.")
    end

    # Realized (snapped) momentum, for the record — from the first curve with data.
    let nk = 48
        idx = isempty(curves) ? k_index(qx_pi, qy_pi, nk) : curves[1].data.idx
        ix  = mod(idx - 1, nk) + 1
        iy  = (idx - 1) ÷ nk + 1
        println("Snapped to grid point ix=$ix, iy=$iy  (q ≈ ($(round(2(ix-1)/nk, digits=3))π, $(round(2(iy-1)/nk, digits=3))π))\n")
    end

    # detQMC point at the fixed q (see the block above `load_data`).
    qmc = Dict{Symbol, Any}()
    if Umin <= qmc_U <= Umax
        # DMFT subtractions apply only in tilde mode (raw mode uses the values as-is).
        subRe = subIm = subD = subM = 0.0
        if tilde
            dref  = dmft_reference(qmc_U_str)
            subRe = imag(dref.Σ_DMFT_ν0)                # ≡ 0 at PHS
            subIm = -real(dref.Σ_DMFT_ν0)
            subD  = dref.chi_D_DMFT
            subM  = dref.chi_M_DMFT
        end
        for (key, fname, vcol, ecol, sub, sgn) in (
                (:ReΣ, "SlfEngkIwnPathGXMG_L16.txt", 4, 5, subRe, qmc_sign),
                (:ImΣ, "SlfEngkIwnPathGXMG_L16.txt", 6, 7, subIm, 1),
                (:χD,  "DenDenkIwnPathGXMG_L16.txt", 4, 5, subD,  1),
                (:χM,  "SpinZZkIwnPathGXMG_L16.txt", 4, 5, subM,  1),
            )
            p = qmc_point(fname, vcol, ecol, qx_pi, qy_pi; sub = sub, sgn = sgn)
            p === nothing || (qmc[key] = p)
        end
        isempty(qmc) || println("detQMC L16 point at U/t = $(qmc_U/t), q = $(q_label(qx_pi, qy_pi))\n")
    end

    # U is the abscissa, so only T goes in the title; all U share the same T via the
    # DMFT reference. Taken from the first U that has one.
    Tref = nothing
    for u in Us
        try
            Tref = dmft_temperature(u)
            break
        catch
        end
    end

    return (; curves, dmft_data, has_dmft, qmc, t, Tref, Us, tag, tilde, show_dmft,
              qx_pi, qy_pi, Umin, Umax, resmax, nmax, nq,
              nls_sel, strats_sel, schemes_sel)
end

"""
    plot_data(d) -> Figure

Draw, save and display the figure from `load_data`'s result. Nothing here re-reads
a cache, so retuning the layout costs the drawing alone:

    julia> data = load_data()
    julia> plot_data(data)
"""
function plot_data(d)
    curves, dmft_data, has_dmft, qmc = d.curves, d.dmft_data, d.has_dmft, d.qmc
    t, Tref, Us, tag, tilde          = d.t, d.Tref, d.Us, d.tag, d.tilde
    qx_pi, qy_pi                     = d.qx_pi, d.qy_pi
    Umin, Umax, resmax, nmax, nq     = d.Umin, d.Umax, d.resmax, d.nmax, d.nq
    nls_sel, strats_sel, schemes_sel = d.nls_sel, d.strats_sel, d.schemes_sel

    # Symbols for the axis labels: raw drops the tilde.
    Σsym = tilde ? "\\tilde{\\Sigma}" : "\\Sigma"
    χsym = tilde ? "\\tilde{\\chi}"   : "\\chi"

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
        fig = Figure(size = (1300, 1000))

        # Attributes shared by all four panels (sizes come from the theme above).
        axis_kw = (
            xticksmirrored = true,
            xtickalign     = 1.0,
            yticksmirrored = true,
            ytickalign     = 1.0,
        )

        # One panel: axis + ×10^mag annotation + every selected curve (and the DMFT
        # reference / detQMC point) for field `fld`, vs U. y data is rescaled by 10^mag so
        # the tick labels read as small numbers; the exponent comes from the data shown.
        function panel!(row, col, fld; title, xlabel = "", ylabel = "")
            vals = [getproperty(c.data, fld) for c in curves]
            has_dmft && push!(vals, getproperty(dmft_data, fld))
            vmax = maximum(maximum(abs, v) for v in vals)
            haskey(qmc, fld) && (vmax = max(vmax, abs(qmc[fld].val)))
            mag   = vmax > 0 ? floor(Int, log10(vmax)) : 0
            scale = 10.0^(-mag)

            ax = Axis(fig[row, col]; axis_kw..., title = title, xlabel = xlabel, ylabel = ylabel)
            # Above the frame rather than in the TopLeft() corner: the corner is where the
            # topmost y tick label sits, and the two collide once that tick is near the top.
            Label(fig[row, col, Top()], L"\times 10^{%$mag}";
                halign = :left, padding = (0, 0, 4, 0))
            tightlimits!(ax, Left(), Right())
            # line widths and marker sizes come from the theme above
            for c in curves
                scatterlines!(ax, c.data.us ./ t, getproperty(c.data, fld) .* scale;
                    label = c.label, color = c.color,
                    marker = c.marker, linestyle = c.linestyle)
            end
            # DMFT reference (raw, momentum- and NL-independent). Gray and dash-dotted, and
            # drawn on top of the solver curves, as in plot_sigma_chi_vs_{q,freq}.jl.
            if has_dmft
                scatterlines!(ax, dmft_data.us ./ t, getproperty(dmft_data, fld) .* scale;
                    label = "DMFT", color = :gray,
                    marker = :utriangle, linestyle = :dashdot)
            end
            # detQMC single point (with error bar)
            if haskey(qmc, fld)
                p = qmc[fld]
                errorbars!(ax, [qmc_U ./ t], [p.val * scale], [p.err * scale]; color = :black, whiskerwidth = 8)
                scatter!(ax, [qmc_U ./ t], [p.val * scale]; label = "detQMC", marker = :circle, color = :transparent, strokewidth = 1.5, strokecolor = :black, markersize = 12)
            end
            return ax
        end

        # The momentum is named in the y-labels (not the title), as in plot_sigma_chi_vs_freq.jl.
        qtex = q_tex(qx_pi, qy_pi)

        # ax1 (the Re Σ panel) carries the full set of series labels, so the shared
        # legend below is built from it.
        ax1 = panel!(2, 1, :ReΣ; title = L"\text{Re } %$(Σsym)",
            ylabel = L"%$(Σsym)(i\nu = \pi T, %$(qtex)) - \Sigma_H")
        panel!(2, 2, :ImΣ; title = L"\text{Im } %$(Σsym)")
        panel!(3, 1, :χD; title = L"%$(χsym)_D",
            xlabel = L"U/t", ylabel = L"%$(χsym)(i\omega = 0, %$(qtex))")
        panel!(3, 2, :χM; title = L"%$(χsym)_M", xlabel = L"U/t")

        # Title (row 0) and a single shared legend (row 1) above all plots. U is the abscissa
        # here, so only T/t goes in the title (all U share the same T via the DMFT reference).
        # Keep the whole LaTeXString on ONE source line: a line break inside L"..." would
        # inject the continuation's indentation as literal whitespace, opening a big gap
        # in the rendered title and pushing it off the figure.
        Tt = Tref === nothing ? nothing : round(Tref / t, digits = 4)
        Label(fig[0, :], Tref === nothing ? L"$\text{2D HM at PHS}$" :
                                            L"$T/t=%$(Tt)$, $\text{PHS}$"; fontsize = 26)

        # ax1 (the Re Σ panel) carries the full set of series labels, so the shared
        # legend is built from it.
        Legend(fig[1, :], ax1;
            orientation  = :horizontal,
            framevisible = false,
            tellheight   = true,
            tellwidth    = false,
        )

        figdir = figures_dir()
        mkpath(figdir)
        qtag  = "q$(replace(string(qx_pi), "." => "p"))_$(replace(string(qy_pi), "." => "p"))"
        stag  = join(vcat([ALGO_FAMILY[s] for s in strats_sel],
                          d.show_dmft ? ["dmft"] : []), "-")   # e.g. "fd-sc-dmft"
        itag  = join(schemes_sel, "-")                         # e.g. "mfRG-anderson"
        nltag = "NL" * join(nls_sel, "")                       # e.g. "NL23"
        utag  = (isfinite(Umin) || isfinite(Umax)) ? "_U$(Us[1])-$(Us[end])" : ""
        # Any cut other than the solver tolerance (1e-8) plots different points, so it
        # is tagged in the file name rather than overwriting the strict figure.
        rtag  = resmax == 1e-8 ? "" : @sprintf("_res%.0e", resmax)
        out   = joinpath(figdir, "sigma_chi_vs_U_$(tag)_$(qtag)_$(stag)_$(itag)_$(nltag)_nmax$(nmax)_nq$(nq)$(utag)$(rtag).svg")
        # Name the backend explicitly: `save` otherwise goes through whichever backend is
        # active, and in a session where GLMakie was activated after CairoMakie was loaded
        # that is GLMakie, which cannot write SVG. `display` below still uses the active
        # backend, so an interactive GLMakie window keeps working.
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
