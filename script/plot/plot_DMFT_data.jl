using CairoMakie
using GLMakie
GLMakie.activate!()
using LaTeXStrings
using MatsubaraFunctions
using fdDGAsolver
using HDF5

@isdefined(ScriptCommon) || include(joinpath(@__DIR__, "..", "common", "ScriptCommon.jl"))
using .ScriptCommon

# -----------------------------------------------------------------------
# Load DMFT data
# -----------------------------------------------------------------------
function load_dmft_data(filename_dmft_partial::String)
    filename_dmft = joinpath(data_dir(), filename_dmft_partial)
    return (; load_triqs_data(filename_dmft)..., source = "DMFT");
end

# -----------------------------------------------------------------------
# Load solver data
# -----------------------------------------------------------------------
# A solver written by generate_vertex.jl holds the same objects as a TRIQS file under
# different names, so it is wrapped into the shape `load_triqs_data` returns rather than
# duplicating the plotting code. Two mappings are not one-to-one:
#
#   G0  the solver's own `G0` is the *reference* propagator, which is identically zero
#       for the parquet approximation. The bath propagator that the DMFT file calls G0
#       is the solver's `Gbare`, so that is what is passed through here.
#   Γ   `S.F` carries the same γa/γp/γt → K1/K2 layout, but its core vertex sits in
#       γ.K3, not in F0 — see the note above `plot_core_vertex`.
#
# μ, t1, t2, t3 are not stored in the solver file (only T and U are recoverable, from
# the Matsubara mesh and from F0). Pass them if the figure titles should show them.
function load_solver_data(filename_solver_partial::String; source = "PA", μ = 0.0, t1 = 1.0, t2 = 0.0, t3 = 0.0)
    filename_solver = joinpath(output_dir(), "parquet_vertex", filename_solver_partial)

    G, Gbare, Σ, Γ = h5open(filename_solver, "r") do f
        load_mesh_function(f, "G"),
        load_mesh_function(f, "Gbare"),
        load_mesh_function(f, "Σ"),
        fdDGAsolver.load_vertex(Vertex, f, "F")
    end

    T = MatsubaraFunctions.temperature(meshes(G, Val(1)))
    U = real(Γ.F0.U)

    return (; G, G0 = Gbare, Σ, Γ, occ = compute_occupation(G), params = (; T, U, μ, t1, t2, t3), source)
end

# -----------------------------------------------------------------------
# Difference of two data sets
# -----------------------------------------------------------------------
# `difference_data(dmft, pa)` returns a third data set holding `dmft - pa` in the same
# shape the two loaders above return, so it goes straight into any of the plot
# functions below:
#
#   dmft = load_dmft_data("...h5")
#   pa   = load_solver_data("...h5")
#   plot_channels(difference_data(dmft, pa))
#
# The two sets generally do not share a frequency box — a TRIQS file and a solver run
# rarely agree on the K1, K2 and K3 sizes — so every quantity is subtracted on the
# largest box the two have in common, class by class. Frequencies are matched by value
# rather than by array position, so it does not matter which set carries the larger box.

# The Matsubara boxes are centred on zero and hence nested: at equal temperature the
# smaller of two meshes is contained in the larger one.
smaller_mesh(m1, m2) = N(m1) <= N(m2) ? m1 : m2

# fa - fb on the largest box the two share. Both are read by frequency value, which
# addresses the same physical frequency in either box.
function mesh_difference(
    fa :: MeshFunction{DD},
    fb :: MeshFunction{DD},
    Q  :: DataType = promote_type(eltype(fa.data), eltype(fb.data)),
    )  where {DD}

    ms   = ntuple(d -> smaller_mesh(meshes(fa)[d], meshes(fb)[d]), DD)
    vals = ntuple(d -> [value(p) for p in ms[d]], DD)
    Δ    = MeshFunction(ms...; data_t = Q)

    for idx in CartesianIndices(Δ.data)
        ω      = ntuple(d -> vals[d][idx[d]], DD)
        Δ[idx] = fa[ω...] - fb[ω...]
    end

    return Δ
end

# Δ of the two core vertices, in the shape `core_vertex` returns (see the note there):
# each core is reconstructed from its own vertex first and the two are subtracted only
# afterwards, on the box they share. Taking the difference this late is what keeps it
# exact — the reconstruction is not a plain array lookup but an evaluation, and it
# reaches outside the K3 box for the crossed spin components.
#
# The result is what `plot_core_vertex` draws for a difference data set; the bare
# interaction is already subtracted off both cores, and cancels between them whenever
# the two sets share it.
function difference_core(Γa, Γb, Q :: DataType = promote_type(eltype(Γa), eltype(Γb)))
    core_a, core_b = core_vertex(Γa), core_vertex(Γb)

    # the cores come back as plain arrays; wrapping them makes them addressable by
    # frequency, as everything else here is
    wrap(core, sym) = MeshFunction((core.mΩ, core.mν, core.mν), getproperty(core, sym))
    Δ(sym)          = mesh_difference(wrap(core_a, sym), wrap(core_b, sym), Q)

    Fp_p, Fp_x, Ft_p, Ft_x = Δ(:Fp_p), Δ(:Fp_x), Δ(:Ft_p), Δ(:Ft_x)
    mΩ, mν = meshes(Fp_p, Val(1)), meshes(Fp_p, Val(2))

    return (; Fp_p = Fp_p.data, Fp_x = Fp_x.data, Ft_p = Ft_p.data, Ft_x = Ft_x.data, mΩ, mν)
end

# Δ of two vertices, stored the way the TRIQS files store a vertex: the whole core goes
# into F0 and the γ.K3 are left at zero. Since core = F0 + Σ_r K3_r,
#
#   ΔF = Δcore + Σ_r (ΔK1_r + ΔK2_r + ΔK2'_r)
#
# still holds, so this evaluates as the difference of the two vertices everywhere inside
# the common box. Its bare interaction is zero, the core having been subtracted off.
#
# Note that reading the core back off this object with `core_vertex` is *not* the same as
# `difference_core` above: the crossed spin components are reconstructed through crossing
# relations that reach beyond the common box. That is why a difference data set carries
# its core along ready-made rather than leaving it to be recovered from F0.
function difference_vertex(Γa, Γb,
                           core :: NamedTuple = difference_core(Γa, Γb),
                           Q    :: DataType   = promote_type(eltype(Γa), eltype(Γb)))

    wrap(sym) = MeshFunction((core.mΩ, core.mν, core.mν), Q.(getproperty(core, sym)))
    F0        = RefVertex(0, wrap(:Fp_p), wrap(:Fp_x), wrap(:Ft_p), wrap(:Ft_x))

    # The channels keep their asymptotics only, K3 being zero on the core box. The
    # hierarchy numK1 ≥ numK2 ≥ numK3 survives taking the smaller box class by class,
    # since it holds for both sets separately.
    function Δchannel(γa, γb)
        K3 = MeshFunction(core.mΩ, core.mν, core.mν; data_t = Q)
        set!(K3, 0)

        # qualified: `Channel` is exported by both fdDGAsolver and Base
        return fdDGAsolver.Channel(mesh_difference(γa.K1, γb.K1, Q),
                                   mesh_difference(γa.K2, γb.K2, Q), K3)
    end

    return Vertex(F0, Δchannel(Γa.γp, Γb.γp), Δchannel(Γa.γt, Γb.γt), Δchannel(Γa.γa, Γb.γa))
end

# `source` overrides the label the figure titles carry; by default the two sources are
# spelled out in the order they are subtracted, e.g. "DMFT - PA".
function difference_data(data_a, data_b; source = nothing)
    # the frequency matching above identifies mesh points by their Matsubara *index*,
    # which only refers to the same frequency if both sets are at the same temperature
    Ta = MatsubaraFunctions.temperature(meshes(data_a.G, Val(1)))
    Tb = MatsubaraFunctions.temperature(meshes(data_b.G, Val(1)))
    @assert Ta ≈ Tb "Data sets are at different temperatures ($Ta vs. $Tb), difference is not defined"

    # U only drops out of the core difference if both sets carry the same interaction
    Ua, Ub = data_a.params.U, data_b.params.U
    Ua == Ub || @warn "Data sets have different U ($Ua vs. $Ub); the bare vertex does not cancel in the difference"

    src_a = get(data_a, :source, "")
    src_b = get(data_b, :source, "")
    src   = something(source, isempty(src_a) || isempty(src_b) ? "difference" : "$src_a - $src_b")

    core = difference_core(data_a.Γ, data_b.Γ)

    # the model parameters are carried over from the first set — they are what the
    # figure titles show, and the two sets are only comparable if they agree on them
    return (; G   = mesh_difference(data_a.G,  data_b.G),
              G0  = mesh_difference(data_a.G0, data_b.G0),
              Σ   = mesh_difference(data_a.Σ,  data_b.Σ),
              Γ   = difference_vertex(data_a.Γ, data_b.Γ, core),
              core,
              occ = data_a.occ - data_b.occ,
              params = data_a.params,
              source = src)
end

# -----------------------------------------------------------------------
# Figure title
# -----------------------------------------------------------------------
# Names the data source ("DMFT" or "PA") ahead of the model parameters, so the two sets
# of figures can be told apart at a glance. `source` is absent for a NamedTuple taken
# straight from `load_triqs_data` instead of through the loaders above, in which case
# the title is just the parameters.
function title_label(data)
    (; T, U, μ, t1, t2) = data.params
    src = get(data, :source, "")

    # the two branches are spelled out rather than composed: interpolating an L-string
    # into another L-string embeds its `$` delimiters and drops the prefix out of math mode
    return isempty(src) ?
        L"T = %$T,\quad U = %$U,\quad \mu = %$μ,\quad t_1 = %$t1,\quad t_2 = %$t2" :
        L"\mathrm{%$src}:\quad T = %$T,\quad U = %$U,\quad \mu = %$μ,\quad t_1 = %$t1,\quad t_2 = %$t2"
end

# -----------------------------------------------------------------------
# Plotting data
# -----------------------------------------------------------------------

# `nν` sets the plotted fermionic window to the first nν Matsubara frequencies on either
# side of zero; the stored boxes run out to ~100 · 2πT. Pass `nν = 0` to show the full box.
function plot_single_particle(data_triqs; nν::Int = 20, data_inspector::Bool = false)
    # -----------------------------------------------------------------------
    # Assign variables
    # -----------------------------------------------------------------------

    (; T, U, μ, t1, t2, t3) = data_triqs.params

    G  = data_triqs.G    # full propagator
    G0 = data_triqs.G0   # bare propagator
    Σ  = data_triqs.Σ    # self-energy

    # -----------------------------------------------------------------------
    # Plotting data
    # -----------------------------------------------------------------------
    # G, G0 and Σ are stored as im*G, im*G0, im*Σ (load_triqs.jl), so the physical
    # function is recovered as f = -im * stored:
    #   Re f = +imag(stored),  Im f = -real(stored)
    phys_re(f) = imag.(f.data)
    phys_im(f) = -real.(f.data)

    if maximum(abs.(phys_re(G))) == maximum(abs.(phys_re(G0))) == maximum(abs.(phys_re(Σ))) == 0
        @info "G, G0, Σ are purely imaginary."
    else
        @info "G, G0, Σ have a real part."
    end

    # all three share the same fermionic mesh
    gν       = G.meshes[1]
    ν_values = [plain_value(ν) for ν in gν]

    # -----------------------------------------------------------------------
    # Figure
    # -----------------------------------------------------------------------
    fig = with_theme(theme_latexfonts()) do
        fig = Figure(size = (1200, 420), fontsize = 18)

        # model parameters as a figure-wide title
        Label(fig[0, 1:3], title_label(data_triqs), fontsize = 22, tellwidth = false)

        # half a mesh spacing (2πT) of padding beyond the nν-th frequency
        ν_lims = (0, 15)#nν > 0 ? (-2π * T * (nν + 1), 2π * T * (nν + 1)) : nothing

        kwargs_ax = (xlabel = L"\nu",
                     xtickalign = 1, xticksmirrored = true,
                     ytickalign = 1, yticksmirrored = true,
                     # tick labels in units of π, as in the other panels
                     xticks = MultiplesTicks(5, π, "π"),
                     # fixed width for the y tick labels keeps the panels aligned
                     yticklabelspace = 55.0,
                     limits = (ν_lims, (nothing, 1e-10)))

        ax_Σ  = Axis(fig[1, 1]; title = L"\Sigma(i\nu)", kwargs_ax...)
        ax_G  = Axis(fig[1, 2]; title = L"G(i\nu)",      kwargs_ax...)
        ax_G0 = Axis(fig[1, 3]; title = L"G_0(i\nu)",    kwargs_ax...)

        # same fermionic mesh in all three panels
        linkxaxes!(ax_Σ, ax_G, ax_G0)

        for (ax, f) in zip((ax_Σ, ax_G, ax_G0), (Σ, G, G0))
            scatterlines!(ax, ν_values, phys_re(f); markersize = 9, label = L"\mathrm{Re}")
            scatterlines!(ax, ν_values, phys_im(f); markersize = 9, linestyle = :dash, label = L"\mathrm{Im}")
        end

        # one legend for the whole figure; all panels share the same two series
        axislegend(ax_Σ; position = :rt, framevisible = false)

        fig
    end

    if data_inspector
        DataInspector(fig)
    end

    display(GLMakie.Screen(), fig)

    return fig
end


# `nΩ` sets the plotted bosonic window to |Ω| ≤ nΩ · 2πT — the K1/K2 boxes reach out to
# ~100 · 2πT, where the channels have long decayed. Pass `nΩ = 0` to show the full box.
function plot_channels(data_triqs; nΩ::Int = 20, data_inspector::Bool = false)
    # -----------------------------------------------------------------------
    # Assign variables
    # -----------------------------------------------------------------------

    (; T, U, μ, t1, t2, t3) = data_triqs.params

    F  = data_triqs.Γ   # Full vertex

    # channels only stored in parallel spin parametrization
    φa = F.γa           # 2P-reducible in a-channel (just K1 and K2)
    φt = F.γt           # 2P-reducible in t-channel (just K1 and K2)
    φp = F.γp           # 2P-reducible in p-channel (just K1 and K2)

        # dim 255
        K1a = φa.K1         # K1 asymptote of a-channel
        K1t = φt.K1         # K1 asymptote of t-channel
        K1p = φp.K1         # K1 asymptote of p-channel

        # dim 147x100
        K2a = φa.K2         # K2 asymptote of a-channel
        K2t = φt.K2         # K2 asymptote of t-channel
        K2p = φp.K2         # K2 asymptote of p-channel

    # -----------------------------------------------------------------------
    # Plotting data
    # -----------------------------------------------------------------------
    if maximum(abs.(imag.(K1a.data))) == 0
        @info "The channels are purely real."
    else
        @info "The channels have an imaginary part."
    end

    K1a_data_real = real.(K1a.data)
    K1t_data_real = real.(K1t.data)
    K1p_data_real = real.(K1p.data)
    K2a_data_real = real.(K2a.data)
    K2t_data_real = real.(K2t.data)
    K2p_data_real = real.(K2p.data)
    K1a_data_imag = imag.(K1a.data)
    K1t_data_imag = imag.(K1t.data)
    K1p_data_imag = imag.(K1p.data)
    K2a_data_imag = imag.(K2a.data)
    K2t_data_imag = imag.(K2t.data)
    K2p_data_imag = imag.(K2p.data)

    # vertex meshes, same for all channels
    gω_K1 = K1a.meshes[1]
    gω_K2 = K2a.meshes[1]
    gν_K2 = K2a.meshes[2]

    # plain frequency values
    ω_K1_values = [plain_value(ω) for ω in gω_K1]
    ω_K2_values = [plain_value(ω) for ω in gω_K2]
    ν_K2_values = [plain_value(ν) for ν in gν_K2]

    # -----------------------------------------------------------------------
    # Figure
    # -----------------------------------------------------------------------
    # Layout (channels in columns a, p, t):
    #   K1a K1p K1t   — K1(Ω) over the full bosonic box
    #   K2a K2p K2t   — K2(Ω, ν) at the ν slice picked by the slider
    fig = with_theme(theme_latexfonts()) do
        fig = Figure(size = (1200, 700), fontsize = 18)

        # model parameters as a figure-wide title
        Label(fig[0, 1:3], title_label(data_triqs), fontsize = 22, tellwidth = false)

        # x is fixed to the requested window, y stays automatic (the channels differ
        # in magnitude, and the K2 slices rescale with ν)
        Ω_lims = nΩ > 0 ? (-2π * T * (nΩ + 0.5), 2π * T * (nΩ + 0.5)) : nothing

        kwargs_ax = (xlabel = L"\Omega",
                     xtickalign = 1, xticksmirrored = true,
                     ytickalign = 1, yticksmirrored = true,
                     # tick labels in units of π, as for the slider
                     xticks = MultiplesTicks(5, π, "π"),
                     # reserve a fixed width for the y tick labels (as `ticklabelspace`
                     # does for the colorbars), so the panels stay aligned and do not
                     # jitter when the slider rescales the K2 axes
                     yticklabelspace = 55.0,
                     limits = (Ω_lims, nothing))

        # row 1: K1, row 2: K2. Only the leftmost panel of each row is labelled.
        ax_K1a = Axis(fig[1, 1]; title = L"K_1^a", ylabel = L"K_1", kwargs_ax...)
        ax_K1p = Axis(fig[1, 2]; title = L"K_1^p", kwargs_ax...)
        ax_K1t = Axis(fig[1, 3]; title = L"K_1^t", kwargs_ax...)
        ax_K2a = Axis(fig[2, 1]; title = L"K_2^a", ylabel = L"K_2", kwargs_ax...)
        ax_K2p = Axis(fig[2, 2]; title = L"K_2^p", kwargs_ax...)
        ax_K2t = Axis(fig[2, 3]; title = L"K_2^t", kwargs_ax...)

        # panels in a row share the same bosonic mesh, so share the x-axis as well.
        # The y-axes are left independent since the channels differ in magnitude.
        linkxaxes!(ax_K1a, ax_K1p, ax_K1t)
        linkxaxes!(ax_K2a, ax_K2p, ax_K2t)

        # slider selecting the fermionic frequency ν of the K2 slices.
        # The fermionic mesh has no zero point; start just above it.
        slider_layout = GridLayout(fig[3, :])
        sl    = Slider(slider_layout[1, 1],
                    range = 1:length(ν_K2_values), startvalue = length(ν_K2_values) ÷ 2 + 1,
                    width = Relative(0.9), tellwidth = false)
        label = Label(slider_layout[2, 1],
                    lift(iν -> "ν = $(round(ν_K2_values[iν] / π, digits = 3))π", sl.value),
                    tellwidth = false)

        # left/right arrow keys step the slider by one ν. set_close_to! snaps to
        # the nearest element of the range, so the ends clamp on their own.
        on(events(fig).keyboardbutton, priority = 10) do event
            if event.action == Keyboard.press || event.action == Keyboard.repeat
                if event.key == Keyboard.left
                    set_close_to!(sl, sl.value[] - 1)
                    return Consume(true)
                elseif event.key == Keyboard.right
                    set_close_to!(sl, sl.value[] + 1)
                    return Consume(true)
                end
            end
            return Consume(false)
        end

        # K1 depends on Ω only — nothing to slice
        for (ax, re, im) in zip((ax_K1a, ax_K1p, ax_K1t),
                                (K1a_data_real, K1p_data_real, K1t_data_real),
                                (K1a_data_imag, K1p_data_imag, K1t_data_imag))
            scatterlines!(ax, ω_K1_values, re; markersize = 6, label = L"\mathrm{Re}")
            scatterlines!(ax, ω_K1_values, im; markersize = 6, linestyle = :dash, label = L"\mathrm{Im}")
        end

        # K2 is stored as [Ω, ν], so a ν slice is a column
        for (ax, re, im) in zip((ax_K2a, ax_K2p, ax_K2t),
                                (K2a_data_real, K2p_data_real, K2t_data_real),
                                (K2a_data_imag, K2p_data_imag, K2t_data_imag))
            slice_re = lift(iν -> re[:, iν], sl.value)
            slice_im = lift(iν -> im[:, iν], sl.value)
            scatterlines!(ax, ω_K2_values, slice_re; markersize = 6, label = L"\mathrm{Re}")
            scatterlines!(ax, ω_K2_values, slice_im; markersize = 6, linestyle = :dash, label = L"\mathrm{Im}")
            # the y-range of a slice changes with ν, so rescale on every update.
            # reset_limits! (unlike autolimits!) keeps the fixed Ω window.
            on(sl.value) do _
                reset_limits!(ax)
            end
        end

        # one legend for the whole figure; all panels share the same two series
        axislegend(ax_K1a; position = :rt, framevisible = false)

        fig
    end

    if data_inspector
        DataInspector(fig)
    end

    display(GLMakie.Screen(), fig)

    return fig
end

# -----------------------------------------------------------------------
# Core vertex — fully 2PI + K3 in all channels
# -----------------------------------------------------------------------
# The core is the same object for both data sources, but they park it in opposite
# slots: TRIQS fills F0 (Fp_p, Fp_x, Ft_p, Ft_x) and leaves γ.K3 empty, while a solver
# run has F0 = the bare RefVertex and carries everything in γa.K3 / γp.K3 / γt.K3.
# Zeroing the K1/K2 asymptotics and evaluating what is left therefore reconstructs it
# the same way in both cases, as F0 + Σ_r K3_r.
#
# The bare interaction is subtracted because F0(Ω, ν, ν', Ch, Sp) already carries ±U.
# On DMFT data this reproduces the stored F0 arrays exactly; on a parquet-approximation
# solver, where the 2PI vertex *is* the bare U, what remains is Σ_r K3_r.
#
# Evaluation runs over the K3 grid of whichever of the two actually holds the data —
# 47x32x32 for the TRIQS files, and the solver's own (larger) K3 box for a solver run.
function core_vertex(Γ)
    Γ_core = copy(Γ)
    for γ in (Γ_core.γa, Γ_core.γp, Γ_core.γt)
        set!(γ.K1, 0)
        set!(γ.K2, 0)
    end

    ref = fdDGAsolver.numK3(Γ.F0)[1] >= fdDGAsolver.numK3(Γ.γp)[1] ? Γ.F0.Fp_p : Γ.γp.K3
    mΩ, mν = meshes(ref, Val(1)), meshes(ref, Val(2))
    Ωs, νs = [value(p) for p in mΩ], [value(p) for p in mν]

    ev(Ch, Sp) = [Γ_core(Ω, ν, νp, Ch, Sp) - fdDGAsolver.bare_vertex(Γ.F0, Ch, Sp)
                  for Ω in Ωs, ν in νs, νp in νs]

    return (; Fp_p = ev(pCh, pSp), Fp_x = ev(pCh, xSp),
              Ft_p = ev(tCh, pSp), Ft_x = ev(tCh, xSp), mΩ, mν)
end

function plot_core_vertex(data_triqs; data_inspector::Bool = false)
    # -----------------------------------------------------------------------
    # Assign variables
    # -----------------------------------------------------------------------

    (; T, U, μ, t1, t2, t3) = data_triqs.params

    F  = data_triqs.Γ   # Full vertex

    # core vertex (fully 2PI + K3 in all channels). A data set may carry its core
    # ready-made — `difference_data` does, since the difference of two cores has to be
    # taken before, not after, they are packed back into a vertex.
    core = haskey(data_triqs, :core) ? data_triqs.core : core_vertex(F)

    # dim 47x32x32 for DMFT data, the solver's K3 box for a solver run
    Fp_p = core.Fp_p    # core vertex in p-channel with parallel spins
    Fp_x = core.Fp_x    # core vertex in p-channel with crossed spins
    Ft_p = core.Ft_p    # core vertex in t-channel with parallel spins
    Ft_x = core.Ft_x    # core vertex in t-channel with crossed spins

    # -----------------------------------------------------------------------
    # Plotting data
    # -----------------------------------------------------------------------
    # vertex data
    Fp_p_data_real = real.(Fp_p)
    Fp_x_data_real = real.(Fp_x)
    Ft_p_data_real = real.(Ft_p)
    Ft_x_data_real = real.(Ft_x)
    Fp_p_data_imag = imag.(Fp_p)
    Fp_x_data_imag = imag.(Fp_x)
    Ft_p_data_imag = imag.(Ft_p)
    Ft_x_data_imag = imag.(Ft_x)

    # the vertex being real in one parametrization implies it
    # being real in all other parametrizations as well.
    if maximum(abs.(imag.(Ft_p))) == 0
        @info "The core vertex is purely real."
    else
        @info "The core vertex has an imaginary part."
    end

    # vertex meshes, all parametrizations share the same meshes
    # fermionic meshes gν, gν' are identical
    gω = core.mΩ
    gν = core.mν

    # plain frequency values
    ω_values = [plain_value(ω) for ω in gω]
    ν_values = [plain_value(ν) for ν in gν]

    # heatmap coordinates are cell centers, so pad the limits by half a cell to
    # avoid clipping the outermost row/column
    Δν     = ν_values[2] - ν_values[1]
    ν_lims = (ν_values[1] - Δν/2, ν_values[end] + Δν/2)

    # -----------------------------------------------------------------------
    # Figure
    # -----------------------------------------------------------------------
    fig = with_theme(theme_latexfonts()) do
        fig = Figure(size = (1000,1050), fontsize = 18)

        # model parameters as a figure-wide title
        Label(fig[0, 1:4], title_label(data_triqs), fontsize = 22, tellwidth = false)

        kwargs_ax = (xlabel=L"\nu", ylabel=L"\nu'",
                    xtickalign = 1, xticksmirrored = true,
                    ytickalign = 1, yticksmirrored = true,
                    # tick labels in units of π, as for the slider
                    xticks = MultiplesTicks(4, π, "π"),
                    yticks = MultiplesTicks(4, π, "π"),
                    limits = (ν_lims..., ν_lims...),
                    aspect = DataAspect())
        # odd columns hold the axes, even columns the associated colorbars
        ax1 = Axis(fig[1, 1]; title = L"F_p^=", kwargs_ax...)
        ax2 = Axis(fig[1, 3]; title = L"F_p^\times", kwargs_ax...)
        ax3 = Axis(fig[2, 1]; title = L"F_t^=", kwargs_ax...)
        ax4 = Axis(fig[2, 3]; title = L"F_t^\times", kwargs_ax...)

        # slider to adjust ω (row 3 holds the ν = ν' cuts added below the heatmaps)
        slider_layout = GridLayout(fig[4, :])
        sl    = Slider(slider_layout[1, 1],
                    range = 1:length(ω_values), startvalue = (length(ω_values) + 1)/2,
                    width = Relative(0.9), tellwidth = false)
        label = Label(slider_layout[2, 1],
                    lift(ω -> "ω = $((ω - (length(ω_values) + 1)/2)*2T)π", sl.value),
                    tellwidth = false)

        # left/right arrow keys step the slider by one ω. set_close_to! snaps to
        # the nearest element of the range, so the ends clamp on their own.
        on(events(fig).keyboardbutton, priority = 10) do event
            if event.action == Keyboard.press || event.action == Keyboard.repeat
                if event.key == Keyboard.left
                    set_close_to!(sl, sl.value[] - 1)
                    return Consume(true)
                elseif event.key == Keyboard.right
                    set_close_to!(sl, sl.value[] + 1)
                    return Consume(true)
                end
            end
            return Consume(false)
        end

        # vertex ω-slices
        slice_Fp_p = lift(sl.value) do ω
            Fp_p_data_real[ω, :, :]
        end
        slice_Fp_x = lift(sl.value) do ω
            Fp_x_data_real[ω, :, :]
        end
        slice_Ft_p = lift(sl.value) do ω
            Ft_p_data_real[ω, :, :]
        end
        slice_Ft_x = lift(sl.value) do ω
            Ft_x_data_real[ω, :, :]
        end

        # slices are stored as [ν, ν'], so ν runs along x and ν' along y
        kwargs_hm = ()
        hm1 = heatmap!(ax1, ν_values, ν_values, slice_Fp_p; kwargs_hm...)
        hm2 = heatmap!(ax2, ν_values, ν_values, slice_Fp_x; kwargs_hm...)
        hm3 = heatmap!(ax3, ν_values, ν_values, slice_Ft_p; kwargs_hm...)
        hm4 = heatmap!(ax4, ν_values, ν_values, slice_Ft_x; kwargs_hm...)

        # one colorbar per panel, since the channels differ in magnitude. 
        kwargs_cb = (width = 12, ticklabelspace = 70.0)
        cb1 = Colorbar(fig[1, 2], hm1; kwargs_cb...)
        cb2 = Colorbar(fig[1, 4], hm2; kwargs_cb...)
        cb3 = Colorbar(fig[2, 2], hm3; kwargs_cb...)
        cb4 = Colorbar(fig[2, 4], hm4; kwargs_cb...)

        # keep each colorbar close to its own panel
        colgap!(fig.layout, 1, 10)
        colgap!(fig.layout, 3, 10)

        # -------------------------------------------------------------------
        # ν = ν' cut through the two t-channel panels, drawn directly below
        # their heatmaps and following the same ω slider.
        # -------------------------------------------------------------------
        kwargs_ax_diag = (xlabel = L"\nu",
                    xtickalign = 1, xticksmirrored = true,
                    ytickalign = 1, yticksmirrored = true,
                    xticks = MultiplesTicks(4, π, "π"),
                    # reserve a fixed width for the y tick labels so the two panels
                    # stay aligned and do not jitter when the slider rescales them
                    yticklabelspace = 55.0,
                    # fixed height, so the row does not take an equal share of the
                    # figure from the (DataAspect) heatmap rows above
                    height = 170,
                    limits = (ν_lims, nothing))

        # placed in the axis columns (1, 3); the even columns belong to the colorbars
        ax_diag_t_p = Axis(fig[3, 1]; title = L"F_t^=(\nu, \nu)", ylabel = L"F", kwargs_ax_diag...)
        ax_diag_t_x = Axis(fig[3, 3]; title = L"F_t^\times(\nu, \nu)", kwargs_ax_diag...)

        # both cuts run over the same fermionic mesh, so share the x-axis. The y-axes
        # are left independent since the two parametrizations differ in magnitude.
        linkxaxes!(ax_diag_t_p, ax_diag_t_x)

        # slices are stored as [ν, ν'], so the cut is the main diagonal
        diagonal(A) = lift(ω -> [A[ω, i, i] for i in eachindex(ν_values)], sl.value)

        for (ax, re, im) in zip((ax_diag_t_p, ax_diag_t_x),
                                (Ft_p_data_real, Ft_x_data_real),
                                (Ft_p_data_imag, Ft_x_data_imag))
            scatterlines!(ax, ν_values, diagonal(re); markersize = 6, label = L"\mathrm{Re}")
            scatterlines!(ax, ν_values, diagonal(im); markersize = 6, linestyle = :dash, label = L"\mathrm{Im}")
            # the y-range of the cut changes with ω, so rescale on every update.
            # reset_limits! (unlike autolimits!) keeps the fixed ν window.
            on(sl.value) do _
                reset_limits!(ax)
            end
        end

        # one legend for both cuts; they share the same two series
        axislegend(ax_diag_t_p; position = :rt, framevisible = false)

        fig
    end

    if data_inspector
        DataInspector(fig)
    end

    display(GLMakie.Screen(), fig)

    return fig
end

# TRIQS only supplies the vertex in the t-channel parametrization.
# The p-channel parametrization is derived from the t-channel parametrization:
#   ω_t = ω_p - ν - ν'
# Therefore a p-channel grid point can only have data if
#   |ω_p - ν - ν'| ≤ ω_max

# K2 classes have peak at ω = 0.

