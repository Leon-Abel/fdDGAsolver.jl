# =============================================================================
# Reading the DMFT reference from TRIQS output.
#
# In fd-pDΓA the reference system is the DMFT impurity problem, solved with
# TRIQS (Lihm 2025, text below Eq. (7)). For each parameter set TRIQS writes
# three files: the propagators, the two- and three-point susceptibilities, and
# the full vertex, the latter two in the physical spin channels D (density),
# M (magnetic) and S (singlet). parse_triqs_data converts them into the
# code's objects and saves them to data/*.h5; load_triqs_data reads such a
# file back.
#
# Conversion TRIQS → code:
#   G, Σ   multiplied by i (code convention, dyson.jl). Σ is shifted by -U/2,
#          so that it carries no Hartree term at half filling, and the bare
#          propagator is shifted to match.
#   K1     K1 = -U² χ, with χ the two-point susceptibility (disconnected part
#          removed).
#   K2     from the three-point susceptibility χ3: s U χ3 / (G G), with s = +1
#          for D and -1 for M and S, is K1 + K2 plus the bare vertex of the
#          channel.
#   core   the full vertex (sign flipped, F_code = -F_TRIQS) minus U and all K1
#          and K2 parts. It becomes the RefVertex of the DMFT vertex
#          (refvertex.jl): (Λ - U) + ∑_r K3_r, never split further.
# Spin channels: S = pSp - xSp, T = pSp + xSp, D = 2 pSp + xSp, M = xSp
# (types.jl). The vertex fields `γp`, `γt`, `γa` hold the channels Φ_p, Φ_t,
# Φ_a.
# =============================================================================

# Spin convention used in the TRIQS input
# ``S = pSp - xSp``
# ``T = pSp + xSp``
# ``D = 2 * pSp + xSp``
# ``M = xSp``

# -----------------------------------------------------------------------------
# Propagators
# -----------------------------------------------------------------------------

"""
    _drop_first_dim(f) -> MeshFunction

Remove the first index of a TRIQS object. That index is the cluster momentum;
with a single impurity site it has length 1.
"""
function _drop_first_dim(f)
    MeshFunction(f.meshes[2:end], dropdims(f.data, dims=1))
end

"""
    load_triqs_2P(filename, U; half_filling = false) -> (; occ, G0, Σ, G)

Read the impurity propagators and the density from the TRIQS file `filename`
and convert them to the code's conventions:

- `occ`: density per spin.
- `G`, `Σ`: stored as iG, iΣ. Σ is shifted by -U/2, so that it has no Hartree
  term at half filling.
- `G0`: TRIQS's non-interacting impurity propagator (Weiss field), stored as
  iG0, with the same -U/2 shift moved into it: `G0⁻¹ → G0⁻¹ - U/2`. This is not
  the solver's reference propagator `S.G0`.

With `half_filling = true` it checks that `occ = 1/2`.
"""
function load_triqs_2P(filename, U; half_filling = false)
    f = h5open(filename, "r")
    occ = h5read(filename, "density")
    G0 = _drop_first_dim(load_triqs_gf(f, "cG0_k_iw"));
    G = _drop_first_dim(load_triqs_gf(f, "cG_k_iw"));
    Σ = _drop_first_dim(load_triqs_gf(f, "cSigma_k_iw"));
    close(f)

    if half_filling && abs(occ - 0.5) > 1e-4
        println("occupation occ = $occ")
        error("half_filling is set to true but the occupation is not 0.5.")
    end

    # Subtract U/2 because we put Σ = 0 at half filling and in TRIQS Σ = U/2 is half filling.
    Σ.data .-= U/2

    # We factor out -im from G and Σ and store im * G and im * Σ
    mult!(G0, im)
    mult!(G, im)
    mult!(Σ, im)

    # Update G0 to take the chemical potential shift by U/2 into account:
    # a constant self-energy U/2 (stored as i U/2) in the Dyson equation
    Σ_const = copy(G0)
    G0_new = copy(G0)
    set!(Σ_const, U/2 * im)
    Dyson!(G0_new, Σ_const, G0)
    G0 = G0_new

    (; occ, G0, Σ, G)
end

# -----------------------------------------------------------------------------
# Conversion of the full DMFT data
# -----------------------------------------------------------------------------

"""
    parse_triqs_data(prefix, T, U; params, half_filling = false, symmetrize = true, filename_output = nothing) -> (; G0, G, Σ, Γ, occ)

Convert the TRIQS output for one parameter set into the code's objects. It
reads the three files `<prefix>_DCA.h5` (propagators), `<prefix>_chi3_fromDCA.h5`
(two- and three-point susceptibilities) and `<prefix>_F_fromDCA.h5` (full
vertex). `T` is the temperature, `U` the interaction.

The DMFT vertex `Γ` (the reference f) is returned in the asymptotic form
`Vertex(core, Φ_p, Φ_t, Φ_a)`:

1. K1 from the two-point susceptibilities: `K1 = -U² χ` in the D, M, S
   channels, after removing the disconnected part of `χ_D`.
2. K2 from the three-point susceptibilities: `s U χ3 / (G G)` (s = +1 for D,
   -1 for M and S) minus K1 minus the bare vertex of the channel (D: +U,
   M: -U, S: +2U, Lihm 2025, Eq. (S3)).
3. Conversion to the stored component pSp of each channel:
   - p: `pSp = S/2`. The triplet part vanishes, since two equal spins cannot
     sit on the same site.
   - t: `pSp = (D - M)/2`.
   - a: `pSp = -M` (crossing, Eq. (S11)).
   K3 is left empty.
4. The core: the full vertex minus U, K1 and K2, stored as the RefVertex
   (refvertex.jl), in the t parametrization and, by converting the
   frequencies, in the p parametrization. Where a channel's K1/K2 argument lies
   outside its box the core is set to zero, because its K1/K2 part could not
   be subtracted there.

Options:
- `half_filling = true`: check the density, report how far G, Σ and the vertex
  are from being real (they must be at half filling), and drop the imaginary
  parts.
- `symmetrize = true`: impose the vertex symmetries (symmetries.jl) and report
  the errors. NOTE: this currently fails, because the symmetry functions are
  passed without their number-type argument (the solvers wrap them as
  `w -> sK1pp(w, Q)`).
- `filename_output`: save `G`, `G0`, `Σ`, `Γ`, `occ` and `params` (with `T`, `U`)
  there, in the format read by [`load_triqs_data`](@ref).
"""
function parse_triqs_data(prefix, T, U; params, half_filling = false, symmetrize = true, filename_output = nothing)

    filename_G = "$(prefix)_DCA.h5"
    filename_chi = "$(prefix)_chi3_fromDCA.h5"
    filename_F = "$(prefix)_F_fromDCA.h5"
    Ω0 = MatsubaraFrequency(T, 0, Boson)

    #-----------------------------------------------------------------------------#
    # Green function and self-energy
    (; occ, G0, Σ, G) = load_triqs_2P(filename_G, U; half_filling)

    #-----------------------------------------------------------------------------#
    # K1 vertex

    # Read K1 vertex: two-point susceptibilities in the D, M, S channels
    f = h5open(filename_chi, "r")
    χD = _drop_first_dim(load_triqs_gf(f, "chi2_D_k"));
    χM = _drop_first_dim(load_triqs_gf(f, "chi2_M_k"));
    χS = _drop_first_dim(load_triqs_gf(f, "chi2_S_k"));
    close(f)

    # Subtract disconnected part ⟨n⟩⟨n⟩ (at Ω = 0 only; 2n² β with n per spin)
    χD[Ω0] -= occ^2 * 2 / T

    # from χ2 compute K1 = -U² χ
    K1D = copy(χD)
    K1M = copy(χM)
    K1S = copy(χS)
    mult!(K1D, -U^2)
    mult!(K1M, -U^2)
    mult!(K1S, -U^2)

    #-----------------------------------------------------------------------------#
    # K2 vertex

    # three-point susceptibilities in the D, M, S channels
    f = h5open(filename_chi, "r")
    χ3D = load_triqs_gf(f, "chi3_D_k") |> _drop_first_dim |> _drop_first_dim;
    χ3M = load_triqs_gf(f, "chi3_M_k") |> _drop_first_dim |> _drop_first_dim;
    χ3S = load_triqs_gf(f, "chi3_S_k") |> _drop_first_dim |> _drop_first_dim;
    close(f)

    # Subtract disconnected part (factor 2 comes from the spin structure)
    # (-i) because we factor out (-i) from G
    for ν in meshes(χ3D, Val(2))
        χ3D[Ω0, ν] -= 2 * occ * G(value(ν)) * -im / T
    end

    # from χ3 and K1 compute K2 = U χ3 / (G G) - K1 - (bare vertex of the channel)
    K2D = copy(χ3D)
    K2M = copy(χ3M)
    K2S = copy(χ3S)

    for ν in meshes(K2D, Val(2)), Ω in meshes(K2D, Val(1))
        # minus sign because we factor out (-i) from G
        K2D[Ω, ν] =  U * χ3D[Ω, ν] / G(value(ν)) / G(value(Ω + ν)) - K1D(value(Ω)) - U
        K2M[Ω, ν] = -U * χ3M[Ω, ν] / G(value(ν)) / G(value(Ω + ν)) - K1M(value(Ω)) + U
        K2S[Ω, ν] = -U * χ3S[Ω, ν] / G(value(ν)) / G(value(Ω - ν)) - K1S(value(Ω)) - 2U
    end

    # Convert from the D, M, S spin basis to the pSp spin basis in each channel.
    # p channel pSp = S * 0.5 (the local triplet part vanishes)
    K1p = copy(K1S) * 0.5
    K2p = copy(K2S) * 0.5

    # t channel pSp = (D - M) * 0.5
    K1t = copy(K1D - K1M) * 0.5
    K2t = copy(K2D - K2M) * 0.5

    # a channel pSp = t channel xSp * (-1) = M * (-1)
    K1a = copy(K1M) * -1
    K2a = copy(K2M) * -1

    # Add dummy K3 vertex (we do not decompose the core part into the K3 vertex and the
    # 2-particle irreducible part, so we set K3 = 0, and the rest is in the RefVertex.)
    _mB = MatsubaraMesh(T, 1, Boson)
    _mF = MatsubaraMesh(T, 1, Fermion)
    _K3 = MeshFunction(_mB, _mF, _mF)
    set!(_K3, 0)

    γp = fdDGAsolver.Channel(K1p, K2p, _K3)
    γt = fdDGAsolver.Channel(K1t, K2t, _K3)
    γa = fdDGAsolver.Channel(K1a, K2a, _K3)

    # Define a K12-only vertex (U + K1 + K2), used to subtract the asymptotic contribution from the core.
    Γ_K12 = Vertex(RefVertex(T, U, eltype(γa)), γp, γt, γa)

    #-----------------------------------------------------------------------------#
    # Core vertex

    # full vertex in the D, M (t parametrization) and S channels; the sign
    # convention of TRIQS is opposite to ours
    f = h5open(filename_F, "r")
    F_D = load_triqs_gf(f, "F_D_k") |> _drop_first_dim |> _drop_first_dim |> _drop_first_dim
    F_M = load_triqs_gf(f, "F_M_k") |> _drop_first_dim |> _drop_first_dim |> _drop_first_dim
    F_S = load_triqs_gf(f, "F_S_k") |> _drop_first_dim |> _drop_first_dim |> _drop_first_dim
    close(f)

    mult!(F_D, -1)
    mult!(F_M, -1)
    mult!(F_S, -1)

    Ft_p = copy(F_D)
    set!(Ft_p, 0)
    Ft_x = copy(Ft_p)
    Fp_p = copy(Ft_p)
    Fp_x = copy(Ft_p)

    # Subtract asymptotic contributions from the core
    for ind in eachindex(Ft_p.data)
        Ω, ν, ω = value.(to_meshes(Ft_p, ind))
        # t channel
        # F_D = (tCh, D), F_M = (tCh, M), F_S = (pCh, S)
        # tCh, pSp = [(tCh, D) - (tCh, M)] / 2
        # tCh, xSp = (tCh, M)
        Ft_p[ind] = (F_D(Ω, ν, ω) - F_M(Ω, ν, ω)) * 0.5
        Ft_x[ind] = F_M(Ω, ν, ω)
        Ft_p.data[ind] -= Γ_K12(Ω, ν, ω, tCh, pSp)
        Ft_x.data[ind] -= Γ_K12(Ω, ν, ω, tCh, xSp)

        # p Channel: the same full vertex, read at the frequencies converted from
        # the p to the t parametrization
        # pCh, pSp = [(tCh, D) - (tCh, M)] / 2 at convert_frequency(pCh -> tCh)
        # pCh, xSp = (tCh, M) at convert_frequency(pCh -> tCh)
        Fp_p[ind] = (F_D(convert_frequency(Ω, ν, ω, pCh, tCh)...) - F_M(convert_frequency(Ω, ν, ω, pCh, tCh)...)) / 2
        Fp_x[ind] = F_M(convert_frequency(Ω, ν, ω, pCh, tCh)...)
        if all(is_inbounds.(convert_frequency(Ω, ν, ω, pCh, tCh), F_D.meshes))
            # Since we do frequency conversion, the box for Fp_p and Fp_x are different from
            # the box for F_D and F_M. We need to subtract the asymptotic contribution only
            # from the box of F_D and F_M.
            Fp_p.data[ind] -= Γ_K12(Ω, ν, ω, pCh, pSp)
            Fp_x.data[ind] -= Γ_K12(Ω, ν, ω, pCh, xSp)
        end
    end

    # Remove the core part if not covered by the K12 vertex: where any channel's
    # K1/K2/K̄2 argument lies outside its box, the asymptotic part could not be
    # subtracted, so the core is set to zero there.
    for ind in eachindex(Ft_p.data)
        Ω, ν, ω = value.(to_meshes(Ft_p, ind))

        for Ch in (aCh, pCh, tCh)
            Ω_, ν_, ω_ = convert_frequency(Ω, ν, ω, tCh, Ch)
            γ = Ch == aCh ? Γ_K12.γa : Ch == pCh ? Γ_K12.γp : Γ_K12.γt

            is_inbounds_K1  = all(is_inbounds.((Ω_,), γ.K1.meshes))
            is_inbounds_K2  = all(is_inbounds.((Ω_, ν_), γ.K2.meshes))
            is_inbounds_K2p = all(is_inbounds.((Ω_, ω_), γ.K2.meshes))

            if ! (is_inbounds_K1 && is_inbounds_K2 && is_inbounds_K2p)
                Ft_p.data[ind] = 0
                Ft_x.data[ind] = 0
            end
        end
    end

    # the same for the core in the p parametrization
    for ind in eachindex(Fp_p.data)
        Ω, ν, ω = value.(to_meshes(Fp_p, ind))

        for Ch in (aCh, pCh, tCh)
            Ω_, ν_, ω_ = convert_frequency(Ω, ν, ω, pCh, Ch)
            γ = Ch == aCh ? Γ_K12.γa : Ch == pCh ? Γ_K12.γp : Γ_K12.γt

            is_inbounds_K1  = all(is_inbounds.((Ω_,), γ.K1.meshes))
            is_inbounds_K2  = all(is_inbounds.((Ω_, ν_), γ.K2.meshes))
            is_inbounds_K2p = all(is_inbounds.((Ω_, ω_), γ.K2.meshes))

            if ! (is_inbounds_K1 && is_inbounds_K2 && is_inbounds_K2p)
                Fp_p.data[ind] = 0
                Fp_x.data[ind] = 0
            end
        end
    end

    # the core as RefVertex, and the full DMFT vertex f = core + channels
    Λ = RefVertex(U, Fp_p, Fp_x, Ft_p, Ft_x)

    Γ = Vertex(Λ, γp, γt, γa)

    if half_filling
        # Impose half filling symmetry.
        # The physical Green function and self-energy are purely imaginary.
        # Since we factor out -im from them,, G and Σ are purely real.
        # The vertices are purely real.
        println("Half-filling violation")
        println("G0    : ", maximum(abs.(imag.(G0.data))))
        println("G     : ", maximum(abs.(imag.(G.data))))
        println("Σ     : ", maximum(abs.(imag.(Σ.data))))
        println("γp.K1 : ", maximum(abs.(imag.(Γ.γp.K1.data))))
        println("γt.K1 : ", maximum(abs.(imag.(Γ.γt.K1.data))))
        println("γa.K1 : ", maximum(abs.(imag.(Γ.γa.K1.data))))
        println("γp.K2 : ", maximum(abs.(imag.(Γ.γp.K2.data))))
        println("γt.K2 : ", maximum(abs.(imag.(Γ.γt.K2.data))))
        println("γa.K2 : ", maximum(abs.(imag.(Γ.γa.K2.data))))
        println("Fp_p  : ", maximum(abs.(imag.(Γ.F0.Fp_p.data))))
        println("Fp_x  : ", maximum(abs.(imag.(Γ.F0.Fp_x.data))))
        println("Ft_p  : ", maximum(abs.(imag.(Γ.F0.Ft_p.data))))
        println("Ft_x  : ", maximum(abs.(imag.(Γ.F0.Ft_x.data))))

        println("Imposing half-filling...")
        G0.data .= real.(G0.data)
        G.data  .= real.(G.data)
        Σ.data  .= real.(Σ.data)
        Γ.γp.K1.data .= real.(Γ.γp.K1.data)
        Γ.γt.K1.data .= real.(Γ.γt.K1.data)
        Γ.γa.K1.data .= real.(Γ.γa.K1.data)
        Γ.γp.K2.data .= real.(Γ.γp.K2.data)
        Γ.γt.K2.data .= real.(Γ.γt.K2.data)
        Γ.γa.K2.data .= real.(Γ.γa.K2.data)
        Γ.F0.Fp_p.data .= real.(Γ.F0.Fp_p.data)
        Γ.F0.Fp_x.data .= real.(Γ.F0.Fp_x.data)
        Γ.F0.Ft_p.data .= real.(Γ.F0.Ft_p.data)
        Γ.F0.Ft_x.data .= real.(Γ.F0.Ft_x.data)
    end

    if symmetrize
        # NOTE: fails at the moment; the symmetry functions need the number
        # type as second argument (see the docstring)
        println("Imposing symmetries...")

        # particle-particle channel
        SGpp1 = my_SymmetryGroup([Symmetry{1}(sK1pp)], Γ.γp.K1)
        SGpp2 = my_SymmetryGroup([Symmetry{2}(sK2pp1), Symmetry{2}(sK2pp2)], Γ.γp.K2)
        SGpp3 = my_SymmetryGroup([Symmetry{3}(sK3pp1), Symmetry{3}(sK3pp2), Symmetry{3}(sK3pp3)], Γ.F0.Fp_p)

        # particle-hole channels
        SGph1 = my_SymmetryGroup([Symmetry{1}(sK1ph)], Γ.γt.K1)
        SGph2 = my_SymmetryGroup([Symmetry{2}(sK2ph1), Symmetry{2}(sK2ph2)], Γ.γt.K2)
        SGph3 = my_SymmetryGroup([Symmetry{3}(sK3ph1), Symmetry{3}(sK3ph2), Symmetry{3}(sK3ph3)], Γ.F0.Ft_p)

        println("symmetrization errors :")
        @info SGpp1(Γ.γp.K1)
        @info SGpp2(Γ.γp.K2)
        @info SGph1(Γ.γt.K1)
        @info SGph1(Γ.γa.K1)
        @info SGph2(Γ.γt.K2)
        @info SGph2(Γ.γa.K2)
        @info SGpp3(Γ.F0.Fp_p)
        @info SGpp3(Γ.F0.Fp_x)
        @info SGph3(Γ.F0.Ft_p)
        @info SGph3(Γ.F0.Ft_x)
    end

    if filename_output !== nothing
        f = h5open(filename_output, "w")
        save!(f, "G", G)
        save!(f, "G0", G0)
        save!(f, "Σ", Σ)
        save!(f, "Γ", Γ)
        f["occ"] = occ
        f["params/T"] = T
        f["params/U"] = U
        for key in keys(params)
            f["params/$key"] = getproperty(params, key)
        end
        close(f)
    end

    return (; G0, G, Σ, Γ, occ)
end

# -----------------------------------------------------------------------------
# Reading converted data
# -----------------------------------------------------------------------------

"""
    load_triqs_data(filename) -> (; G, G0, Σ, Γ, occ, params)

Read DMFT data written by [`parse_triqs_data`](@ref) (the files in data/):
the impurity propagators `G`, `G0` (non-interacting) and self-energy `Σ`
(stored as iG, iΣ, Σ without U/2), the DMFT vertex `Γ` (the reference f for
fd-pDΓA), the density per spin `occ`, and the model parameters `params`
(e.g. `T`, `U`, `μ`, `t1`, `t2`, `t3`).
"""
function load_triqs_data(filename)
    f = h5open(filename, "r")
    G = load_mesh_function(f, "G")
    G0 = load_mesh_function(f, "G0")
    Σ = load_mesh_function(f, "Σ")
    Γ = fdDGAsolver.load_vertex(Vertex, f, "Γ")
    occ = read(f, "occ")
    params = (; (Symbol(key) => val for (key, val) in read(f, "params"))...)
    close(f)
    (; G, G0, Σ, Γ, occ, params)
end
