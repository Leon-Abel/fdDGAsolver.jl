# =============================================================================
# Schwinger-Dyson equation (SDE) for the lattice solver NL2.
#
# As for NL (nonlocal/SDE.jl; Lihm 2025, Eq. (S36) with momenta), but the
# kernels keep the fermionic momentum k of the self-energy leg, like K2 does:
#
#   Lpp(Ω, ν; q, k) = U · 1/β 1/N_k ∑_{ν̃,k̃} Π_p(Ω, ν̃; q, k̃) Φ_p(Ω, Ω - ν̃, ν; q, q - k̃, k)
#   Lph(Ω, ν; q, k) = U · 1/β 1/N_k ∑_{ν̃,k̃} Π_a(Ω, ν̃; q, k̃) [Φ_a + Φ_t](Ω, ν, ν̃; q, k, k̃)
#
# They are closed with one propagator,
#
#   Σ(ν, k) = 1/β 1/N_q ∑_{Ω,q} [ G(Ω - ν, q - k) Lpp(Ω, ν; q, k)
#                               + G(Ω + ν, q + k) Lph(Ω, ν; q, k) ] + Σ_U² + Σ_H,
#
# in real space with FFTs over both momenta. As in the local code, SDE!
# (SDE.jl) applies this to every level of the nested vertex F → F.F0 → … .
# NL3 uses the same kernels, because its bubbles and SDE buffers have the
# same shape.
#
# NOTE (open question (a), docs/physics_and_notation.md): for a full vertex
# the kernels below take F(…; only γr) - F.F0(…; only γr). The lattice
# evaluator adds F.F0 in full (it does not pass these switches on), while a
# local Vertex applies them. So one level contributes its own Φ_r plus the
# other two channels of F.F0. With f = NL2_Vertex(Γ_dmft) (zero channels
# around the local DMFT vertex, as in the scripts) the DMFT reducible vertex
# enters SDE[f] about three times, and Σ = σ + SDE[F](G) - SDE[f](g) cancels
# the excess only where G = g. The own-channel form in the commented-out
# lines (as in NL) avoids this for plain vertices. MBE vertices apply the
# switches at every level and are not affected.
#
# Symbols as in SDE.jl: Φ_r reducible vertex of channel r (fields `γa`, `γp`,
# `γt`), Σ_H = U(n - 1/2) Hartree term, Σ_U² second-order term (Eq. (S46)).
# Paper ↔ code: q ↔ P; ν̃, k̃ (summed) ↔ ω, q; G and Σ stored as iG and iΣ.
# =============================================================================

# -----------------------------------------------------------------------------
# Channel kernels L
# -----------------------------------------------------------------------------

"""
    SDE_channel_L_pp!(Lpp::NL2_MF_K2, Πpp::NL2_MF_Π, F, SGpp2; mode, subtract_asymptotic = false) -> nothing

NL2 version of the p-channel SDE kernel,
`Lpp(Ω, ν; q, k) = U · 1/β 1/N_k ∑_{ν̃,k̃} Π_p(Ω, ν̃; q, k̃) Φ_p(Ω, Ω - ν̃, ν; q, q - k̃, k)`,
with `Φ_p` taken as `F - F.F0` with only the p channel switched on (see the
NOTE in the file header). For a `RefVertex` F it uses `F - U` instead.
`subtract_asymptotic = true` is not implemented.
"""
function SDE_channel_L_pp!(
    Lpp   :: NL2_MF_K2{Q},
    Πpp   :: NL2_MF_Π{Q},
    F     :: AbstractVertex{Q},
    SGpp2 :: SymmetryGroup
    ;
    mode  :: Symbol,
    subtract_asymptotic :: Bool = false,
    )     :: Nothing where {Q}

    subtract_asymptotic && error("Not implemented")

    # Lpp(Ω, ν; q, k) for one point: sum over ν̃ (called ω) and k̃ (called q)
    @inline function diagram(wtpl)

        Ω, ν, P, k = wtpl
        val     = zero(Q)
        Πslice  = view(Πpp, Ω, :, P, :)

        for i in eachindex(Πslice)
            ω = value(meshes(Πpp, Val(2))[i.I[1]])
            q = value(meshes(Πpp, Val(4))[i.I[2]])

            # own-channel form, as in NL:
            # val += bare_vertex(F) * Πslice[i] * F.γp(Ω, Ω - ω, ν, P, P - q, k)
            # U Π_p Φ_p(Ω, Ω - ν̃, ν; q, q - k̃, k), with Φ_p = F - F.F0 (p channel only)
            val += bare_vertex(F) * Πslice[i] * (
                +    F(Ω, Ω - ω, ν, P, P - q, k, pCh, pSp; γp = true, γa = false, γt = false)
                - F.F0(Ω, Ω - ω, ν, P, P - q, k, pCh, pSp; γp = true, γa = false, γt = false) )
        end

        # 1/β 1/N_k ∑_{ν̃,k̃}
        return temperature(F) * val / length(meshes(Πpp, Val(4)))
    end

    # evaluate one point per symmetry class; the symmetry group fills in the rest
    SGpp2(Lpp, InitFunction{4, Q}(diagram); mode = mode)

    return nothing
end

"""
    SDE_channel_L_ph!(Lph::NL2_MF_K2, Πph::NL2_MF_Π, F, SGph2; mode, subtract_asymptotic = false) -> nothing

NL2 version of the particle-hole SDE kernel,
`Lph(Ω, ν; q, k) = U · 1/β 1/N_k ∑_{ν̃,k̃} Π_a(Ω, ν̃; q, k̃) [Φ_a + Φ_t](Ω, ν, ν̃; q, k, k̃)`,
with each `Φ_r` taken as `F - F.F0` with only channel r switched on (see the
NOTE in the file header). For a `RefVertex` F it uses `F - U` in the a and t
channel forms. `subtract_asymptotic = true` is not implemented.
"""
function SDE_channel_L_ph!(
    Lph   :: NL2_MF_K2{Q},
    Πph   :: NL2_MF_Π{Q},
    F     :: AbstractVertex{Q},
    SGph2 :: SymmetryGroup
    ;
    mode  :: Symbol,
    subtract_asymptotic :: Bool = false,
    )     :: Nothing where {Q}

    subtract_asymptotic && error("Not implemented")

    # Lph(Ω, ν; q, k) for one point: sum over ν̃ (called ω) and k̃ (called q)
    @inline function diagram(wtpl)

        Ω, ν, P, k = wtpl
        val     = zero(Q)
        Πslice  = view(Πph, Ω, :, P, :)

        for i in eachindex(Πslice)
            ω = value(meshes(Πph, Val(2))[i.I[1]])
            q = value(meshes(Πph, Val(4))[i.I[2]])

            # own-channel form, as in NL:
            # val += bare_vertex(F) * Πslice[i] * (F.γt(Ω, ν, ω, P, k, q) + F.γa(Ω, ν, ω, P, k, q))
            # U Π_a [Φ_a + Φ_t](Ω, ν, ν̃; q, k, k̃), each as F - F.F0 (one channel only)
            val += bare_vertex(F) * Πslice[i] * (
              +    F(Ω, ν, ω, P, k, q, aCh, pSp; γa = true, γp = false, γt = false)
              +    F(Ω, ν, ω, P, k, q, tCh, pSp; γt = true, γp = false, γa = false)
              - F.F0(Ω, ν, ω, P, k, q, aCh, pSp; γa = true, γp = false, γt = false)
              - F.F0(Ω, ν, ω, P, k, q, tCh, pSp; γt = true, γp = false, γa = false) )
        end

        # 1/β 1/N_k ∑_{ν̃,k̃}
        return temperature(F) * val / length(meshes(Πph, Val(4)))
    end

    # evaluate one point per symmetry class; the symmetry group fills in the rest
    SGph2(Lph, InitFunction{4, Q}(diagram); mode)

    return nothing
end



# RefVertex methods: U Π (F_ref - U). A RefVertex has no momentum dependence.

# RefVertex method of SDE_channel_L_pp! (NL2), see the docstring above.
function SDE_channel_L_pp!(
    Lpp   :: NL2_MF_K2{Q},
    Πpp   :: NL2_MF_Π{Q},
    F     :: RefVertex{Q},
    SGpp2 :: SymmetryGroup
    ;
    mode  :: Symbol,
    subtract_asymptotic :: Bool = false,
    )     :: Nothing where {Q}

    # Lpp(Ω, ν; q, k) for one point: sum over ν̃ (called ω) and k̃ (called q)
    @inline function diagram(wtpl)

        Ω, ν, P, k = wtpl
        val     = zero(Q)
        Πslice  = view(Πpp, Ω, :, P, :)

        for i in eachindex(Πslice)
            ω = value(meshes(Πpp, Val(2))[i.I[1]])
            q = value(meshes(Πpp, Val(4))[i.I[2]])

            # RefVertex has no momentum dependence
            val += bare_vertex(F) * Πpp[Ω, ω, P, q] * (F(Ω, Ω - ω, ν, pCh, pSp) - bare_vertex(F, pCh, pSp))
        end

        return temperature(F) * val / length(meshes(Πpp, Val(4)))
    end

    # evaluate one point per symmetry class; the symmetry group fills in the rest
    SGpp2(Lpp, InitFunction{4, Q}(diagram); mode)

    return nothing
end

# RefVertex method of SDE_channel_L_ph! (NL2), see the docstring above.
function SDE_channel_L_ph!(
    Lph   :: NL2_MF_K2{Q},
    Πph   :: NL2_MF_Π{Q},
    F     :: RefVertex{Q},
    SGph2 :: SymmetryGroup
    ;
    mode  :: Symbol,
    subtract_asymptotic :: Bool = false,
    )     :: Nothing where {Q}

    # Lph(Ω, ν; q, k) for one point: sum over ν̃ (called ω) and k̃ (called q)
    @inline function diagram(wtpl)

        Ω, ν, P, k = wtpl
        val     = zero(Q)
        Πslice  = view(Πph, Ω, :, P, :)

        for i in eachindex(Πslice)
            ω = value(meshes(Πph, Val(2))[i.I[1]])
            q = value(meshes(Πph, Val(4))[i.I[2]])

            # RefVertex has no momentum dependence
            val += bare_vertex(F) * Πph[Ω, ω, P, q] * (F(Ω, ν, ω, aCh, pSp) + F(Ω, ν, ω, tCh, pSp) - bare_vertex(F, aCh, pSp) - bare_vertex(F, tCh, pSp))
        end

        return temperature(F) * val / length(meshes(Πph, Val(4)))
    end

    # evaluate one point per symmetry class; the symmetry group fills in the rest
    SGph2(Lph, InitFunction{4, Q}(diagram); mode)

    return nothing
end

# -----------------------------------------------------------------------------
# Self-energy of one vertex level
# -----------------------------------------------------------------------------

"""
    SDE_compute!(Σ, G, Πpp::NL2_MF_Π, Πph::NL2_MF_Π, Lpp, Lph, F, SGΣ, SGpp2, SGph2; mode, use_real_space = true, include_U² = true, include_Hartree = true, subtract_asymptotic = false) -> Σ

NL2 version of `SDE_compute!` (SDE.jl): the self-energy contribution of one
vertex level `F`, written into `Σ` (stored as iΣ). It builds the kernels
`Lpp(Ω, ν; q, k)`, `Lph(Ω, ν; q, k)` and closes them with one propagator (see
the file header):

- `use_real_space = true`: as a convolution in real space. G(R) comes from the
  fine G mesh, the kernels L(R, R′) from the coarse vertex mesh; R and R′ run
  over [-L/2, L/2]², with half weight on the boundary shell of an even L, and
  the result is transformed back to the momentum mesh of Σ.
- otherwise: the direct momentum sum, which needs the same momentum mesh for G,
  Σ and the kernels.

A RefVertex contribution is divided by 3 (its three channel forms are the same
vertex). Optionally `Σ_U²` and the Hartree term `U (n - 1/2)` are added.
"""
function SDE_compute!(
    Σ     :: NL_MF_G{Q},
    G     :: NL_MF_G{Q},
    Πpp   :: NL2_MF_Π{Q},
    Πph   :: NL2_MF_Π{Q},
    Lpp   :: NL2_MF_K2{Q},
    Lph   :: NL2_MF_K2{Q},
    F     :: Union{AbstractVertex{Q}, RefVertex{Q}},
    SGΣ   :: SymmetryGroup,
    SGpp2 :: SymmetryGroup,
    SGph2 :: SymmetryGroup,
    ;
    mode  :: Symbol,
    use_real_space :: Bool = true,
    include_U² = true,
    include_Hartree = true,
    subtract_asymptotic = false,
    )     :: NL_MF_G{Q} where {Q}
    # contribution of this level's channels Φ_a, Φ_p, Φ_t (see the NOTE above)

    SDE_channel_L_pp!(Lpp, Πpp, F, SGpp2; mode, subtract_asymptotic)
    SDE_channel_L_ph!(Lph, Πph, F, SGph2; mode, subtract_asymptotic)

    T = temperature(meshes(G, Val(1)))

    if use_real_space
        # Real-space evaluation
        # Σ( R + R') += G(-R) * Lpp(R, R')
        # Σ(-R + R') += G(-R) * Lph(R, R')

        # linear sizes of the momentum meshes of G, Σ and the kernels
        LG = bz(meshes(G, Val(2))).L
        LΣ = bz(meshes(Σ, Val(2))).L
        L  = bz(meshes(Lpp, Val(3))).L

        n1 = length(meshes(Lpp, Val(1)))
        n2 = length(meshes(Lpp, Val(2)))

        # G(R) and L(R, R′) by FFT over the momentum directions. fft! works in
        # place, so afterwards Lpp and Lph hold the real-space kernels (times
        # L^4), not the momentum-space ones.
        G_data_R = fft(reshape(G.data, :, LG, LG), (2, 3)) / LG^2
        Σ_data_R = zeros(eltype(Σ.data), length(meshes(Σ, Val(1))), LΣ, LΣ)

        Lpp_data_R = fft!(Base.ReshapedArray(Lpp.data, (n1, n2, L, L, L, L), ()), (3, 4, 5, 6)) ./ L^4
        Lph_data_R = fft!(Base.ReshapedArray(Lph.data, (n1, n2, L, L, L, L), ()), (3, 4, 5, 6)) ./ L^4

        # distances R and R′ in [-L/2, L/2]²
        Rs_L_1d = (-div(L, 2)) : div(L, 2)
        Rs = collect(Iterators.product(Rs_L_1d, Rs_L_1d))

        for (Rp1, Rp2) in Rs
            Rp_vec = (Rp1, Rp2)

            # Index of Rp in L(R, Rp)
            iRp_L = mod.(Rp_vec, (L, L)) .+ 1

            for (R1, R2) in Rs
                R_vec = (R1, R2)

                # Half weight per coordinate on the boundary shell ±L/2: L(R, R′)
                # is periodic in L, so both points read the same kernel value,
                # which is split evenly between them.
                weight = 1.0
                if mod(L, 2) == 0
                    abs(Rp1) == div(L, 2) && (weight /= 2)
                    abs(Rp2) == div(L, 2) && (weight /= 2)
                    abs(R1) == div(L, 2) && (weight /= 2)
                    abs(R2) == div(L, 2) && (weight /= 2)
                end

                # Index of R in L(R, Rp)
                iR_L  = mod.(R_vec, (L, L)) .+ 1

                # Index of -R in G
                imR_G = mod.(.-R_vec, (LG, LG)) .+ 1

                # Index of R + Rp and -R + Rp in Σ
                iRpR_Σ = mod.(   R_vec .+ Rp_vec, (LΣ, LΣ)) .+ 1
                iRmR_Σ = mod.(.- R_vec .+ Rp_vec, (LΣ, LΣ)) .+ 1

                Lpp_R = MeshFunction((meshes(Lpp, Val(1)), meshes(Lpp, Val(2))), view(Lpp_data_R, :, :, iR_L..., iRp_L...))
                Lph_R = MeshFunction((meshes(Lph, Val(1)), meshes(Lph, Val(2))), view(Lph_data_R, :, :, iR_L..., iRp_L...))

                G_R = MeshFunction((meshes(G, Val(1)),), view(G_data_R, :, imR_G...))
                Σ_R_pp = MeshFunction((meshes(Σ, Val(1)),), view(Σ_data_R, :, iRpR_Σ...))
                Σ_R_ph = MeshFunction((meshes(Σ, Val(1)),), view(Σ_data_R, :, iRmR_Σ...))

                for iν in eachindex(meshes(Lpp, Val(2)))
                    ν = value(meshes(Lpp, Val(2))[iν])

                    if is_inbounds(ν, meshes(Σ, Val(1)))

                        for iΩ in eachindex(meshes(Lpp, Val(1)))
                            Ω = value(meshes(Lpp, Val(1))[iΩ])

                            # p channel closed with G(Ω - ν), a and t with
                            # G(Ω + ν); G_R(…) is zero outside the G mesh
                            Σ_R_pp[ν] += G_R(Ω - ν) * Lpp_R[iΩ, iν] * weight
                            Σ_R_ph[ν] += G_R(Ω + ν) * Lph_R[iΩ, iν] * weight
                        end

                    end
                end

            end
        end

        Σ_data_R .*= T

        # back to momentum space
        Σ.data .= reshape(bfft(Σ_data_R, (2, 3)), :, LΣ^2)

        # impose the symmetries of Σ
        SGΣ(Σ)

    else
        # Momentum space evaluation
        # Σ(k) += G(P - k) * Lpp(P, k)
        # Σ(k) += G(P + k) * Lph(P, k)

        # It is required that Σ and L has the same mesh.
        # (One may use linear interpolation, but that is not implemented here.)
        mG = meshes(G, Val(2))
        @assert mG == meshes(Lpp, Val(4)) == meshes(Lph, Val(4))

        # iΣ(ν, k) for one point: sum over the bosonic frequency and momentum
        @inline function diagram(wtpl)

            ν, k = wtpl
            val = zero(Q)

            for iP in eachindex(meshes(Lpp, Val(3)))

                P = value(meshes(Lpp, Val(3))[iP])

                if is_inbounds(ν, meshes(Lpp, Val(2)))
                    Lppslice = view(Lpp, :, ν, P, k)
                    Lphslice = view(Lph, :, ν, P, k)

                    for i in eachindex(Lppslice)
                        Ω = value(meshes(Lpp, Val(1))[i])

                        if is_inbounds(Ω - ν, meshes(G, Val(1)))
                            val += G[Ω - ν, fold_back(P - k, mG)] * Lppslice[i]
                        end

                        if is_inbounds(Ω + ν, meshes(G, Val(1)))
                            val += G[Ω + ν, fold_back(P + k, mG)] * Lphslice[i]
                        end
                    end
                end

            end

            # 1/β 1/N_q ∑_{Ω,q}
            return T * val / length(meshes(Lpp, 3))
        end

        # evaluate one point per symmetry class; the symmetry group fills in the rest
        SGΣ(Σ, InitFunction{2, Q}(diagram); mode)

    end

    if F isa RefVertex
        # If F is a RefVertex, a, p, and t channel contributions are all equivalent and
        # counted three times. We divide by 3 to get the correct result.
        Σ.data .*= 1/3
    end

    if include_U²
        # second-order diagram SDE[U], Eq. (S46)
        Σ_U² = SDE_U2_using_G(Σ, G, SGΣ, bare_vertex(F); mode)
        # Σ_U² = SDE_U2_using_Π(Σ, G, Πpp, Πph, SGΣ, bare_vertex(F); mode)
        add!(Σ, Σ_U²)
    end

    if include_Hartree
        # Hartree term U (n - 1/2), relative to half filling
        n = compute_occupation(G)
        # We store im * Σ in S.Σ, so we multiply im.
        Σ.data .+= Q((n - 1/2) * bare_vertex(F) * im)
    end

    return Σ
end
