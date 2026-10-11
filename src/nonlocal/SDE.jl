# =============================================================================
# Schwinger-Dyson equation (SDE) for the s-wave lattice solver NL.
#
# The physics is that of the local SDE (SDE.jl; Lihm 2025, Eqs. (S34)-(S38)),
# now with momenta (Eq. (S36)):
#
#   Σ(ν, k) = 1/β 1/N_q ∑_{Ω,q} [ G(Ω - ν, q - k) Lpp(Ω, ν; q)
#                               + G(Ω + ν, q + k) Lph(Ω, ν; q) ]
#             + Σ_U² + Σ_H
#
# with the kernels of one vertex level F (s-wave, so they keep only the bosonic
# momentum q):
#
#   Lpp(Ω, ν; q) = U · 1/β ∑_ν̃ Π_p(Ω, ν̃; q) Φ_p(Ω, Ω - ν̃, ν; q)
#   Lph(Ω, ν; q) = U · 1/β ∑_ν̃ Π_a(Ω, ν̃; q) [Φ_a + Φ_t](Ω, ν, ν̃; q)
#
# Π are the k-averaged bubbles of NL (nonlocal/bubble.jl). The sum over q is a
# convolution in momentum, so it is done in real space with FFTs. As in the
# local code, SDE! (SDE.jl) applies this to every level of the nested vertex
# F → F.F0 → …, and adds Σ_U² and the Hartree term once.
#
# The kernels use only the level's own channels (`F.γp`, `F.γt`, `F.γa`), so
# each reducible vertex is counted exactly once (compare open question (a) in
# docs/physics_and_notation.md for NL2).
#
# Symbols as in SDE.jl: Φ_r reducible vertex of channel r (fields `γa`, `γp`,
# `γt`), Σ_H = U(n - 1/2) Hartree term, Σ_U² second-order term (Eq. (S46)).
# Paper ↔ code: q ↔ P; ν̃ (summed) ↔ ω; G and Σ stored as iG and iΣ.
# =============================================================================

# -----------------------------------------------------------------------------
# Channel kernels L
# -----------------------------------------------------------------------------

"""
    SDE_channel_L_pp!(Lpp, Πpp, F::Union{Vertex, NL_Vertex}, SGpp2; mode, subtract_asymptotic = false) -> nothing

Lattice (NL) version of the p-channel SDE kernel (see the local method in
SDE.jl): `Lpp(Ω, ν; q) = U · 1/β ∑_ν̃ Π_p(Ω, ν̃; q) Φ_p(Ω, Ω - ν̃, ν; q)`, with `Φ_p`
the vertex level's own p channel. `F` may be a local `Vertex` (e.g. the DMFT
reference, which ignores q).

`subtract_asymptotic = true` leaves out the part of `Φ_p` that does not depend on
ν (its ν → ∞ limit). That part was meant to be computed separately as a
GW-type self-energy, which is not done anywhere in the package. The option is
not used.
"""
function SDE_channel_L_pp!(
    Lpp   :: NL_MF_K2{Q},
    Πpp   :: NL_MF_Π{Q},
    F     :: Union{Vertex{Q}, NL_Vertex{Q}},
    SGpp2 :: SymmetryGroup
    ;
    mode  :: Symbol,
    subtract_asymptotic :: Bool = false,
    )     :: Nothing where {Q}

    # Lpp(Ω, ν; q) for one point: sum over the internal frequency ν̃ (called ω)
    @inline function diagram(wtpl)

        Ω, ν, P = wtpl
        val     = zero(Q)
        Πslice  = view(Πpp, Ω, :, P)

        for i in eachindex(Πslice)
            ω = value(meshes(Πpp, Val(2))[i])

            # U Π_p Φ_p(Ω, Ω - ν̃, ν; q), the level's own p channel only
            val += bare_vertex(F) * Πslice[i] * F.γp(Ω, Ω - ω, ν, P)

            if subtract_asymptotic
                # Subtract the asymptotic contribution ν → νInf.
                # This term is computed as GW self-energy.
                val -= bare_vertex(F) * Πslice[i] * F.γp(Ω, Ω - ω, νInf, P)
            end
        end

        return temperature(F) * val
    end

    # evaluate one point per symmetry class; the symmetry group fills in the rest
    SGpp2(Lpp, InitFunction{3, Q}(diagram); mode)

    return nothing
end

"""
    SDE_channel_L_ph!(Lph, Πph, F::Union{Vertex, NL_Vertex}, SGph2; mode, subtract_asymptotic = false) -> nothing

Lattice (NL) version of the particle-hole SDE kernel (see the local method in
SDE.jl): `Lph(Ω, ν; q) = U · 1/β ∑_ν̃ Π_a(Ω, ν̃; q) [Φ_a + Φ_t](Ω, ν, ν̃; q)`, with
the level's own a and t channels. `subtract_asymptotic` as in the p-channel
kernel.
"""
function SDE_channel_L_ph!(
    Lph   :: NL_MF_K2{Q},
    Πph   :: NL_MF_Π{Q},
    F     :: Union{Vertex{Q}, NL_Vertex{Q}},
    SGph2 :: SymmetryGroup
    ;
    mode  :: Symbol,
    subtract_asymptotic :: Bool = false,
    )     :: Nothing where {Q}

    # Lph(Ω, ν; q) for one point: sum over the internal frequency ν̃ (called ω)
    @inline function diagram(wtpl)

        Ω, ν, P = wtpl
        val     = zero(Q)
        Πslice  = view(Πph, Ω, :, P)

        for i in eachindex(Πslice)
            ω = value(meshes(Πph, Val(2))[i])

            # U Π_a [Φ_t + Φ_a](Ω, ν, ν̃; q), the level's own channels only
            val += bare_vertex(F) * Πslice[i] * (F.γt(Ω, ν, ω, P) + F.γa(Ω, ν, ω, P))

            if subtract_asymptotic
                # Subtract the asymptotic contribution ν → νInf.
                # This term is computed as GW self-energy.
                val -= bare_vertex(F) * Πslice[i] * (F.γt(Ω, νInf, ω, P) + F.γa(Ω, νInf, ω, P))
            end
        end

        return temperature(F) * val
    end

    # evaluate one point per symmetry class; the symmetry group fills in the rest
    SGph2(Lph, InitFunction{3, Q}(diagram); mode)

    return nothing
end

# RefVertex methods: U Π (F_ref - U). A RefVertex has no momentum dependence.

# RefVertex method of SDE_channel_L_pp! (NL), see SDE.jl.
function SDE_channel_L_pp!(
    Lpp   :: NL_MF_K2{Q},
    Πpp   :: NL_MF_Π{Q},
    F     :: RefVertex{Q},
    SGpp2 :: SymmetryGroup
    ;
    mode  :: Symbol,
    subtract_asymptotic :: Bool = false,
    )     :: Nothing where {Q}

    # Lpp(Ω, ν; q) for one point: sum over the internal frequency ν̃ (called ω)
    @inline function diagram(wtpl)

        Ω, ν, P = wtpl
        val     = zero(Q)

        for iω in eachindex(meshes(Πpp, Val(2)))
            ω = value(meshes(Πpp, Val(2))[iω])

            # RefVertex has no momentum dependence
            val += bare_vertex(F) * Πpp[Ω, ω, P] * (F(Ω, Ω - ω, ν, pCh, pSp) - bare_vertex(F, pCh, pSp))
        end

        return temperature(F) * val
    end

    # evaluate one point per symmetry class; the symmetry group fills in the rest
    SGpp2(Lpp, InitFunction{3, Q}(diagram); mode)

    return nothing
end

# RefVertex method of SDE_channel_L_ph! (NL), see SDE.jl.
function SDE_channel_L_ph!(
    Lph   :: NL_MF_K2{Q},
    Πph   :: NL_MF_Π{Q},
    F     :: RefVertex{Q},
    SGph2 :: SymmetryGroup
    ;
    mode  :: Symbol,
    subtract_asymptotic :: Bool = false,
    )     :: Nothing where {Q}

    # Lph(Ω, ν; q) for one point: sum over the internal frequency ν̃ (called ω)
    @inline function diagram(wtpl)

        Ω, ν, P = wtpl
        val     = zero(Q)

        for iω in eachindex(meshes(Πph, Val(2)))
            ω = value(meshes(Πph, Val(2))[iω])

            # RefVertex has no momentum dependence
            val += bare_vertex(F) * Πph[Ω, ω, P] * (F(Ω, ν, ω, aCh, pSp) + F(Ω, ν, ω, tCh, pSp) - bare_vertex(F, aCh, pSp) - bare_vertex(F, tCh, pSp))
        end

        return temperature(F) * val
    end

    # evaluate one point per symmetry class; the symmetry group fills in the rest
    SGph2(Lph, InitFunction{3, Q}(diagram); mode)

    return nothing
end

# -----------------------------------------------------------------------------
# Self-energy of one vertex level
# -----------------------------------------------------------------------------

"""
    SDE_compute!(Σ, G, Πpp, Πph, Lpp, Lph, F, SGΣ, SGpp2, SGph2; mode = :threads, use_real_space = true, include_U² = true, include_Hartree = true, subtract_asymptotic = false) -> Σ

Lattice (NL) version of `SDE_compute!` (SDE.jl): the self-energy contribution
of one vertex level `F`, written into `Σ` (stored as iΣ). It builds the kernels
`Lpp`, `Lph`, closes them with one propagator (`SDE_compute_inner!`), divides a
RefVertex contribution by 3 (its three channel forms are the same vertex), and
optionally adds `Σ_U²` and the Hartree term `U (n - 1/2)`.
"""
function SDE_compute!(
    Σ     :: NL_MF_G{Q},
    G     :: NL_MF_G{Q},
    Πpp   :: NL_MF_Π{Q},
    Πph   :: NL_MF_Π{Q},
    Lpp   :: NL_MF_K2{Q},
    Lph   :: NL_MF_K2{Q},
    F,#     :: Union{NL_Vertex{Q}, Vertex{Q}, RefVertex{Q}},
    SGΣ   :: SymmetryGroup,
    SGpp2 :: SymmetryGroup,
    SGph2 :: SymmetryGroup,
    ;
    mode  :: Symbol = :threads,
    use_real_space :: Bool = true,
    include_U² = true,
    include_Hartree = true,
    subtract_asymptotic = false,
    )     :: NL_MF_G{Q} where {Q}
    # contribution of this level's own channels Φ_a, Φ_p, Φ_t

    SDE_channel_L_pp!(Lpp, Πpp, F, SGpp2; mode, subtract_asymptotic)
    SDE_channel_L_ph!(Lph, Πph, F, SGph2; mode, subtract_asymptotic)

    SDE_compute_inner!(Σ, G, Lpp, Lph, SGΣ; mode, use_real_space)

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
        # Hartree term U (n - 1/2); n is the density per spin
        n = compute_occupation(G)
        # We store im * Σ in S.Σ, so we multiply im.
        Σ.data .+= Q((n - 1/2) * bare_vertex(F) * im)
    end

    return Σ
end;

"""
    SDE_compute_inner!(Σ, G, Lpp, Lph, SGΣ; mode = :threads, use_real_space = true) -> Σ

Close the SDE kernels with one propagator and write the result into `Σ`:

    iΣ(ν, k) = 1/β 1/N_q ∑_{Ω,q} [ iG(Ω - ν, q - k) Lpp(Ω, ν; q) + iG(Ω + ν, q + k) Lph(Ω, ν; q) ]

With `use_real_space = true` the sum over q, a convolution, is done in real
space. G(R) is taken from the fine mesh of G, the kernels L(R) from the coarse
vertex mesh. Distances R in [-L/2, L/2]² are summed, with half weight on the
boundary shell of an even L, and the result is transformed back to the
momentum mesh of Σ. Otherwise the momentum sum is done directly (Σ and the
kernels then need the same momentum mesh). Ω runs over the K2 box of the
kernels, and ν over the frequencies of Σ inside that box.
"""
function SDE_compute_inner!(
    Σ     :: NL_MF_G{Q},
    G     :: NL_MF_G{Q},
    Lpp   :: NL_MF_K2{Q},
    Lph   :: NL_MF_K2{Q},
    SGΣ   :: SymmetryGroup,
    ;
    mode  :: Symbol = :threads,
    use_real_space :: Bool = true,
    )     :: NL_MF_G{Q} where {Q}

    meshK_G = meshes(G, Val(2))
    T = temperature(meshes(G, Val(1)))

    set!(Σ, 0)

    if use_real_space
        # Real-space evaluation
        # Σ( R) += G(-R) * Lpp(R)
        # Σ(-R) += G(-R) * Lph(R)

        # linear sizes of the momentum meshes of G, Σ and the kernels
        LG = bz(meshes(G, Val(2))).L
        LΣ = bz(meshes(Σ, Val(2))).L
        L  = bz(meshes(Lpp, Val(3))).L

        n1 = length(meshes(Lpp, Val(1)))
        n2 = length(meshes(Lpp, Val(2)))

        # G(R) and L(R) by FFT over the momentum directions
        G_data_R = fft(reshape(G.data, :, LG, LG), (2, 3)) / LG^2
        Σ_data_R = zeros(eltype(Σ.data), length(meshes(Σ, Val(1))), LΣ, LΣ)

        Lpp_data_R = fft(reshape(Lpp.data, n1, n2, L, L), (3, 4)) / L^2
        Lph_data_R = fft(reshape(Lph.data, n1, n2, L, L), (3, 4)) / L^2

        # distances R in [-L/2, L/2]²
        Rs_L_1d = (-div(L, 2)) : div(L, 2)
        Rs = collect(Iterators.product(Rs_L_1d, Rs_L_1d))

        for (R1, R2) in Rs
            R_vec = (R1, R2)

            # Half weight on the boundary shell R = ±L/2: L(R) is periodic in L,
            # so both points read the same kernel value, which is split
            # evenly between them.
            weight = 1.0
            if mod(L, 2) == 0
                abs(R1) == div(L, 2) && (weight /= 2)
                abs(R2) == div(L, 2) && (weight /= 2)
            end

            # Index of R in L(R, Rp)
            iR_L  = mod.(R_vec, (L, L)) .+ 1

            # Index of -R in G
            imR_G = mod.(.-R_vec, (LG, LG)) .+ 1

            # Index of R and -R in Σ
            ipR_Σ = mod.(  R_vec, (LΣ, LΣ)) .+ 1
            imR_Σ = mod.(.-R_vec, (LΣ, LΣ)) .+ 1

            Lpp_R = MeshFunction((meshes(Lpp, Val(1)), meshes(Lpp, Val(2))), view(Lpp_data_R, :, :, iR_L...))
            Lph_R = MeshFunction((meshes(Lph, Val(1)), meshes(Lph, Val(2))), view(Lph_data_R, :, :, iR_L...))

            G_R = MeshFunction((meshes(G, Val(1)),), view(G_data_R, :, imR_G...))
            Σ_R_pp = MeshFunction((meshes(Σ, Val(1)),), view(Σ_data_R, :, ipR_Σ...))
            Σ_R_ph = MeshFunction((meshes(Σ, Val(1)),), view(Σ_data_R, :, imR_Σ...))

            for iν in eachindex(meshes(Lpp, Val(2)))
                ν = value(meshes(Lpp, Val(2))[iν])

                if is_inbounds(ν, meshes(Σ, Val(1)))

                    for iΩ in eachindex(meshes(Lpp, Val(1)))
                        Ω = value(meshes(Lpp, Val(1))[iΩ])

                        # p channel closed with G(Ω - ν), a and t with G(Ω + ν)
                        if is_inbounds(Ω - ν, meshes(G, Val(1)))
                            Σ_R_pp[ν] += G_R[Ω - ν] * Lpp_R[iΩ, iν] * weight
                        end
                        if is_inbounds(Ω + ν, meshes(G, Val(1)))
                            Σ_R_ph[ν] += G_R[Ω + ν] * Lph_R[iΩ, iν] * weight
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
        # Σ(k) += G(P - k) * Lpp(P)
        # Σ(k) += G(P + k) * Lph(P)

        # iΣ(ν, k) for one point: sum over the bosonic frequency and momentum
        @inline function diagram(wtpl)

            ν, k = wtpl
            val = zero(Q)

            k_vec = euclidean(k, meshK_G)

            for iP in eachindex(meshes(Lpp, Val(3)))

                P = value(meshes(Lpp, Val(3))[iP])
                P_vec = euclidean(P, meshes(Lpp, Val(3)))

                Pmk_G = meshK_G[MatsubaraFunctions.mesh_index(P_vec - k_vec, meshK_G)]
                Ppk_G = meshK_G[MatsubaraFunctions.mesh_index(P_vec + k_vec, meshK_G)]

                if is_inbounds(ν, meshes(Lpp, Val(2)))
                    Lppslice = view(Lpp, :, ν, P)
                    Lphslice = view(Lph, :, ν, P)

                    for i in eachindex(Lppslice)
                        Ω = value(meshes(Lpp, Val(1))[i])

                        if is_inbounds(Ω - ν, meshes(G, Val(1)))
                            val += G[Ω - ν, Pmk_G] * Lppslice[i]
                        end

                        if is_inbounds(Ω + ν, meshes(G, Val(1)))
                            val += G[Ω + ν, Ppk_G] * Lphslice[i]
                        end
                    end
                end

            end

            return T * val / length(meshes(Lpp, Val(3)))
        end

        # compute Σ
        SGΣ(Σ, InitFunction{2, Q}(diagram); mode)
    end

    return Σ
end

# -----------------------------------------------------------------------------
# Second-order self-energy
# -----------------------------------------------------------------------------

"""
    SDE_U2_using_Π(Σ::NL_MF_G, G, Πpp, Πph, SGΣ, U; mode) -> Σ_U²

Lattice version of the second-order self-energy built from the bubbles (see
the local method in SDE.jl). NOTE: it sums over a fourth (fermionic momentum)
dimension of the bubbles, but the NL bubbles `NL_MF_Π` have only three
dimensions, so this does not work. It is not used by the solvers.
"""
function SDE_U2_using_Π(
    Σ   :: NL_MF_G{Q},
    G   :: NL_MF_G{Q},
    Πpp :: NL_MF_Π{Q},
    Πph :: NL_MF_Π{Q},
    SGΣ :: SymmetryGroup,
    U   :: Number,
    ;
    mode :: Symbol,
    )   :: NL_MF_G{Q} where {Q}
    # 2nd-order perturbative contribution to the self-energy

    T = temperature(meshes(Σ, Val(1)))
    Σ_U² = copy(Σ)
    set!(Σ_U², 0)

    # Integrate fermionic frequency and momentum for the bubble

    Πppsum = MeshFunction((meshes(Πpp, Val(1)), meshes(Πpp, Val(3)),),
        dropdims(sum(Πpp.data, dims=(2,4)), dims=(2,4)) .* (T / length(meshes(Πpp, Val(4)))))
    Πphsum = MeshFunction((meshes(Πph, Val(1)), meshes(Πph, Val(3)),),
        dropdims(sum(Πph.data, dims=(2,4)), dims=(2,4)) .* (T / length(meshes(Πph, Val(4)))))

    meshK_G = meshes(G, Val(2))

    # iΣ_U²(ν, k) for one point
    @inline function diagram(wtpl)

        ν, k = wtpl
        val  = zero(Q)

        k_vec = euclidean(k, meshK_G)

        for iP in eachindex(meshes(Πppsum, Val(2)))

            P = value(meshes(Πppsum, Val(2))[iP])
            P_vec = euclidean(P, meshes(Πppsum, Val(2)))

            Pmk_G = meshK_G[MatsubaraFunctions.mesh_index(P_vec - k_vec, meshK_G)]
            Ppk_G = meshK_G[MatsubaraFunctions.mesh_index(P_vec + k_vec, meshK_G)]

            for iΩ in eachindex(meshes(Πppsum, Val(1)))

                Ω = value(meshes(Πppsum, Val(1))[iΩ])

                if is_inbounds(Ω - ν, meshes(G, Val(1)))
                    val += G[Ω - ν, Pmk_G] * Πppsum[iΩ, iP]
                end

                if is_inbounds(Ω + ν, meshes(G, Val(1)))
                    val += G[Ω + ν, Ppk_G] * Πphsum[iΩ, iP]
                end

            end

        end

        return val * U^2 * T / 2 / length(meshes(Πph, Val(3)))
    end

    # compute Σ
    SGΣ(Σ_U², InitFunction{2, Q}(diagram); mode)

    return Σ_U²
end


"""
    SDE_U2_using_G(Σ::NL_MF_G, G, SGΣ, U; mode = :threads) -> Σ_U²

Lattice version of the second-order self-energy (Lihm 2025, Eq. (S46)). In real
space the diagram is a product at each distance R:

    iΣ_U²(ν, R) = U² · 1/β² ∑_{ν1,ν2} iG(ν1, -R) iG(ν2, R) iG(ν1 - ν2 + ν, R)

It is computed for all R of the fine mesh of G and transformed back to
momentum space.
"""
function SDE_U2_using_G(
    Σ   :: NL_MF_G{Q},
    G   :: NL_MF_G{Q},
    SGΣ :: SymmetryGroup,
    U   :: Number,
    ;
    mode  :: Symbol = :threads,
    )   :: NL_MF_G{Q} where {Q}

    # Σ_U²(ν, R) = U² * T² * ∑_{ω1, ω2, ν} G(ω1, -R) * G(ω2, R) * G(ω1 - ω2 + ν, R)

    T = temperature(meshes(Σ, Val(1)))
    Σ_U² = copy(Σ)
    set!(Σ_U², 0)

    # G(R) and G(-R) from the two FFT directions
    L = bz(meshes(G, Val(2))).L
    G_pR =  fft(reshape(G.data, :, L, L), (2, 3)) / L^2
    G_mR = bfft(reshape(G.data, :, L, L), (2, 3)) / L^2

    Σ_U²_R = zeros(eltype(Σ.data), length(meshes(Σ, Val(1))), L, L)

    Threads.@threads for (iR1, iR2) in collect(Iterators.product(axes(G_pR, 2), axes(G_pR, 3)))
        Gp = MeshFunction((meshes(G, Val(1)),), view(G_pR, :, iR1, iR2))
        Gm = MeshFunction((meshes(G, Val(1)),), view(G_mR, :, iR1, iR2))

        for i2 in eachindex(meshes(G, Val(1))), i1 in eachindex(meshes(G, Val(1)))
            ω1 = value(meshes(G, Val(1))[i1])
            ω2 = value(meshes(G, Val(1))[i2])
            gg = Gm[ω1] * Gp[ω2]

            for iν in eachindex(meshes(Σ, Val(1)))
                ν = value(meshes(Σ, Val(1))[iν])

                # the third propagator must lie on the G mesh too
                if is_inbounds(ω1 - ω2 + ν, meshes(G, Val(1)))
                    Σ_U²_R[iν, iR1, iR2] += gg * Gp[ω1 - ω2 + ν]
                end
            end
        end
    end

    Σ_U²_R .*= U^2 * T^2

    # back to momentum space, then impose the symmetries
    Σ_U².data .= reshape(bfft(Σ_U²_R, (2, 3)), :, L^2)

    SGΣ(Σ_U²)

    Σ_U²
end;

# -----------------------------------------------------------------------------
# Diagnostics
# -----------------------------------------------------------------------------

"""
    self_energy_sanity_check(Σ::NL_MF_G) -> Bool

Causality check of a lattice self-energy at every momentum: Im Σ(iν, k) ≤ 0 for
ν > 0 and ≥ 0 for ν < 0, i.e. the stored `real(iΣ)` ≥ 0 for ν > 0 and ≤ 0 for
ν < 0. Warns at every violation and returns whether all passed.
"""
function self_energy_sanity_check(Σ :: NL_MF_G)
    passed = true
    # sanity check
    for k in meshes(Σ, Val(2)), ν in meshes(Σ, Val(1))
        if plain_value(ν) > 0 && real(Σ[ν, k]) < 0
            passed = false
            @warn "Σ violates causality at (n, k) = $(index(ν)), $(index(k))"
        elseif plain_value(ν) < 0 && real(Σ[ν, k]) > 0
            passed = false
            @warn "Σ violates causality at (n, k) = $(index(ν)), $(index(k))"
        end
    end
    passed
end
