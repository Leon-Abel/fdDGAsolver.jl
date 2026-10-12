# =============================================================================
# Lattice bubbles of the solver NL2, which keep the fermionic momentum.
#
# NL2 does not average the bubble over the loop momentum k. It stores the
# product of the two propagators at every bosonic momentum q (code: P) and
# fermionic momentum k:
#
#   Πpp(Ω, ν; q, k) = G(ν, k) G(Ω - ν, q - k)    (p channel)
#   Πph(Ω, ν; q, k) = G(ν, k) G(Ω + ν, q + k)    (a and t channels)
#
# The sum over the loop momentum then happens in the BSE kernels, together
# with the k-dependent K2 of the vertex (nonlocal_2/BSEa/).
#
# The solver uses bubbles_real_space!. G is transformed to real space on the
# fine mesh (L_G × L_G), only distances in the coarse box [-L/2, L/2]² are
# kept (as in the NL bubble, nonlocal/bubble.jl), and the products are
# transformed back to the coarse vertex mesh (L × L) that q and k live on.
# The direct product at the mesh momenta (bubbles_momentum_space!) differs
# from it only by the part of G at distances beyond L/2. For L = L_G the two
# agree, which test_hubbard.jl checks.
#
# As everywhere, G is stored as iG, so the stored bubble is (iG)(iG) = -GG.
# =============================================================================

"""
    bubbles_momentum_space!(Πpp::NL2_MF_Π, Πph::NL2_MF_Π, G) -> nothing

NL2 bubbles from the direct product,
`Πpp(Ω, ν; q, k) = G(ν, k) G(Ω - ν, q - k)` and
`Πph(Ω, ν; q, k) = G(ν, k) G(Ω + ν, q + k)`, with q and k from the coarse bubble
mesh looked up on the G mesh. Frequencies outside the G mesh give zero. Not
used by the solver (see [`bubbles!`](@ref)); used in the tests.
"""
function bubbles_momentum_space!(
    Πpp :: NL2_MF_Π{Q},
    Πph :: NL2_MF_Π{Q},
    G   :: NL_MF_G{Q},
    ) :: Nothing where {Q}

    set!(Πpp, 0)
    set!(Πph, 0)
    meshK_G = meshes(G, Val(2))

    for iP in eachindex(meshes(Πpp, Val(3))), ik in eachindex(meshes(Πpp, Val(4)))
        P = euclidean(meshes(Πpp, Val(3))[iP], meshes(Πpp, Val(3)))
        k = euclidean(meshes(Πpp, Val(4))[ik], meshes(Πpp, Val(4)))

        # k, P - k and P + k on the G mesh
        k_G   = meshK_G[MatsubaraFunctions.mesh_index(    k, meshK_G)]
        Pmk_G = meshK_G[MatsubaraFunctions.mesh_index(P - k, meshK_G)]
        Ppk_G = meshK_G[MatsubaraFunctions.mesh_index(P + k, meshK_G)]

        for iΩ in eachindex(meshes(Πpp, Val(1))), iν in eachindex(meshes(Πpp, Val(2)))
            Ω = value(meshes(Πpp, Val(1))[iΩ])
            ν = value(meshes(Πpp, Val(2))[iν])

            if is_inbounds(ν, meshes(G, Val(1)))
                if is_inbounds(Ω - ν, meshes(G, Val(1)))
                    Πpp[iΩ, iν, iP, ik] = G[ν, k_G] * G[Ω - ν, Pmk_G]
                end

                if is_inbounds(Ω + ν, meshes(G, Val(1)))
                    Πph[iΩ, iν, iP, ik] = G[ν, k_G] * G[Ω + ν, Ppk_G]
                end
            end
        end

    end

    return nothing
end




"""
    bubbles_real_space!(Πpp::NL2_MF_Π, Πph::NL2_MF_Π, G1, G2 = G1; use_G_tail = false) -> nothing

NL2 bubbles computed in real space:

    Πpp(Ω, ν; q, k) = G2(ν, k) G1(Ω - ν, q - k)
    Πph(Ω, ν; q, k) = G2(ν, k) G1(Ω + ν, q + k)

With `G(k) = ∑_R G(R) e^{ikR}`, the pp product is
`∑_{R,R′} G1(R) G2(R′) e^{iqR} e^{ik(R′-R)}`, so `G1(R) G2(R′)` is added to the
real-space bubble at (R, R′ - R), and for ph at (R, R′ + R). R and R′ run over
the coarse box [-L/2, L/2]² (G at longer distances is dropped); a Fourier
transform over both distances then gives Π(q, k) on the coarse mesh. `G2` defaults to `G1`; a different `G2`
gives a mixed bubble.

`use_G_tail = true` uses the tail `iG ≈ 1/ν` outside the G mesh, but only for
R = 0 (see the NOTE in the code). It is off by default and not used.
"""
function bubbles_real_space!(
    Πpp :: NL2_MF_Π{Q},
    Πph :: NL2_MF_Π{Q},
    G1  :: NL_MF_G{Q},
    G2  :: NL_MF_G{Q} = G1
    ;
    use_G_tail :: Bool = false
    ) :: Nothing where {Q}

    # G(R) = ∑_k G(k) exp(-ikR) / N_k
    # Πpp(R, R' - R) = G(R) * G(R')
    # Πph(R, R' + R) = G(R) * G(R')
    # Π(P, k) = ∑_{R, R'} Π(R, R') exp(iPR) exp(ikR')

    set!(Πpp, 0)
    set!(Πph, 0)

    # linear sizes of the fine (G) and coarse (bubble) momentum meshes
    LG = bz(meshes(G1, Val(2))).L
    L  = bz(meshes(Πpp, Val(3))).L

    n1 = length(meshes(Πpp, Val(1)))
    n2 = length(meshes(Πpp, Val(2)))

    # G(ν, R) for all ν, by FFT over the two momentum directions
    G1_real_space = fft(reshape(G1.data, :, LG, LG), (2, 3)) / LG^2
    G2_real_space = fft(reshape(G2.data, :, LG, LG), (2, 3)) / LG^2

    # the bubble arrays seen as (Ω, ν, R1, R2), filled in real space first
    Πpp_R = Base.ReshapedArray(Πpp.data, (n1, n2, L, L, L, L), ())
    Πph_R = Base.ReshapedArray(Πph.data, (n1, n2, L, L, L, L), ())

    # Scan G(R) and G(R') for R and R' in [-L/2, L/2]²
    Rs_G_1d = (-div(L, 2)) : div(L, 2)
    Rs = collect(Iterators.product(Rs_G_1d, Rs_G_1d))

    for (Rp1, Rp2) in Rs
        Rp_vec = (Rp1, Rp2)

        for (R1, R2) in Rs
            R_vec = (R1, R2)

            # Half weight per coordinate on the shell |R| = L_G/2 of an even
            # fine mesh, where R and -R are the same site (only for L = L_G);
            # as in nonlocal/bubble.jl.
            weight = 1.0
            if mod(LG, 2) == 0
                abs(Rp1) == div(LG, 2) && (weight /= 2)
                abs(Rp2) == div(LG, 2) && (weight /= 2)
                abs(R1) == div(LG, 2) && (weight /= 2)
                abs(R2) == div(LG, 2) && (weight /= 2)
            end

            # Index of R and Rp in G
            iR_G  = mod.(R_vec,  (LG, LG)) .+ 1
            iRp_G = mod.(Rp_vec, (LG, LG)) .+ 1

            # Index of R, Rp - R, and Rp + R in Π
            iR_Π   = mod.(R_vec, (L, L)) .+ 1
            iRmR_Π = mod.(Rp_vec .- R_vec, (L, L)) .+ 1
            iRpR_Π = mod.(Rp_vec .+ R_vec, (L, L)) .+ 1

            G1_R  = MeshFunction((meshes(G1, Val(1)),), view(G1_real_space, :, iR_G...))
            G2_Rp = MeshFunction((meshes(G2, Val(1)),), view(G2_real_space, :, iRp_G...))

            for iΩ in eachindex(meshes(Πpp, Val(1))), iν in eachindex(meshes(Πpp, Val(2)))
                Ω = value(meshes(Πpp, Val(1))[iΩ])
                ν = value(meshes(Πpp, Val(2))[iν])

                # Πpp_R[iΩ, iν, iR_Π..., iRmR_Π...] += G1_R(Ω - ν) * G2_Rp(ν) * weight
                # Πph_R[iΩ, iν, iR_Π..., iRpR_Π...] += G1_R(Ω + ν) * G2_Rp(ν) * weight

                # NOTE: the tail 1/(iν) of G is purely on site, but this branch
                # tests only R = 0 and also applies the tail to G2(R′) at R′ ≠ 0
                # (and not at R′ = 0 for R ≠ 0). Harmless while use_G_tail = false.
                if R_vec == (0, 0) && use_G_tail
                    Πpp_R[iΩ, iν, iR_Π..., iRmR_Π...] += green_with_tail(G1_R, Ω - ν) * green_with_tail(G2_Rp, ν) * weight
                    Πph_R[iΩ, iν, iR_Π..., iRpR_Π...] += green_with_tail(G1_R, Ω + ν) * green_with_tail(G2_Rp, ν) * weight
                else
                    Πpp_R[iΩ, iν, iR_Π..., iRmR_Π...] += G1_R(Ω - ν) * G2_Rp(ν) * weight
                    Πph_R[iΩ, iν, iR_Π..., iRpR_Π...] += G1_R(Ω + ν) * G2_Rp(ν) * weight
                end
            end
        end
    end

    # back to momentum space: Π(q, k) = ∑_{R1,R2} Π(R1, R2) exp(iqR1 + ikR2)
    bfft!(Πpp_R, (3, 4, 5, 6))
    bfft!(Πph_R, (3, 4, 5, 6))

    return nothing
end
