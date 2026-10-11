# =============================================================================
# Lattice bubbles of the s-wave solver NL, and the propagator with its
# high-frequency tail.
#
# The NL bubble is averaged over the fermionic loop momentum k, as in the
# s-wave approximation (Lihm 2025, Eq. (S18b)):
#
#   Πpp(Ω, ν; P) = 1/N_k ∑_k G(ν, k) G(Ω - ν, P - k)    (p channel)
#   Πph(Ω, ν; P) = 1/N_k ∑_k G(ν, k) G(Ω + ν, P + k)    (a and t channels)
#
# A sum of this kind is a convolution in momentum. It becomes a product in
# real space, so the code Fourier transforms G to G(R), multiplies, and
# transforms back (bubbles_real_space!). The direct k-sum is kept for checks
# (bubbles_momentum_space!).
#
# G lives on a fine momentum mesh (L_G × L_G points), the bubble on the coarse
# vertex mesh (L × L). In real space only distances R in [-L/2, L/2]² are kept,
# which is all the coarse mesh can resolve.
#
# As everywhere, G is stored as iG, so the stored bubble is (iG)(iG) = -GG
# (bubble.jl, BSE_templates.jl).
# =============================================================================

# -----------------------------------------------------------------------------
# Solver wrappers
# -----------------------------------------------------------------------------

"""
    bubbles_momentum_space!(S::AbstractSolver)
    bubbles_momentum_space!(Πpp, Πph, G) -> nothing

Lattice bubbles from the direct momentum sum,
`Π(Ω, ν; P) = 1/N_k ∑_k G(ν, k) G(Ω ∓ ν, P ∓ k)`, for frequencies inside the G
mesh only. Slower than [`bubbles_real_space!`](@ref); used in tests.
"""
function bubbles_momentum_space!(S :: AbstractSolver)
    bubbles_momentum_space!(S.Πpp, S.Πph, S.G)
end


# Solver method of bubbles_real_space!, see below.
function bubbles_real_space!(S :: AbstractSolver)
    bubbles_real_space!(S.Πpp, S.Πph, S.G)
end

# Array method of bubbles_momentum_space!, see the docstring above.
function bubbles_momentum_space!(
    Πpp :: NL_MF_Π{Q},
    Πph :: NL_MF_Π{Q},
    G   :: NL_MF_G{Q},
    ) :: Nothing where {Q}

    set!(Πpp, 0)
    set!(Πph, 0)
    meshK_G = meshes(G, Val(2))

    for iP in eachindex(meshes(Πpp, Val(3)))
        P = euclidean(meshes(Πpp, Val(3))[iP], meshes(Πpp, Val(3)))

        for ik in eachindex(meshes(G, Val(2)))
            k = euclidean(meshes(G, Val(2))[ik], meshes(G, Val(2)))

            # k, P - k and P + k on the G mesh
            k_G   = meshK_G[MatsubaraFunctions.mesh_index(    k, meshK_G)]
            Pmk_G = meshK_G[MatsubaraFunctions.mesh_index(P - k, meshK_G)]
            Ppk_G = meshK_G[MatsubaraFunctions.mesh_index(P + k, meshK_G)]

            for iΩ in eachindex(meshes(Πpp, Val(1))), iν in eachindex(meshes(Πpp, Val(2)))
                Ω = value(meshes(Πpp, Val(1))[iΩ])
                ν = value(meshes(Πpp, Val(2))[iν])

                if is_inbounds(ν, meshes(G, Val(1)))
                    if is_inbounds(Ω - ν, meshes(G, Val(1)))
                        Πpp[iΩ, iν, iP] += G[ν, k_G] * G[Ω - ν, Pmk_G]
                    end

                    if is_inbounds(Ω + ν, meshes(G, Val(1)))
                        Πph[iΩ, iν, iP] += G[ν, k_G] * G[Ω + ν, Ppk_G]
                    end
                end
            end
        end

    end

    # average over k
    Πpp.data ./= length(meshes(G, Val(2)))
    Πph.data ./= length(meshes(G, Val(2)))

    return nothing
end

# -----------------------------------------------------------------------------
# Propagator with its high-frequency tail
# -----------------------------------------------------------------------------

"""
    green_with_tail(G, ν) -> value
    green_with_tail(G, ν, k) -> value

The stored propagator iG at frequency `ν` (and momentum `k`), or its
high-frequency tail outside the frequency mesh: `G ≈ 1/(iν)`, so `iG ≈ 1/ν`.
For a momentum given as a vector (`SVector`), G is interpolated linearly
between the mesh points.
"""
function green_with_tail(G :: MeshFunction{1, Q}, ν :: MatsubaraFrequency{Fermion}) :: Q where {Q}
    # Evaluate Green function G using the tail 1 / (im * ν) for extrapolation
    # Since we factor out (-im) factor in the Green function, the tail is 1 / ν.
    return is_inbounds(ν, meshes(G, Val(1))) ? G[ν] : 1 / value(ν)
    # return is_inbounds(ν, meshes(G, Val(1))) ? G[ν] : zero(Q)
end

# green_with_tail at a mesh momentum, see the docstring above
function green_with_tail(G :: MeshFunction{2, Q}, ν :: MatsubaraFrequency{Fermion}, k :: BrillouinPoint) :: Q where {Q}
    return is_inbounds(ν, meshes(G, Val(1))) ? G(ν, k) : 1 / value(ν)
end

# Linear interpolation of G(ν, ·) in momentum: weighted sum over the
# neighboring mesh points given by the interpolation parameter k_itp.
function _evaluate_G(G :: MeshFunction{2, Q}, ν, k_itp) :: Q where {Q}
    k_indices = indices(k_itp)
    k_weights = weights(k_itp)

    val = Q(0)
    for (i, w) in zip(k_indices, k_weights)
        val += w * G[ν, i]
    end
    val
end

# green_with_tail at an arbitrary momentum vector (interpolated), see the docstring above
function green_with_tail(G :: MeshFunction{2, Q}, ν :: MatsubaraFrequency{Fermion}, k :: SVector{2}) :: Q where {Q}
    if is_inbounds(ν, meshes(G, Val(1)))
        k_itp = MatsubaraFunctions.InterpolationParam(k, meshes(G, Val(2)))
        return _evaluate_G(G, ν, k_itp)
    else
        return 1 / value(ν)
    end

end

# -----------------------------------------------------------------------------
# Bubbles by real-space convolution
# -----------------------------------------------------------------------------

"""
    bubbles_real_space!(S::AbstractSolver)
    bubbles_real_space!(Πpp, Πph, G1, G2 = G1; use_G_tail = true)

Lattice bubbles of the s-wave solver NL, averaged over the loop momentum:

    Πpp(Ω, ν; P) = 1/N_k ∑_k G2(ν, k) G1(Ω - ν, P - k)
    Πph(Ω, ν; P) = 1/N_k ∑_k G2(ν, k) G1(Ω + ν, P + k)

computed in real space. G(R) is the Fourier transform of G(k) on the fine
mesh. The products `G1(R) G2(R)` (pp) and `G1(R) G2(-R)` (ph) are formed for R
in the coarse box [-L/2, L/2]², and the result is transformed back to the
coarse momentum mesh of the bubble.

With `use_G_tail = true`, frequencies outside the G mesh use the tail
`iG ≈ 1/ν`, but only at R = 0: the 1/(iν) tail of G(k, iν) is the same for
every k, so in real space it sits entirely on site. `G2` defaults to `G1`; a
different `G2` gives a mixed bubble. The solver method computes the target
bubbles from `S.G`.
"""
function bubbles_real_space!(
    Πpp :: NL_MF_Π{Q},
    Πph :: NL_MF_Π{Q},
    G1  :: NL_MF_G{Q},
    G2  :: NL_MF_G{Q} = G1
    ;
    use_G_tail :: Bool = true
    ) where {Q}

    # G(R) = ∑_k G(k) exp(-ikR) / N_k
    # Πpp(R) = G1(R) * G2( R)
    # Πph(R) = G1(R) * G2(-R)
    # Π(P) = ∑_{R} Π(R) exp(iPR)

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
    Πpp_R = Base.ReshapedArray(Πpp.data, (n1, n2, L, L), ())
    Πph_R = Base.ReshapedArray(Πph.data, (n1, n2, L, L), ())

    # Scan G(R) and G(R') for R and R' in [-L/2, L/2]²
    Rs_G_1d = (-div(L, 2)) : div(L, 2)
    Rs = collect(Iterators.product(Rs_G_1d, Rs_G_1d))

    for (R1, R2) in Rs
        R_vec = (R1, R2)

        # Half weight on the shell |R| = L_G/2 of an even fine mesh: there R and
        # -R are the same lattice site, which the box would count twice. This
        # only happens for L = L_G. For L < L_G the points ±L/2 are different
        # sites, and both count fully.
        weight = 1.0
        if mod(LG, 2) == 0
            abs(R1) == div(LG, 2) && (weight /= 2)
            abs(R2) == div(LG, 2) && (weight /= 2)
        end

        # Index of R and Rp in G
        ipR_G = mod.(  R_vec, (LG, LG)) .+ 1
        imR_G = mod.(.-R_vec, (LG, LG)) .+ 1

        # Index of R, Rp - R, and Rp + R in Π
        iR_Π   = mod.(R_vec, (L, L)) .+ 1

        G1_pR = MeshFunction((meshes(G1, Val(1)),), view(G1_real_space, :, ipR_G...))
        G2_pR = MeshFunction((meshes(G2, Val(1)),), view(G2_real_space, :, ipR_G...))
        G2_mR = MeshFunction((meshes(G2, Val(1)),), view(G2_real_space, :, imR_G...))

        for iΩ in eachindex(meshes(Πpp, Val(1))), iν in eachindex(meshes(Πpp, Val(2)))
            Ω = value(meshes(Πpp, Val(1))[iΩ])
            ν = value(meshes(Πpp, Val(2))[iν])

            if R_vec == (0, 0) && use_G_tail
                # on site: use the 1/ν tail outside the G mesh
                Πpp_R[iΩ, iν, iR_Π...] += green_with_tail(G1_pR, Ω - ν) * green_with_tail(G2_pR, ν) * weight
                Πph_R[iΩ, iν, iR_Π...] += green_with_tail(G1_pR, Ω + ν) * green_with_tail(G2_mR, ν) * weight
            else
                Πpp_R[iΩ, iν, iR_Π...] += G1_pR(Ω - ν) * G2_pR(ν) * weight
                Πph_R[iΩ, iν, iR_Π...] += G1_pR(Ω + ν) * G2_mR(ν) * weight
            end
        end
    end

    # back to momentum space: Π(P) = ∑_R Π(R) exp(iPR)
    bfft!(Πpp_R, (3, 4))
    bfft!(Πph_R, (3, 4))

    return nothing
end
