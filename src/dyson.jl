# =============================================================================
# Dyson equation, occupation, and the chemical potential for a given density.
#
# The Dyson equation G = (G₀⁻¹ - Σ)⁻¹ gives the full propagator from the bare
# one and the self-energy. The occupation per spin follows from G, and for the
# Hubbard model μ can be tuned so that it matches a target density (used to
# keep the density at its DMFT value, Lihm 2025, text below Eq. (7)).
#
# Sign and phase convention, used in the whole code: Green functions and
# self-energies are stored multiplied by the imaginary unit, i.e. the arrays
# hold iG and iΣ instead of G and Σ. At half filling these are real. With
# G̃ = iG, Σ̃ = iΣ, G̃₀ = iG₀ the Dyson equation becomes
#
#   G = (G₀⁻¹ - Σ)⁻¹   =>   G̃ = (G̃₀⁻¹ + Σ̃)⁻¹
#
# i.e. the minus sign in front of Σ turns into a plus sign. A consequence is
# that the stored bubbles (iG)(iG) = -GG carry an extra minus sign
# (bubble.jl).
# =============================================================================

"""
    Dyson!(S::AbstractSolver)
    Dyson!(G, Σ, Gbare) -> nothing

Dyson equation, in place: overwrite `G` with `(Gbare⁻¹ - Σ)⁻¹`, written for the
stored iG, iΣ as `iG = ((iGbare)⁻¹ + iΣ)⁻¹`. Works for local (`G(ν)`) and lattice
(`G(ν, k)`) propagators. The solver method updates the target propagator
`S.G` from `S.Σ` and `S.Gbare`.
"""
function Dyson!(S::AbstractSolver)
    Dyson!(S.G, S.Σ, S.Gbare)
end

# Dyson! for a local propagator G(ν), see the docstring above.
function Dyson!(
    G     :: MF_G,
    Σ     :: MF_G,
    Gbare :: MF_G,
    ) :: Nothing

    for ν in meshes(G, Val(1))
        # G and Σ actually store im * G and im * Σ, so -Σ in the Dyson equation becomes +Σ.
        G[ν] = 1 / (1 / Gbare[ν] + Σ[ν])
    end

    return nothing
end

# Dyson! for a lattice propagator G(ν, k), see the docstring above.
function Dyson!(
    G     :: NL_MF_G,
    Σ     :: NL_MF_G,
    Gbare :: NL_MF_G,
    ) :: Nothing

    for k in meshes(G, Val(2)), ν in meshes(G, Val(1))
        # G and Σ is im * G and im * Σ, so -Σ in the Dyson equation becomes +Σ.
        G[ν, k] = 1 / (1 / Gbare[ν, k] + Σ[ν, k])
    end

    return nothing
end

"""
    compute_occupation(G) -> n

Occupation per spin, `n = ⟨n_σ⟩`, from the propagator: the local `G(ν)`, or the
lattice `G(ν, k)` averaged over the Brillouin zone.

`n = T ∑_ν G(iν) e^{iν0⁺}`. The 1/(iν) tail of G gives the 1/2; the rest is
`T ∑_ν Re G(iν)`, which converges on a finite frequency mesh. Since iG is stored,
`Re G = Im(iG)`.
"""
function compute_occupation(
    G :: MF_G
    ) :: Float64
    # ⟨n⟩ = G(τ = 0⁻) = T ∑_ν G(iν) e^{iν0⁺} (from the inverse Fourier transform)
    # the ½ offset comes from the asymptotic tail G(iν) ~ 1/(iν) for large ν;
    # summing the full Green's function over a finite Matsubara grid gives a converged result
    # only after adding this analytic contribution from the high-frequency tail.
    return 0.5 + imag(sum(G.data)) * temperature(meshes(G, Val(1)))
end

# compute_occupation for a lattice propagator, see the docstring above.
function compute_occupation(G :: NL_MF_G)
    #⟨n⟩ = (1/N_k) ∑_k [ ½ + T ∑_ν (im * G(iν, k)) ]
    #   = ½ + T/N_k · Im[ ∑_{ν,k} (im * G(iν, k)) ]
    return 0.5 + imag(sum(G.data)) * temperature(meshes(G, Val(1))) / length(meshes(G, Val(2)))
end

"""
    compute_hubbard_chemical_potential(occ_target, Σ, hubbard_params) -> μ

Chemical potential μ of the Hubbard model for which the occupation per spin
equals `occ_target`, at fixed self-energy `Σ`. For a trial μ it builds the bare
propagator ([`hubbard_bare_Green`](@ref) with `hubbard_params`, e.g.
`(; t1, t2)`), solves the Dyson equation and computes the occupation. A
bracketing root finder then solves `n(μ) = occ_target`. The returned μ uses the
code's convention, without the Hartree U/2 (models/hubbard.jl).
"""
function compute_hubbard_chemical_potential(occ_target, Σ, hubbard_params)
    # temporary Green's function with the same mesh structure as Σ
    G_tmp = copy(Σ)

    # occupation for a given trial chemical potential μ
    function occupation(μ)
        Gbare = hubbard_bare_Green(meshes(Σ)...; μ, hubbard_params...)
        Dyson!(G_tmp, Σ, Gbare)
        compute_occupation(G_tmp)
    end

    # Use a bracketed scalar root-finder to solve occupation(μ) = occ_target.
    # The search interval [-4|t₁|, +4|t₁|] covers the bandwidth W = 8|t₁| of
    # the nearest-neighbor square lattice.
    μ = find_zero(μ -> occupation(μ) - occ_target, (-4 * abs(hubbard_params.t1), 4 * abs(hubbard_params.t1)))

    return μ
end