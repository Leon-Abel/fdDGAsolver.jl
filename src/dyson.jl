# dyson.jl
#
# Implements the Dyson equation for the single-particle Green's function, 
# along with utilities for computing the electron occupation
# and the chemical potential.
#
# IMPORTANT SIGN/PHASE CONVENTION: 
# Throughout this codebase, all Green's functions and self-energies are stored 
# multiplied by the imaginary unit `im`, i.e. the arrays contain (im * G) and (im * Σ)
# rather than G and Σ themselves. This transforms the Dyson equation as follows:
# Writing G̃ = im*G, Σ̃ = im*Σ, G̃₀ = im*G₀:
#   G(iν) = [G₀⁻¹(iν) - Σ(iν)]⁻¹
#   => im * G = [1/(im*G₀) - (-1)*(im*Σ)]⁻¹  -- note the sign flip: -Σ becomes +Σ̃
#   => G̃ = [G̃₀⁻¹ + Σ̃]⁻¹

"""
    Dyson!(S::AbstractSolver)
 
Wrapper that applies the Dyson equation in-place using the fields of a solver object `S`. 
Delegates to the appropriate `Dyson!(G, Σ, Gbare)` method depending on the type of `S.G`.
"""
function Dyson!(S::AbstractSolver)
    Dyson!(S.G, S.Σ, S.Gbare)
end

"""
    Dyson!(G::MF_G, Σ::MF_G, Gbare::MF_G,)::Nothing 
    
Applies the Dyson equation in-place for a **local** (momentum-independent) Green's function
defined on a fermionic Matsubara frequency mesh.
"""
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

"""
    Dyson!(G::NL_MF_G, Σ::NL_MF_G, Gbare::NL_MF_G,)::Nothing
 
Applies the Dyson equation in-place for a **nonlocal** (momentum-resolved) Green's function
defined on a fermionic Matsubara frequency mesh and a Brillouin-zone k-mesh.
"""
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
    compute_occupation(G::MF_G)::Float64
 
Computes the single-spin electron occupation ⟨n⟩ from the **local**
Matsubara Green's function.
"""
function compute_occupation(
    G :: MF_G
    ) :: Float64
    # ⟨n⟩ = T Σ_ν G(iν) e^{-iν0⁺} (from inverse fourier transform)
    # the ½ offset comes from the asymptotic tail G(iν) ~ 1/(iν) for large ν;
    # summing the full Green's function over a finite Matsubara grid gives a converged result 
    # only after adding this analytic contribution from the high-frequency tail.
    return 0.5 + imag(sum(G.data)) * temperature(meshes(G, Val(1)))
end

"""
    compute_occupation(G::NL_MF_G)
 
Computes the single-spin electron occupation ⟨n⟩ for the **nonlocal** (lattice) Green's
function by averaging over the Brillouin zone.
"""
function compute_occupation(G :: NL_MF_G)
    #⟨n⟩ = (1/N_k) Σ_k [ ½ + T Σ_ν (im * G(iν, k)) ]
    #   = ½ + T/N_k · Im[ Σ_{ν,k} (im * G(iν, k)) ]
    return 0.5 + imag(sum(G.data)) * temperature(meshes(G, Val(1))) / length(meshes(G, Val(2)))
end

"""
    compute_hubbard_chemical_potential(occ_target, Σ, hubbard_params)
 
Finds the chemical potential μ for the Hubbard model such that the electron occupation
matches a target value `occ_target` (per spin), given a fixed self-energy `Σ`.
 
This is a root-finding problem: 
    f(μ) = ⟨n⟩(μ) - occ_target = 0
where ⟨n⟩(μ) is obtained by constructing the bare Green's function G₀(iν, k; μ),
solving the Dyson equation for G(iν, k), and computing the occupation.
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

    # Use a bracketed scalar root-finder to solve occupation(μ) = occ_target
    # the search-interval [-4|t₁|, +4|t₁|] contains the full non-interacting bandwidth 
    # W = 8|t₁| of the 2D square lattice (nearest-neighbor only).
    μ = find_zero(μ -> occupation(μ) - occ_target, (-4 * abs(hubbard_params.t1), 4 * abs(hubbard_params.t1)))

    return μ
end