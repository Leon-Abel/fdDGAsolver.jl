"""
    siam_bare_Green(mesh, ::Type{Q}; Δ, e, D)

Compute the bare (non-interacting) Green function G₀(iν) of the
**Single Impurity Anderson Model** (SIAM).

The bare Green function G₀(iν) is defined by integrating out the bath degrees
of freedom exactly (no interaction U), yielding the impurity Dyson equation:

    G₀(iν)⁻¹ = iν - e - Δ(iν)
 
- `Δ` : Hybridization function encoding the coupling to the bath.
- `e` : Impurity on-site energy plus Hartree contribution of the self-energy, therefore 
        `e = 0` corresponds to particle-hole symmetry (half-filling)
- `D` : Half bandwidth. Use `D = Inf` for the wide-band limit; `D < Inf` for a finite box-shaped bath.
"""
function siam_bare_Green(
    mesh :: FMesh,
         :: Type{Q} = ComplexF64,
    ;
    Δ, e, D,
    ) where {Q}

    G0 = MeshFunction(mesh; data_t = Q)

    # We store im·G₀.
    # For every fermionic Matsubara frequency iν_n = 2π(n + 1/2) * T we evaluate
    # the non-interacting impurity Green function with the chosen bath model.
    for ν in mesh
        # extract the real frequency value iν_n = 2π(n + 1/2) * T as a Float.
        ν_value = plain_value(ν)

        # In the wide-band limit (D → ∞) the hybridization function becomes purely imaginary and
        # frequency-independent: Δ(iν) = -iΔ · sgn(ν)   →   iG₀(iν) = 1 / (ν + ie + Δ · sgn(ν))
        # Δ ==Γ (spectral hybridization function)
        if D == Inf
            G0[ν] = 1 / (ν_value + im * e + Δ * sign(ν_value))

        # For a box-shaped bath density of states ρ(ε) = 1/(2D) on [-D, D], the
        # hybridization function evaluates to:
        # Δ(iν) = -i (2Δ/π) arctan(D/ν)   →   iG₀(iν) = 1 / (ν + ie + (2Δ/π) · arctan(D/ν))
        else
            G0[ν] = 1 / (ν_value + im * e + 2Δ / π * atan(D / ν_value))
        end
    end

    G0 :: MF_G{Q}
end

# make siam_bare_Green() publicly available to package
export siam_bare_Green
