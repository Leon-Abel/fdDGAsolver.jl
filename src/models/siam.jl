# =============================================================================
# Bare propagator of the single-impurity Anderson model (SIAM, also called
# AIM).
#
# One interacting impurity level coupled to a non-interacting bath (Lihm 2025,
# Eq. (6)). Integrating out the bath gives the bare impurity propagator
#
#   G₀(iν) = 1 / (iν - e - Δ(iν)),
#
# where the hybridization function Δ(iν) carries the effect of the bath. The
# interaction U enters only later, through the vertex.
#
# As everywhere in the code, the propagator is stored multiplied by i (see
# dyson.jl), and e is measured from half filling, so e = 0 is the
# particle-hole symmetric point.
# =============================================================================

"""
    siam_bare_Green(mesh, Q = ComplexF64; Δ, e, D) -> MeshFunction

Bare impurity propagator of the Anderson model on the fermionic Matsubara
`mesh`, stored as `iG₀(iν)` with `G₀(iν)⁻¹ = iν - e - Δ(iν)`.

- `Δ`: hybridization strength Γ = π ρ(0) |V|², the width the bath gives the
  impurity level. It sets the scale of Δ(iν).
- `e`: impurity level, measured from half filling (it includes the Hartree
  shift of the self-energy), so `e = 0` is particle-hole symmetric.
- `D`: half bandwidth of the box-shaped bath density of states. `D = Inf` is
  the wide-band limit.

For a box bath, `Δ(iν) = -i (2Δ/π) arctan(D/ν)`; in the wide-band limit this
becomes `-iΔ sgn(ν)`. Lihm 2025 uses U = 5.75, V = 2 and D = 10 for the AIM
(text below Eq. (6)), i.e. `Δ = π V² / (2D) ≈ 0.63`.
"""
function siam_bare_Green(
    mesh :: FMesh,
         :: Type{Q} = ComplexF64,
    ;
    Δ, e, D,
    ) where {Q}

    G0 = MeshFunction(mesh; data_t = Q)

    # We store im·G₀.
    # For every fermionic Matsubara frequency iν_n = iπT(2n + 1) we evaluate
    # the non-interacting impurity Green function with the chosen bath model.
    for ν in mesh
        # the real number ν_n = πT(2n + 1)
        ν_value = plain_value(ν)

        # In the wide-band limit (D → ∞) the hybridization function becomes purely imaginary and
        # frequency-independent: Δ(iν) = -iΔ · sgn(ν)   →   iG₀(iν) = 1 / (ν + ie + Δ · sgn(ν))
        # (here Δ is the hybridization strength Γ)
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
