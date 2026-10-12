# =============================================================================
# Symmetries of the K2 class of the lattice solver NL2, which keeps the
# fermionic momentum: K2(Ω, ν, q, k).
#
# The same symmetries as for the s-wave K2 (nonlocal/symmetries.jl), now with
# the fermionic momentum k transformed as well. The square-lattice point group
# acts on both momenta, and complex conjugation flips both (q, k → -q, -k). The
# double crossing (Lihm 2025, Eq. (S4c); in the particle-hole channels combined
# with ν ↔ ν′) maps k in the same way as its frequency ν:
#   p channel:  (Ω, ν, q, k) → (Ω, Ω - ν, q, q - k)
#   a, t:       (Ω, ν, q, k) → (-Ω, Ω + ν, -q, q + k)
# K1 and K3 of NL2 use the s-wave symmetries of nonlocal/symmetries.jl.
# =============================================================================

# -----------------------------------------------------------------------------
# Point group
# -----------------------------------------------------------------------------

"""
    sK2_NL2_ref(w, m) -> (w′, op)

Diagonal reflection of both momenta of K2(Ω, ν, q, k).
"""
function sK2_NL2_ref(
    w :: Tuple{MatsubaraFrequency, MatsubaraFrequency, BrillouinPoint{2}, BrillouinPoint{2}},
    m :: KMesh
    )
    return (w[1], w[2], _ref(w[3], m), _ref(w[4], m)), Operation{ComplexF64}()
end

"""
    sK2_NL2_rot(w, m) -> (w′, op)

Rotation by π/2 of both momenta of K2(Ω, ν, q, k).
"""
function sK2_NL2_rot(
    w :: Tuple{MatsubaraFrequency, MatsubaraFrequency, BrillouinPoint{2}, BrillouinPoint{2}},
    m :: KMesh
    )
    return (w[1], w[2], _rot(w[3], m), _rot(w[4], m)), Operation{ComplexF64}()
end

# -----------------------------------------------------------------------------
# Particle-particle channel
# -----------------------------------------------------------------------------

"""
    sK2_NL2_pp1(w, m) -> (w′, op)

Complex conjugation: `K2(Ω, ν, q, k) = K2(-Ω, -ν, -q, -k)*`.
"""
function sK2_NL2_pp1(
    w :: Tuple{MatsubaraFrequency, MatsubaraFrequency, BrillouinPoint{2}, BrillouinPoint{2}},
    m :: KMesh
    )
    return (-w[1], -w[2], fold_back(-w[3], m), fold_back(-w[4], m)), Operation{ComplexF64}(sgn = false, con = true)
end

"""
    sK2_NL2_pp2(w, m) -> (w′, op)

Double crossing of the p-channel K2: `K2(Ω, ν, q, k) = K2(Ω, Ω - ν, q, q - k)`.
"""
function sK2_NL2_pp2(
    w :: Tuple{MatsubaraFrequency, MatsubaraFrequency, BrillouinPoint{2}, BrillouinPoint{2}},
    m :: KMesh
    )
    return (w[1], w[1] - w[2], w[3], fold_back(w[3] - w[4], m)), Operation{ComplexF64}()
end

# -----------------------------------------------------------------------------
# Particle-hole channels
# -----------------------------------------------------------------------------

"""
    sK2_NL2_ph1(w, m) -> (w′, op)

Complex conjugation of the particle-hole K2 (same as for p).
"""
function sK2_NL2_ph1(
    w :: Tuple{MatsubaraFrequency, MatsubaraFrequency, BrillouinPoint{2}, BrillouinPoint{2}},
    m :: KMesh
    )
    return sK2_NL2_pp1(w, m)
end

"""
    sK2_NL2_ph2(w, m) -> (w′, op)

Double crossing of the particle-hole K2, combined with ν ↔ ν′:
`K2(Ω, ν, q, k) = K2(-Ω, Ω + ν, -q, q + k)`.
"""
function sK2_NL2_ph2(
    w :: Tuple{MatsubaraFrequency, MatsubaraFrequency, BrillouinPoint{2}, BrillouinPoint{2}},
    m :: KMesh
    )
    return (-w[1], w[1] + w[2], fold_back(-w[3], m), fold_back(w[3] + w[4], m)), Operation{ComplexF64}()
end
