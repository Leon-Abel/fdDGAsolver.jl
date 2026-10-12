# =============================================================================
# Symmetries of the K3 class of the lattice solver NL3, which keeps both
# fermionic momenta: K3(Ω, ν, ν′; q, k, k′) ("without s-wave truncation").
#
# The same symmetries as for the s-wave K3 (nonlocal/symmetries.jl), now with
# the fermionic momenta transformed together with their frequencies: the
# square-lattice point group acts on all three momenta, complex conjugation
# flips all of them, the exchange ν ↔ ν′ comes with k ↔ k′, and the double
# crossing (Lihm 2025, Eq. (S4c); in the particle-hole channels combined with
# ν ↔ ν′) maps each fermionic momentum like its frequency:
#   p channel:  (Ω, ν, ν′; q, k, k′) → (Ω, Ω - ν, Ω - ν′; q, q - k, q - k′)
#   a, t:       (Ω, ν, ν′; q, k, k′) → (-Ω, Ω + ν, Ω + ν′; -q, q + k, q + k′)
# K1 and K2 of NL3 use the symmetries of NL and NL2.
# =============================================================================

# -----------------------------------------------------------------------------
# Point group
# -----------------------------------------------------------------------------

"""
    sK3_NL3_ref(w, m) -> (w′, op)

Diagonal reflection of all three momenta of K3(Ω, ν, ν′; q, k, k′).
"""
function sK3_NL3_ref(
    w :: Tuple{MatsubaraFrequency, MatsubaraFrequency, MatsubaraFrequency, BrillouinPoint{2}, BrillouinPoint{2}, BrillouinPoint{2}},
    m :: KMesh
    )
    return (w[1], w[2], w[3], _ref(w[4], m), _ref(w[5], m), _ref(w[6], m)), Operation{ComplexF64}()
end

"""
    sK3_NL3_rot(w, m) -> (w′, op)

Rotation by π/2 of all three momenta of K3(Ω, ν, ν′; q, k, k′).
"""
function sK3_NL3_rot(
    w :: Tuple{MatsubaraFrequency, MatsubaraFrequency, MatsubaraFrequency, BrillouinPoint{2}, BrillouinPoint{2}, BrillouinPoint{2}},
    m :: KMesh
    )
    return (w[1], w[2], w[3], _rot(w[4], m), _rot(w[5], m), _rot(w[6], m)), Operation{ComplexF64}()
end

# -----------------------------------------------------------------------------
# Particle-particle channel
# -----------------------------------------------------------------------------

"""
    sK3_NL3_pp1(w, m) -> (w′, op)

Complex conjugation: `K3(Ω, ν, ν′; q, k, k′) = K3(-Ω, -ν, -ν′; -q, -k, -k′)*`.
"""
function sK3_NL3_pp1(
    w :: Tuple{MatsubaraFrequency, MatsubaraFrequency, MatsubaraFrequency, BrillouinPoint{2}, BrillouinPoint{2}, BrillouinPoint{2}},
    m :: KMesh
    )
    return (-w[1], -w[2], -w[3], fold_back(-w[4], m), fold_back(-w[5], m), fold_back(-w[6], m)), Operation{ComplexF64}(sgn = false, con = true)
end

"""
    sK3_NL3_pp2(w, m) -> (w′, op)

Exchange ν ↔ ν′ together with k ↔ k′: `K3(Ω, ν, ν′; q, k, k′) = K3(Ω, ν′, ν; q, k′, k)`.
"""
function sK3_NL3_pp2(
    w :: Tuple{MatsubaraFrequency, MatsubaraFrequency, MatsubaraFrequency, BrillouinPoint{2}, BrillouinPoint{2}, BrillouinPoint{2}},
    m :: KMesh
    )
    return (w[1], w[3], w[2], w[4], w[6], w[5]), Operation{ComplexF64}()
end

"""
    sK3_NL3_pp3(w, m) -> (w′, op)

Double crossing of the p-channel K3:
`K3(Ω, ν, ν′; q, k, k′) = K3(Ω, Ω - ν, Ω - ν′; q, q - k, q - k′)`.
"""
function sK3_NL3_pp3(
    w :: Tuple{MatsubaraFrequency, MatsubaraFrequency, MatsubaraFrequency, BrillouinPoint{2}, BrillouinPoint{2}, BrillouinPoint{2}},
    m :: KMesh
    )
    return (w[1], w[1] - w[2], w[1] - w[3], w[4], fold_back(w[4] - w[5], m), fold_back(w[4] - w[6], m)), Operation{ComplexF64}()
end

# -----------------------------------------------------------------------------
# Particle-hole channels
# -----------------------------------------------------------------------------

"""
    sK3_NL3_ph1(w, m) -> (w′, op)

Complex conjugation of the particle-hole K3 (same as for p).
"""
function sK3_NL3_ph1(
    w :: Tuple{MatsubaraFrequency, MatsubaraFrequency, MatsubaraFrequency, BrillouinPoint{2}, BrillouinPoint{2}, BrillouinPoint{2}},
    m :: KMesh
    )
    return sK3_NL3_pp1(w, m)
end

"""
    sK3_NL3_ph2(w, m) -> (w′, op)

Exchange ν ↔ ν′ together with k ↔ k′ in the particle-hole K3 (same as for p).
"""
function sK3_NL3_ph2(
    w :: Tuple{MatsubaraFrequency, MatsubaraFrequency, MatsubaraFrequency, BrillouinPoint{2}, BrillouinPoint{2}, BrillouinPoint{2}},
    m :: KMesh
    )
    return sK3_NL3_pp2(w, m)
end

"""
    sK3_NL3_ph3(w, m) -> (w′, op)

Double crossing of the particle-hole K3, combined with ν ↔ ν′:
`K3(Ω, ν, ν′; q, k, k′) = K3(-Ω, Ω + ν, Ω + ν′; -q, q + k, q + k′)`.
"""
function sK3_NL3_ph3(
    w :: Tuple{MatsubaraFrequency, MatsubaraFrequency, MatsubaraFrequency, BrillouinPoint{2}, BrillouinPoint{2}, BrillouinPoint{2}},
    m :: KMesh
    )
    return (-w[1], w[1] + w[2], w[1] + w[3], fold_back(-w[4], m), fold_back(w[4] + w[5], m), fold_back(w[4]+ w[6], m)), Operation{ComplexF64}()
end
