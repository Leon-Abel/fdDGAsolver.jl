# =============================================================================
# Symmetries of the lattice self-energy and of the lattice vertex classes
# (s-wave solver NL; also used by NL2 and NL3).
#
# These are the local symmetries of symmetries.jl (complex conjugation, double
# crossing, ν ↔ ν′; Lihm 2025, Eq. (S4)), now with the momentum arguments
# transformed as well, plus the point-group symmetries of the square lattice
# (the "reciprocal space crystal symmetries" of Lihm 2025, Sec. S-I H):
#   reflection   (kx, ky) → (ky, kx), across the zone diagonal
#   rotation     (kx, ky) → (ky, -kx), by π/2
# Together these two generate all eight symmetries of the square.
#
# Each function takes a tuple w of arguments, (frequencies…, momentum), and the
# momentum mesh m. It returns the arguments of the related point and an
# `Operation` (sgn: change the sign, con: complex conjugate). A symmetry group
# uses them to compute one point per set of related points. Under complex
# conjugation all momenta flip sign too (k → -k), folded back into the
# Brillouin zone.
#
# For the s-wave classes K2(Ω, ν, q) and K3(Ω, ν, ν′, q) only the bosonic
# momentum q is transformed. The double crossing leaves q unchanged in the p
# channel and maps q → -q in the particle-hole channels.
# =============================================================================

# -----------------------------------------------------------------------------
# Point-group operations on a momentum
# -----------------------------------------------------------------------------

"""
    _ref(k, m) -> k′

Reflection of the momentum `k` across the zone diagonal: (kx, ky) → (ky, kx).
"""
function _ref(k :: BrillouinPoint{2}, m :: KMesh) :: BrillouinPoint{2}
    # reflection along zone diagonal: (x, y) -> (y, x)
    return BrillouinPoint(k[2], k[1])
end

"""
    _rot(k, m) -> k′

Rotation of the momentum `k` by π/2: (kx, ky) → (ky, -kx), folded back into
the Brillouin zone of the mesh `m`.
"""
function _rot(k :: BrillouinPoint{2}, m :: KMesh) :: BrillouinPoint{2}
    # rotation by π/2: (x, y) -> (y, -x)
    return fold_back(BrillouinPoint(k[2], -k[1]), m)
end

# -----------------------------------------------------------------------------
# Self-energy Σ(ν, k)
# -----------------------------------------------------------------------------

"""
    sΣ_conj(w, m) -> (w′, op)

Complex conjugation of the lattice self-energy, `Σ(-ν, -k) = Σ(ν, k)*`. Since
iΣ is stored, this reads `iΣ(-ν, -k) = -[iΣ(ν, k)]*`.
"""
function sΣ_conj(w :: Tuple{MatsubaraFrequency, BrillouinPoint{2}}, m :: KMesh)
    # we store iΣ, so the symmetry Σ -> Σ* becomes iΣ -> -(iΣ)*
    return (-w[1], fold_back(-w[2], m)), Operation{ComplexF64}(sgn = true, con = true)
end

"""
    sΣ_ref(w, m) -> (w′, op)

Diagonal reflection of the momentum of the self-energy.
"""
function sΣ_ref(w :: Tuple{MatsubaraFrequency, BrillouinPoint{2}}, m :: KMesh)
    return (w[1], _ref(w[2], m)), Operation{ComplexF64}()
end

"""
    sΣ_rot(w, m) -> (w′, op)

Rotation by π/2 of the momentum of the self-energy.
"""
function sΣ_rot(w :: Tuple{MatsubaraFrequency, BrillouinPoint{2}}, m :: KMesh)
    return (w[1], _rot(w[2], m)), Operation{ComplexF64}()
end

# -----------------------------------------------------------------------------
# K1(Ω, q), the same for all channels
# -----------------------------------------------------------------------------

"""
    sK1_conj(w, m) -> (w′, op)

Complex conjugation of K1: `K1(Ω, q) = K1(-Ω, -q)*`.
"""
function sK1_conj(w :: Tuple{MatsubaraFrequency, BrillouinPoint{2}}, m :: KMesh)
    return (-w[1], fold_back(-w[2], m)), Operation{ComplexF64}(sgn = false, con = true)
end

"""
    sK1_ref(w, m) -> (w′, op)

Diagonal reflection of the bosonic momentum of K1.
"""
function sK1_ref(w :: Tuple{MatsubaraFrequency, BrillouinPoint{2}}, m :: KMesh)
    return (w[1], _ref(w[2], m)), Operation{ComplexF64}()
end

"""
    sK1_rot(w, m) -> (w′, op)

Rotation by π/2 of the bosonic momentum of K1.
"""
function sK1_rot(w :: Tuple{MatsubaraFrequency, BrillouinPoint{2}}, m :: KMesh)
    return (w[1], _rot(w[2], m)), Operation{ComplexF64}()
end

# -----------------------------------------------------------------------------
# K2(Ω, ν, q) in the s-wave approximation
# -----------------------------------------------------------------------------

"""
    sK2_ref(w, m) -> (w′, op)

Diagonal reflection of the bosonic momentum of an s-wave K2 class.
"""
function sK2_ref(w :: Tuple{MatsubaraFrequency, MatsubaraFrequency, BrillouinPoint{2}}, m :: KMesh)
    return (w[1], w[2], _ref(w[3], m)), Operation{ComplexF64}()
end

"""
    sK2_rot(w, m) -> (w′, op)

Rotation by π/2 of the bosonic momentum of an s-wave K2 class.
"""
function sK2_rot(w :: Tuple{MatsubaraFrequency, MatsubaraFrequency, BrillouinPoint{2}}, m :: KMesh)
    return (w[1], w[2], _rot(w[3], m)), Operation{ComplexF64}()
end

# symmetries in the particle-particle channel

"""
    sK2pp1(w, m) -> (w′, op)

Complex conjugation of the p-channel K2: `K2(Ω, ν, q) = K2(-Ω, -ν, -q)*`.
"""
function sK2pp1(w :: Tuple{MatsubaraFrequency, MatsubaraFrequency, BrillouinPoint{2}}, m :: KMesh)
    return (-w[1], -w[2], fold_back(-w[3], m)), Operation{ComplexF64}(sgn = false, con = true)
end

"""
    sK2pp2(w, m) -> (w′, op)

Double crossing of the p-channel K2: `K2(Ω, ν, q) = K2(Ω, Ω - ν, q)`.
"""
function sK2pp2(w :: Tuple{MatsubaraFrequency, MatsubaraFrequency, BrillouinPoint{2}}, m :: KMesh)
    return (w[1], w[1] - w[2], w[3]), Operation{ComplexF64}()
end

# symmetries in the particle-hole channel

"""
    sK2ph1(w, m) -> (w′, op)

Complex conjugation of the particle-hole K2 (same as for p).
"""
function sK2ph1(w :: Tuple{MatsubaraFrequency, MatsubaraFrequency, BrillouinPoint{2}}, m :: KMesh)
    return sK2pp1(w, m)
end

"""
    sK2ph2(w, m) -> (w′, op)

Double crossing of the particle-hole K2, combined with ν ↔ ν′:
`K2(Ω, ν, q) = K2(-Ω, Ω + ν, -q)`.
"""
function sK2ph2(w :: Tuple{MatsubaraFrequency, MatsubaraFrequency, BrillouinPoint{2}}, m :: KMesh)
    return (-w[1], w[1] + w[2], fold_back(-w[3], m)), Operation{ComplexF64}()
end

# -----------------------------------------------------------------------------
# K3(Ω, ν, ν′, q) in the s-wave approximation
# -----------------------------------------------------------------------------

"""
    sK3_ref(w, m) -> (w′, op)

Diagonal reflection of the bosonic momentum of an s-wave K3 class.
"""
function sK3_ref(
    w :: Tuple{MatsubaraFrequency, MatsubaraFrequency, MatsubaraFrequency, BrillouinPoint{2}},
    m :: KMesh
    )
    return (w[1], w[2], w[3], _ref(w[4], m)), Operation{ComplexF64}()
end

"""
    sK3_rot(w, m) -> (w′, op)

Rotation by π/2 of the bosonic momentum of an s-wave K3 class.
"""
function sK3_rot(
    w :: Tuple{MatsubaraFrequency, MatsubaraFrequency, MatsubaraFrequency, BrillouinPoint{2}},
    m :: KMesh
    )
    return (w[1], w[2], w[3], _rot(w[4], m)), Operation{ComplexF64}()
end

# symmetries in the particle-particle channel

"""
    sK3pp1(w, m) -> (w′, op)

Complex conjugation of the p-channel K3: `K3(Ω, ν, ν′, q) = K3(-Ω, -ν, -ν′, -q)*`.
"""
function sK3pp1(
    w :: Tuple{MatsubaraFrequency, MatsubaraFrequency, MatsubaraFrequency, BrillouinPoint{2}},
    m :: KMesh
    )
    return (-w[1], -w[2], -w[3], fold_back(-w[4], m)), Operation{ComplexF64}(sgn = false, con = true)
end

"""
    sK3pp2(w, m) -> (w′, op)

Exchange ν ↔ ν′ in the p-channel K3: `K3(Ω, ν, ν′, q) = K3(Ω, ν′, ν, q)`.
"""
function sK3pp2(
    w :: Tuple{MatsubaraFrequency, MatsubaraFrequency, MatsubaraFrequency, BrillouinPoint{2}},
    m :: KMesh
    )
    return (w[1], w[3], w[2], w[4]), Operation{ComplexF64}()
end

"""
    sK3pp3(w, m) -> (w′, op)

Double crossing of the p-channel K3: `K3(Ω, ν, ν′, q) = K3(Ω, Ω - ν, Ω - ν′, q)`.
"""
function sK3pp3(
    w :: Tuple{MatsubaraFrequency, MatsubaraFrequency, MatsubaraFrequency, BrillouinPoint{2}},
    m :: KMesh
    )
    return (w[1], w[1] - w[2], w[1] - w[3], w[4]), Operation{ComplexF64}()
end

# symmetries in the particle-hole channel

"""
    sK3ph1(w, m) -> (w′, op)

Complex conjugation of the particle-hole K3 (same as for p).
"""
function sK3ph1(
    w :: Tuple{MatsubaraFrequency, MatsubaraFrequency, MatsubaraFrequency, BrillouinPoint{2}},
    m :: KMesh
    )
    return sK3pp1(w, m)
end

"""
    sK3ph2(w, m) -> (w′, op)

Exchange ν ↔ ν′ in the particle-hole K3 (same as for p).
"""
function sK3ph2(
    w :: Tuple{MatsubaraFrequency, MatsubaraFrequency, MatsubaraFrequency, BrillouinPoint{2}},
    m :: KMesh
    )
    return sK3pp2(w, m)
end

"""
    sK3ph3(w, m) -> (w′, op)

Double crossing of the particle-hole K3, combined with ν ↔ ν′:
`K3(Ω, ν, ν′, q) = K3(-Ω, Ω + ν, Ω + ν′, -q)`.
"""
function sK3ph3(
    w :: Tuple{MatsubaraFrequency, MatsubaraFrequency, MatsubaraFrequency, BrillouinPoint{2}},
    m :: KMesh
    )
    return (-w[1], w[1] + w[2], w[1] + w[3], fold_back(-w[4], m)), Operation{ComplexF64}()
end

# -----------------------------------------------------------------------------
# Bubble Π(Ω, ν, q, k) (unused)
# -----------------------------------------------------------------------------
# Symmetries of a bubble that keeps its fermionic momentum k. They are only
# used in commented-out code (nonlocal/ParquetSolver.jl).

"""
    sΠ_ref(w, m) -> (w′, op)

Diagonal reflection of both momenta of a bubble Π(Ω, ν, q, k). Unused.
"""
function sΠ_ref(
    w :: Tuple{MatsubaraFrequency, MatsubaraFrequency, BrillouinPoint{2}, BrillouinPoint{2}},
    m :: KMesh
    )
    return (w[1], w[2], _ref(w[3], m), _ref(w[4], m)), Operation{ComplexF64}()
end

"""
    sΠ_rot(w, m) -> (w′, op)

Rotation by π/2 of both momenta of a bubble Π(Ω, ν, q, k). Unused.
"""
function sΠ_rot(
    w :: Tuple{MatsubaraFrequency, MatsubaraFrequency, BrillouinPoint{2}, BrillouinPoint{2}},
    m :: KMesh
    )
    return (w[1], w[2], _rot(w[3], m), _rot(w[4], m)), Operation{ComplexF64}()
end

# symmetries in the particle-particle channel

"""
    sΠ_pp1(w, m) -> (w′, op)

Complex conjugation of a particle-particle bubble. Unused.
"""
function sΠ_pp1(
    w :: Tuple{MatsubaraFrequency, MatsubaraFrequency, BrillouinPoint{2}, BrillouinPoint{2}},
    m :: KMesh
    )
    return (-w[1], -w[2], fold_back(-w[3], m), fold_back(-w[4], m)), Operation{ComplexF64}(sgn = false, con = true)
end

"""
    sΠ_pp2(w, m) -> (w′, op)

Exchange of the two propagators of a particle-particle bubble:
`Π(Ω, ν, q, k) = Π(Ω, Ω - ν, q, q - k)`. Unused.
"""
function sΠ_pp2(
    w :: Tuple{MatsubaraFrequency, MatsubaraFrequency, BrillouinPoint{2}, BrillouinPoint{2}},
    m :: KMesh
    )
    return (w[1], w[1] - w[2], w[3], fold_back(w[3] - w[4], m)), Operation{ComplexF64}()
end

# symmetries in the particle-hole channel

"""
    sΠ_ph1(w, m) -> (w′, op)

Complex conjugation of a particle-hole bubble (same as for pp). Unused.
"""
function sΠ_ph1(
    w :: Tuple{MatsubaraFrequency, MatsubaraFrequency, BrillouinPoint{2}, BrillouinPoint{2}},
    m :: KMesh
    )
    return sΠ_pp1(w, m)
end

"""
    sΠ_ph2(w, m) -> (w′, op)

Exchange of the two propagators of a particle-hole bubble:
`Π(Ω, ν, q, k) = Π(-Ω, Ω + ν, -q, q + k)`. Unused.
"""
function sΠ_ph2(
    w :: Tuple{MatsubaraFrequency, MatsubaraFrequency, BrillouinPoint{2}, BrillouinPoint{2}},
    m :: KMesh
    )
    return (-w[1], w[1] + w[2], fold_back(-w[3], m), fold_back(w[3] + w[4], m)), Operation{ComplexF64}()
end
