# =============================================================================
# Symmetries of the local self-energy and of the local vertex classes 
# K1, K2 and K3.
#
# Each function takes a tuple w of Matsubara frequencies and returns the
# frequencies w′ of a symmetry-related point, together with an `Operation`
# (sgn: change the sign, con: complex conjugate) that turns the value at one
# point into the value at the other. A symmetry group
# (MatsubaraFunctions.SymmetryGroup) uses them to compute only one point per
# set of related points and fill in the others (see init_sym_grp! in
# ParquetSolver.jl).
#
# The symmetries used (Lihm 2025, Eq. (S4) and the text below it):
#   complex conjugation  Σ(-ν) = Σ(ν)*. For a vertex, flipping the sign of all
#                        frequencies conjugates it.
#   double crossing      exchanging both the two incoming and the two outgoing
#                        legs leaves the vertex unchanged (Eq. (S4c)).
#   ν ↔ ν′               time reversal combined with complex conjugation; it
#                        needs SU(2) symmetry, real hopping and no field.
#
# These take a different form in each channel's frequency parametrization
# (convention.jl). So there are separate functions for the particle-particle
# classes (pp: the p channel) and the particle-hole classes (ph: the a and t
# channels). Arguments are (Ω, ν, ν′) in that parametrization: w[1] is the
# bosonic Ω, w[2] and w[3] the fermionic ν and ν′.
#
# Every function takes ::Type{Q}, the number type of the mesh function it acts
# on, so that the returned Operation{Q} matches it.
# =============================================================================

# -----------------------------------------------------------------------------
# Self-energy
# -----------------------------------------------------------------------------

"""
    sΣ(w, ::Type{Q}) -> (w′, op)

Complex-conjugation symmetry of the self-energy, `Σ(-ν) = Σ(ν)*`. It holds
because Σ(τ) is real in imaginary time. The code stores iΣ, for which it
reads `iΣ(-ν) = -[iΣ(ν)]*`: map ν → -ν, change the sign and conjugate.
"""
function sΣ(
    w :: NTuple{1, MatsubaraFrequency},
    :: Type{Q},
    ) :: Tuple{NTuple{1, MatsubaraFrequency}, Operation{Q}} where {Q <: Number}

    # Maps w -> -w, flips sign, and applies conjugation.
    # We store iΣ, so the symmetry Σ -> Σ* becomes iΣ -> -(iΣ)*.
    return (-w[1],), Operation{Q}(sgn = true, con = true)
end

# -----------------------------------------------------------------------------
# Particle-particle (p channel) symmetries
# -----------------------------------------------------------------------------
# In the p channel, Ω is the total frequency of the two incoming particles.
# Exchanging both pairs of legs maps (ν, ν′) to (Ω - ν, Ω - ν′).

"""
    sK1pp(w, ::Type{Q}) -> (w′, op)

Complex conjugation for the p-channel K1 class: `K1(Ω) = K1(-Ω)*`.
"""
function sK1pp(
    w :: NTuple{1, MatsubaraFrequency},
    :: Type{Q},
    ) :: Tuple{NTuple{1, MatsubaraFrequency}, Operation{Q}} where {Q <: Number}

    return (-w[1],), Operation{Q}(sgn = false, con = true)
end

"""
    sK2pp1(w, ::Type{Q}) -> (w′, op)

Complex conjugation for the p-channel K2 class: `K2(Ω, ν) = K2(-Ω, -ν)*`.
"""
function sK2pp1(
    w :: NTuple{2, MatsubaraFrequency},
    :: Type{Q},
    ) :: Tuple{NTuple{2, MatsubaraFrequency}, Operation{Q}} where {Q <: Number}

    return (-w[1], -w[2]), Operation{Q}(sgn = false, con = true)
end

"""
    sK2pp2(w, ::Type{Q}) -> (w′, op)

Double crossing for the p-channel K2 class: `K2(Ω, ν) = K2(Ω, Ω - ν)`.
This is Eq. (S4c) with the outer frequency ν′ sent to infinity.
"""
function sK2pp2(
    w :: NTuple{2, MatsubaraFrequency},
    :: Type{Q},
    ) :: Tuple{NTuple{2, MatsubaraFrequency}, Operation{Q}} where {Q <: Number}

    return (w[1], w[1] - w[2]), Operation{Q}()
end

"""
    sK3pp1(w, ::Type{Q}) -> (w′, op)

Complex conjugation for the p-channel K3 class:
`K3(Ω, ν, ν′) = K3(-Ω, -ν, -ν′)*`.
"""
function sK3pp1(
    w :: NTuple{3, MatsubaraFrequency},
    :: Type{Q},
    ) :: Tuple{NTuple{3, MatsubaraFrequency}, Operation{Q}} where {Q <: Number}

    return (-w[1], -w[2], -w[3]), Operation{Q}(sgn = false, con = true)
end

"""
    sK3pp2(w, ::Type{Q}) -> (w′, op)

Exchange of the two fermionic frequencies for the p-channel K3 class:
`K3(Ω, ν, ν′) = K3(Ω, ν′, ν)`.
"""
function sK3pp2(
    w :: NTuple{3, MatsubaraFrequency},
    :: Type{Q},
    ) :: Tuple{NTuple{3, MatsubaraFrequency}, Operation{Q}} where {Q <: Number}

    return (w[1], w[3], w[2]), Operation{Q}()
end

"""
    sK3pp3(w, ::Type{Q}) -> (w′, op)

Double crossing for the p-channel K3 class (Eq. (S4c)):
`K3(Ω, ν, ν′) = K3(Ω, Ω - ν, Ω - ν′)`.
"""
function sK3pp3(
    w :: NTuple{3, MatsubaraFrequency},
    :: Type{Q},
    ) :: Tuple{NTuple{3, MatsubaraFrequency}, Operation{Q}} where {Q <: Number}

    return (w[1], w[1] - w[2], w[1] - w[3]), Operation{Q}()
end

# -----------------------------------------------------------------------------
# Particle-hole (a and t channel) symmetries
# -----------------------------------------------------------------------------
# Complex conjugation and ν ↔ ν′ have the same form as in the p channel, so
# those functions just call the pp versions. The double crossing (Eq. (S4c))
# reads (Ω, ν, ν′) → (-Ω, ν′ + Ω, ν + Ω) in the a and t parametrizations.
# Combined with ν ↔ ν′ this gives the map (-Ω, ν + Ω, ν′ + Ω) used below.

"""
    sK1ph(w, ::Type{Q}) -> (w′, op)

Complex conjugation for the particle-hole K1 class: `K1(Ω) = K1(-Ω)*`.
"""
function sK1ph(
    w :: NTuple{1, MatsubaraFrequency},
    :: Type{Q},
    ) :: Tuple{NTuple{1, MatsubaraFrequency}, Operation{Q}} where {Q <: Number}

    return sK1pp(w, Q)
end

"""
    sK2ph1(w, ::Type{Q}) -> (w′, op)

Complex conjugation for the particle-hole K2 class: `K2(Ω, ν) = K2(-Ω, -ν)*`.
"""
function sK2ph1(
    w :: NTuple{2, MatsubaraFrequency},
    :: Type{Q},
    ) :: Tuple{NTuple{2, MatsubaraFrequency}, Operation{Q}} where {Q <: Number}

    return sK2pp1(w, Q)
end

"""
    sK2ph2(w, ::Type{Q}) -> (w′, op)

Double crossing for the particle-hole K2 class: `K2(Ω, ν) = K2(-Ω, Ω + ν)`.
This is Eq. (S4c) with the outer frequency ν′ sent to infinity, combined with
ν ↔ ν′.
"""
function sK2ph2(
    w :: NTuple{2, MatsubaraFrequency},
    :: Type{Q},
    ) :: Tuple{NTuple{2, MatsubaraFrequency}, Operation{Q}} where {Q <: Number}

    return (-w[1], w[1] + w[2]), Operation{Q}()
end

"""
    sK3ph1(w, ::Type{Q}) -> (w′, op)

Complex conjugation for the particle-hole K3 class:
`K3(Ω, ν, ν′) = K3(-Ω, -ν, -ν′)*`.
"""
function sK3ph1(
    w :: NTuple{3, MatsubaraFrequency},
    :: Type{Q},
    ) :: Tuple{NTuple{3, MatsubaraFrequency}, Operation{Q}} where {Q <: Number}

    return sK3pp1(w, Q)
end

"""
    sK3ph2(w, ::Type{Q}) -> (w′, op)

Exchange of the two fermionic frequencies for the particle-hole K3 class:
`K3(Ω, ν, ν′) = K3(Ω, ν′, ν)`.
"""
function sK3ph2(
    w :: NTuple{3, MatsubaraFrequency},
    :: Type{Q},
    ) :: Tuple{NTuple{3, MatsubaraFrequency}, Operation{Q}} where {Q <: Number}

    return sK3pp2(w, Q)
end

"""
    sK3ph3(w, ::Type{Q}) -> (w′, op)

Double crossing for the particle-hole K3 class, combined with ν ↔ ν′:
`K3(Ω, ν, ν′) = K3(-Ω, Ω + ν, Ω + ν′)`.
"""
function sK3ph3(
    w :: NTuple{3, MatsubaraFrequency},
    :: Type{Q},
    ) :: Tuple{NTuple{3, MatsubaraFrequency}, Operation{Q}} where {Q <: Number}

    return (-w[1], w[1] + w[2], w[1] + w[3]), Operation{Q}()
end
