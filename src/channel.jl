# =============================================================================
# One channel's reducible vertex Φ_r in the asymptotic decomposition (local).
#
# The vertex Φ_r reducible in channel r ∈ {a, p, t} depends on three
# frequencies (Ω, ν, ν′), given in channel r's parametrization (convention.jl).
# Storing it on a large three-dimensional grid is too expensive. Instead, Φ_r is
# split by how it behaves when the fermionic frequencies become large (Lihm
# 2025, Eq. (S7)):
#
#   Φ_r(Ω, ν, ν′) = K1_r(Ω) + K2_r(Ω, ν) + K̄2_r(Ω, ν′) + K3_r(Ω, ν, ν′)
#
#   K1   what remains for ν, ν′ → ∞; depends only on the bosonic Ω. It is
#        proportional to the susceptibility of the channel (Eq. (S19a)).
#   K2   what remains for ν′ → ∞, minus K1. It is tied to the three-leg
#        (Hedin) vertex (Eq. (S19c)).
#   K̄2   the same for ν → ∞. It is not stored: by the ν ↔ ν′ symmetry of the
#        vertex, K̄2(Ω, ν′) = K2(Ω, ν′), read from the K2 array.
#   K3   the rest. It decays in every frequency direction.
#
# Each class is stored on its own frequency box, with numK1 ≥ numK2 ≥ numK3
# positive frequencies, and is zero outside it. K1 and K2 have few arguments
# and capture the slow high-frequency decay of Φ_r, so they get large boxes.
# K3 decays fast, so a small box suffices. Cutting each class at its box is
# the only approximation in this representation.
#
# Paper ↔ code: ω (bosonic) ↔ Ω; ν′ ↔ νp; K̄2 ↔ K2' (same array as K2).
# A Channel holds one Φ_r; a vertex stores three of them in its fields
# `γa`, `γp`, `γt` (vertex.jl). Lattice versions: nonlocal*/channel.jl.
# =============================================================================

"""
    AbstractReducibleVertex{Q}

Supertype of the containers for one channel's reducible vertex `Φ_r` with
number type `Q`: the local [`Channel`](@ref) and the lattice versions
`NL_Channel`, `NL2_Channel`, `NL3_Channel`. All of them store the asymptotic
classes in fields `K1`, `K2`, `K3`.
"""
abstract type AbstractReducibleVertex{Q}; end

# number type of the stored data
Base.eltype(::Type{<: AbstractReducibleVertex{Q}}) where {Q} = Q

# -----------------------------------------------------------------------------
# Channel struct
# -----------------------------------------------------------------------------

"""
    Channel{Q} <: AbstractReducibleVertex{Q}

The reducible vertex `Φ_r` of one channel, without momentum dependence, stored
as its asymptotic classes (Lihm 2025, Eq. (S7)):

- `K1`: K1(Ω), on a bosonic mesh with `numK1` positive frequencies.
- `K2`: K2(Ω, ν), on a (bosonic, fermionic) mesh of size `numK2`. Also used
  for K̄2(Ω, ν′).
- `K3`: K3(Ω, ν, ν′), on a (bosonic, fermionic, fermionic) mesh of size
  `numK3`; ν and ν′ share one fermionic mesh.

Constructors:
- `Channel(K1, K2, K3)`: from existing mesh functions, without checks.
- `Channel(T, numK1, numK2, numK3, Q = ComplexF64)`: all classes zero, at
  temperature `T`. Checks `numK1 ≥ numK2 ≥ numK3`.
"""
struct Channel{Q <: Number} <: AbstractReducibleVertex{Q}
    K1 :: MF_K1{Q}
    K2 :: MF_K2{Q}
    K3 :: MF_K3{Q}

    # Constructor from existing mesh functions.
    function Channel(
        K1 :: MF_K1{Q},
        K2 :: MF_K2{Q},
        K3 :: MF_K3{Q},
        )  :: Channel{Q} where {Q}

        return new{Q}(K1, K2, K3)
    end

    # Constructor from temperature and box sizes; all classes start at zero.
    function Channel(
        T     :: Float64,
        numK1 :: Int64,
        numK2 :: NTuple{2, Int64},
        numK3 :: NTuple{2, Int64},
              :: Type{Q} = ComplexF64,
        ) where {Q}

        # K1: depends only on the bosonic frequency Ω
        mK1Ω = MatsubaraMesh(T, numK1, Boson) # bosonic mesh for Ω, 2*numK1 - 1 points
        K1 = MeshFunction(mK1Ω; data_t = Q)
        set!(K1, 0)

        # K2: depends on (Ω, ν)
        @assert numK1 >= numK2[1] "No. bosonic frequencies in K1 must be larger than no. bosonic frequencies in K2"
        @assert numK1 >= numK2[2] "No. bosonic frequencies in K1 must be larger than no. fermionic frequencies in K2"
        mK2Ω = MatsubaraMesh(T, numK2[1], Boson)    # bosonic mesh for Ω, 2*numK2[1] - 1 points
        mK2ν = MatsubaraMesh(T, numK2[2], Fermion)  # fermionic mesh for ν, 2*numK2[2] points
        K2   = MeshFunction(mK2Ω, mK2ν; data_t = Q)
        set!(K2, 0)

        # K3: depends on (Ω, ν, ν′); ν and ν′ use the same mesh
        @assert all(numK2 .>= numK3) "Number of frequencies in K2 must be larger than in K3"
        mK3Ω = MatsubaraMesh(T, numK3[1], Boson)    # bosonic mesh for Ω, 2*numK3[1] - 1 points
        mK3ν = MatsubaraMesh(T, numK3[2], Fermion)  # fermionic mesh for ν and ν′, 2*numK3[2] points
        K3   = MeshFunction(mK3Ω, mK3ν, mK3ν; data_t = Q)
        set!(K3, 0)

        return new{Q}(K1, K2, K3) :: Channel{Q}
    end
end

# -----------------------------------------------------------------------------
# Getters
# -----------------------------------------------------------------------------

"""
    temperature(γ::AbstractReducibleVertex) -> Float64

Temperature T = 1/β of the Matsubara meshes, read from the K1 mesh.
"""
function MatsubaraFunctions.temperature(
    γ :: AbstractReducibleVertex
    ) :: Float64

    return MatsubaraFunctions.temperature(meshes(γ.K1, Val(1)))
end

"""
    numK1(γ::AbstractReducibleVertex) -> Int64

Number of positive bosonic frequencies in the K1 box.
"""
function numK1(
    γ :: AbstractReducibleVertex
    ) :: Int64

    return N(meshes(γ.K1, Val(1)))
end

"""
    numK2(γ::AbstractReducibleVertex) -> (nΩ, nν)

Number of positive bosonic and fermionic frequencies in the K2 box.
"""
function numK2(
    γ :: AbstractReducibleVertex
    ) :: NTuple{2, Int64}

    return N(meshes(γ.K2, Val(1))), N(meshes(γ.K2, Val(2)))
end

"""
    numK3(γ::AbstractReducibleVertex) -> (nΩ, nν)

Number of positive bosonic and fermionic frequencies in the K3 box. ν and ν′
share the same fermionic mesh.
"""
function numK3(
    γ :: AbstractReducibleVertex
    ) :: NTuple{2, Int64}

    return N(meshes(γ.K3, Val(1))), N(meshes(γ.K3, Val(2)))
end

# -----------------------------------------------------------------------------
# Setters
# -----------------------------------------------------------------------------

"""
    set!(γ1::AbstractReducibleVertex, γ2::AbstractReducibleVertex) -> nothing
    set!(γ1::AbstractReducibleVertex, val::Number) -> nothing

Overwrite K1, K2 and K3 of `γ1` with those of `γ2` (same meshes required), or
set every entry to the number `val`, e.g. `set!(γ, 0)`.
"""
function MatsubaraFunctions.set!(
    γ1 :: AbstractReducibleVertex,
    γ2 :: AbstractReducibleVertex
    )  :: Nothing

    set!(γ1.K1, γ2.K1)
    set!(γ1.K2, γ2.K2)
    set!(γ1.K3, γ2.K3)

    return nothing
end

# set! with a number, see the docstring above
function MatsubaraFunctions.set!(
    γ1  :: AbstractReducibleVertex{Q},
    val :: Number,
    )   :: Nothing where {Q}

    set!(γ1.K1, Q(val))
    set!(γ1.K2, Q(val))
    set!(γ1.K3, Q(val))

    return nothing
end

# -----------------------------------------------------------------------------
# Comparison
# -----------------------------------------------------------------------------

"""
    ==(γ1::AbstractReducibleVertex, γ2::AbstractReducibleVertex) -> Bool

True if K1, K2 and K3 are equal entry by entry.
"""
function Base.:(==)(
    γ1 :: AbstractReducibleVertex,
    γ2 :: AbstractReducibleVertex
    )  :: Bool
    return (γ1.K1 == γ2.K1) && (γ1.K2 == γ2.K2) && (γ1.K3 == γ2.K3)
end

# -----------------------------------------------------------------------------
# Arithmetic
# -----------------------------------------------------------------------------

"""
    add!(γ1::AbstractReducibleVertex, γ2::AbstractReducibleVertex) -> nothing

In place `γ1 += γ2`, class by class.
"""
function MatsubaraFunctions.add!(
    γ1 :: AbstractReducibleVertex,
    γ2 :: AbstractReducibleVertex
    )  :: Nothing

    add!(γ1.K1, γ2.K1)
    add!(γ1.K2, γ2.K2)
    add!(γ1.K3, γ2.K3)

    return nothing
end

"""
    mult_add!(γ1::AbstractReducibleVertex, γ2::AbstractReducibleVertex, val) -> nothing

In place `γ1 += val * γ2`, class by class.
"""
function MatsubaraFunctions.mult_add!(
    γ1 :: AbstractReducibleVertex,
    γ2 :: AbstractReducibleVertex,
    val :: Number,
    )  :: Nothing

    mult_add!(γ1.K1, γ2.K1, val)
    mult_add!(γ1.K2, γ2.K2, val)
    mult_add!(γ1.K3, γ2.K3, val)

    return nothing
end

# -----------------------------------------------------------------------------
# Size, norm and conversion to a flat vector
# -----------------------------------------------------------------------------
# The nonlinear solvers (solve.jl, mfRG.jl) see the vertex as one long vector
# of unknowns. flatten / unflatten! convert between the two forms.

"""
    length(γ::AbstractReducibleVertex) -> Int64

Total number of stored values: the sizes of the K1, K2 and K3 arrays added.
"""
function Base.length(
    γ :: AbstractReducibleVertex
    ) :: Int64

    return length(γ.K1.data) + length(γ.K2.data) + length(γ.K3.data)
end

"""
    absmax(γ::AbstractReducibleVertex) -> Float64

Largest absolute value over K1, K2 and K3. Used to monitor convergence.
"""
function MatsubaraFunctions.absmax(
    γ :: AbstractReducibleVertex
    ) :: Float64

    return max(absmax(γ.K1), absmax(γ.K2), absmax(γ.K3))
end

"""
    flatten!(γ::AbstractReducibleVertex, x) -> nothing

Write all data into the vector `x`: first K1, then K2, then K3. The length of
`x` must equal `length(γ)`.
"""
function MatsubaraFunctions.flatten!(
    γ :: AbstractReducibleVertex,
    x :: AbstractVector
    ) :: Nothing

    offset = 0
    lenK1  = length(γ.K1.data)
    lenK2  = length(γ.K2.data)
    lenK3  = length(γ.K3.data)

    flatten!(γ.K1, @view x[1 + offset : offset + lenK1])
    offset += lenK1

    flatten!(γ.K2, @view x[1 + offset : offset + lenK2])
    offset += lenK2

    flatten!(γ.K3, @view x[1 + offset : offset + lenK3])
    offset += lenK3

    # ensure that dimensions of channel are compatible with length of vector
    @assert offset == length(x) "Dimension mismatch between channel and target vector"
    return nothing
end

"""
    flatten(γ::AbstractReducibleVertex) -> Vector

All data in one new vector, in the order K1, K2, K3 (see `flatten!`).
"""
function MatsubaraFunctions.flatten(
    γ :: AbstractReducibleVertex{Q}
    ) :: Vector{Q} where {Q}

    x = Array{Q}(undef, length(γ))
    flatten!(γ, x)

    return x
end

"""
    unflatten!(γ::AbstractReducibleVertex, x) -> nothing

Read K1, K2 and K3 back from the vector `x`, written by `flatten!`.
"""
function MatsubaraFunctions.unflatten!(
    γ :: AbstractReducibleVertex,
    x :: AbstractVector
    ) :: Nothing

    offset = 0
    lenK1  = length(γ.K1.data)
    lenK2  = length(γ.K2.data)
    lenK3  = length(γ.K3.data)

    unflatten!(γ.K1, @view x[1 + offset : offset + lenK1])
    offset += lenK1

    unflatten!(γ.K2, @view x[1 + offset : offset + lenK2])
    offset += lenK2

    unflatten!(γ.K3, @view x[1 + offset : offset + lenK3])
    offset += lenK3

    # ensure that dimensions of channel are compatible with length of vector
    @assert offset == length(x) "Dimension mismatch between channel and target vector"
    return nothing
end

# -----------------------------------------------------------------------------
# Copy
# -----------------------------------------------------------------------------

"""
    copy(γ::Channel) -> Channel

Independent copy, with new arrays for K1, K2 and K3.
"""
function Base.:copy(
    γ :: Channel{Q}
    ) :: Channel{Q} where {Q}
    return Channel(copy(γ.K1), copy(γ.K2), copy(γ.K3))
end

# -----------------------------------------------------------------------------
# Evaluation
# -----------------------------------------------------------------------------
# A Channel is called like a function to get Φ_r at given frequencies. There
# are four methods: ν and ν′ can each be finite or νInf (infinite).

"""
    (γ::Channel)(Ω, ν, νp; K1 = true, K2 = true, K3 = true) -> value

Evaluate the reducible vertex of this channel at frequencies given in the
channel's own parametrization (Lihm 2025, Eq. (S7)):

    Φ_r(Ω, ν, ν′) = K1(Ω) + K2(Ω, ν) + K̄2(Ω, ν′) + K3(Ω, ν, ν′)

K̄2 is read from the K2 array. A class whose arguments lie outside its box
contributes zero. The keywords `K1`, `K2`, `K3` switch single classes off (`K2`
covers K2 and K̄2).

`ν` and/or `νp` may be [`νInf`](@ref). The classes that need that frequency to
be finite then drop out, which gives the high-frequency limits used in the
Bethe-Salpeter equations (Eq. (S22)):

    Φ_r(Ω, ∞, ν′) = K1(Ω) + K̄2(Ω, ν′)
    Φ_r(Ω, ν, ∞)  = K1(Ω) + K2(Ω, ν)
    Φ_r(Ω, ∞, ∞)  = K1(Ω)
"""
@inline function (γ :: Channel{Q})(
    Ω  :: MatsubaraFrequency,
    ν  :: MatsubaraFrequency,
    νp :: MatsubaraFrequency
    ;
    K1 :: Bool = true,
    K2 :: Bool = true,
    K3 :: Bool = true,
    )  :: Q where {Q}

    val = zero(Q)

    # check if Ω is in-bounds for K1
    if is_inbounds(Ω, meshes(γ.K1, Val(1)))
        # add K1[Ω] if K1=true
        K1 && (val += γ.K1[Ω])

        # check if Ω is in-bounds for K2. Nested because numK1 >= numK2
        if is_inbounds(Ω, meshes(γ.K2, Val(1)))

            # check if ν, ν' are in-bounds for K2
            ν1_inbounds = is_inbounds(ν, meshes(γ.K2, Val(2)))
            ν2_inbounds = is_inbounds(νp, meshes(γ.K2, Val(2)))

            if ν1_inbounds && ν2_inbounds
                # Both fermionic arguments in range: add K2(Ω,ν) + K̄2(Ω,ν′) + K3(Ω,ν,ν′) if K2=true and K3=true respectively
                K2 && (val += γ.K2[Ω, ν] + γ.K2[Ω, νp])
                K3 && (val += γ.K3(Ω, ν, νp)) # bounds check for Ω, ν, νp handled in MeshFunction()

            elseif ν1_inbounds
                # Only ν in range: add K2(Ω, ν) if K2=true; K3 requires both legs to be in range
                K2 && (val += γ.K2[Ω, ν])

            elseif ν2_inbounds
                # Only ν′ in range: add K̄2(Ω, ν′) if K2=true
                K2 && (val += γ.K2[Ω, νp])
            end
        end
    end

    return val
end

# Method for ν = ∞: only K1(Ω) + K̄2(Ω, ν′) remain.
@inline function (γ :: Channel{Q})(
    Ω  :: MatsubaraFrequency,
    ν  :: InfiniteMatsubaraFrequency,
    νp :: MatsubaraFrequency
    ;
    K1 :: Bool = true,
    K2 :: Bool = true,
    K3 :: Bool = true,
    )  :: Q where {Q}

    val = zero(Q)

    if K1
        if is_inbounds(Ω, meshes(γ.K1, Val(1)))
            val += γ.K1[Ω]

            if K2 && is_inbounds(Ω, meshes(γ.K2, Val(1))) && is_inbounds(νp, meshes(γ.K2, Val(2)))
                val += γ.K2[Ω, νp]
            end
        end
    else
        # K1 not included
        if K2 && is_inbounds(Ω, meshes(γ.K2, Val(1))) && is_inbounds(νp, meshes(γ.K2, Val(2)))
            val += γ.K2[Ω, νp]
        end
    end

    return val
end

# Method for ν′ = ∞: only K1(Ω) + K2(Ω, ν) remain.
@inline function (γ :: Channel{Q})(
    Ω  :: MatsubaraFrequency,
    ν  :: MatsubaraFrequency,
    νp :: InfiniteMatsubaraFrequency
    ;
    K1 :: Bool = true,
    K2 :: Bool = true,
    K3 :: Bool = true,
    )  :: Q where {Q}

    val = zero(Q)

    if K1
        if is_inbounds(Ω, meshes(γ.K1, Val(1)))
            K1 && (val += γ.K1[Ω])

            if K2 && is_inbounds(Ω, meshes(γ.K2, Val(1))) && is_inbounds(ν, meshes(γ.K2, Val(2)))
                val += γ.K2[Ω, ν]
            end
        end
    else
        # K1 not included
        if K2 && is_inbounds(Ω, meshes(γ.K2, Val(1))) && is_inbounds(ν, meshes(γ.K2, Val(2)))
            val += γ.K2[Ω, ν]
        end
    end

    return val
end


# Method for ν = ν′ = ∞: only K1(Ω) remains.
@inline function (γ :: Channel{Q})(
    Ω  :: MatsubaraFrequency,
    ν  :: InfiniteMatsubaraFrequency,
    νp :: InfiniteMatsubaraFrequency
    ;
    K1 :: Bool = true,
    K2 :: Bool = true,
    K3 :: Bool = true,
    )  :: Q where {Q}

    val = zero(Q)

    if K1 && is_inbounds(Ω, meshes(γ.K1, Val(1)))
        val += γ.K1[Ω]
    end

    return val
end

# -----------------------------------------------------------------------------
# From cumulative to separate classes
# -----------------------------------------------------------------------------

"""
    reduce!(γ::Channel) -> nothing

Turn "cumulative" classes into the separate classes of Eq. (S7), in place.
Before the call, K2 holds the full limit `Φ_r(Ω, ν, ∞) = K1 + K2` and K3 holds
`Φ_r` itself. Afterwards

    K2(Ω, ν)     ← K2(Ω, ν) - K1(Ω)
    K3(Ω, ν, ν′) ← K3(Ω, ν, ν′) - K1(Ω) - K2(Ω, ν) - K2(Ω, ν′)

(the last line uses the already reduced K2). Not used by the solvers; the
tests use it to build channels from a known `Φ_r`.
"""
function reduce!(
    γ :: Channel
    ) :: Nothing

    # K2(Ω, ν) -= K1(Ω) on the K2 box
    for iΩ in eachindex(meshes(γ.K2, Val(1)))
        Ω = value(meshes(γ.K2, Val(1))[iΩ])
        K1val = γ.K1[Ω]

        for iν in eachindex(meshes(γ.K2, Val(2)))
            ν = value(meshes(γ.K2, Val(2))[iν])
            γ.K2[Ω, ν] -= K1val
        end
    end

    # K3(Ω, ν, ν′) -= K1(Ω) + K2(Ω, ν) + K̄2(Ω, ν′) on the K3 box
    for iΩ in eachindex(meshes(γ.K3, Val(1)))
        Ω = value(meshes(γ.K3, Val(1))[iΩ])
        K1val = γ.K1[Ω]

        for iν in eachindex(meshes(γ.K3, Val(2)))
            ν = value(meshes(γ.K3, Val(2))[iν])
            K2val = γ.K2[Ω, ν]

            for iνp in eachindex(meshes(γ.K3, Val(3)))
                νp = value(meshes(γ.K3, Val(3))[iνp])
                γ.K3[Ω, ν, νp] -= K1val + K2val + γ.K2[Ω, νp]
            end
        end
    end

    return nothing
end

# -----------------------------------------------------------------------------
# HDF5 input and output
# -----------------------------------------------------------------------------

"""
    save!(file, label, γ::AbstractReducibleVertex) -> nothing

Write K1, K2 and K3 to the HDF5 groups `label/K1`, `label/K2`, `label/K3`.
"""
function MatsubaraFunctions.save!(
    file  :: HDF5.File,
    label :: String,
    γ     :: AbstractReducibleVertex
    )     :: Nothing

    MatsubaraFunctions.save!(file, label * "/K1", γ.K1)
    MatsubaraFunctions.save!(file, label * "/K2", γ.K2)
    MatsubaraFunctions.save!(file, label * "/K3", γ.K3)

    return nothing
end

"""
    load_channel(T, file, label) -> T

Read K1, K2 and K3 from the groups below `label` (written by `save!`) and build
a channel of type `T`, e.g. `Channel` or `NL2_Channel`.
"""
function load_channel(
          :: Type{T},
    file  :: HDF5.File,
    label :: String
    )     :: T where {T <: AbstractReducibleVertex}

    K1 = load_mesh_function(file, label * "/K1")
    K2 = load_mesh_function(file, label * "/K2")
    K3 = load_mesh_function(file, label * "/K3")

    return T(K1, K2, K3)
end
