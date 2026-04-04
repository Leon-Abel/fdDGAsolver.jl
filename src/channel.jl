# channel.jl
#
# Defines the asymptotic decomposition of the 2-particle reducible vertex Φ_r in a single
# diagrammatic channel r ∈ {a, p, t} (antiparallel, parallel, transverse-antiparallel).
#
# The storage of Φ_r is made tractable by the asymptotic (K1/K2/K3) decomposition,
# which exploits the frequency decay of Φ_r(Ω, ν, ν') as ν or ν' → ∞:
#
#   Φ_r(Ω, ν, ν') ≈ K1_r(Ω) + K2_r(Ω, ν) + K2'_r(Ω, ν') + K3_r(Ω, ν, ν')
#
# where K1 retains only the bosonic frequency Ω (transfer frequency), K2 and K2' add
# one fermionic leg each, and K3 is the remainder carrying full (Ω, ν, ν') dependence.
# The K3 grid can be truncated at a much smaller frequency cutoff than the K2 or K1 grids.

"""
    abstract type AbstractReducibleVertex{Q}
 
Abstract supertype for all representations of a channel-r two-particle reducible vertex Φ_r.
"""
abstract type AbstractReducibleVertex{Q}; end

# Expose the scalar type of a reducible vertex.
Base.eltype(::Type{<: AbstractReducibleVertex{Q}}) where {Q} = Q

# --------------------------------------------------------------------------- #
# Channel struct
# --------------------------------------------------------------------------- #

"""
    struct Channel{Q <: Number} <: AbstractReducibleVertex{Q}
 
Concrete representation of the 2-particle reducible vertex Φ_r in the asymptotic
decomposition for a single diagrammatic channel r.
 
The frequency mesh sizes satisfy the hierarchy `numK1 ≥ numK2 ≥ numK3` to respect the
fact that lower-class vertices require a larger frequency window for convergence.
"""
struct Channel{Q <: Number} <: AbstractReducibleVertex{Q}
    K1 :: MF_K1{Q}
    K2 :: MF_K2{Q}
    K3 :: MF_K3{Q}

    """
        Channel(K1, K2, K3)
 
    Construct a `Channel` directly from pre-allocated `MeshFunction` objects for each
    asymptotic class. No consistency checks are performed on the meshes.
    """
    function Channel(
        K1 :: MF_K1{Q},
        K2 :: MF_K2{Q},
        K3 :: MF_K3{Q},
        )  :: Channel{Q} where {Q}

        return new{Q}(K1, K2, K3)
    end

    """
        Channel(T, numK1, numK2, numK3 [, Q])
 
    Construct and zero-initialize a `Channel` from mesh size parameters.
 
    The frequency hierarchy `numK1 ≥ numK2 ≥ numK3` is enforced by assertions,
    consistent with the asymptotic decay of each class.
    """
    function Channel(
        T     :: Float64,
        numK1 :: Int64,
        numK2 :: NTuple{2, Int64},
        numK3 :: NTuple{2, Int64},
              :: Type{Q} = ComplexF64,
        ) where {Q}

        # K1: depends only on bosonic transfer frequency Ω
        mK1Ω = MatsubaraMesh(T, numK1, Boson) # bosonic mesh for transfer frequency Ω, 2*numK1 - 1 Ω-points
        K1 = MeshFunction(mK1Ω; data_t = Q)
        set!(K1, 0)

        # K2: depends on (Ω, ν)
        @assert numK1 >= numK2[1] "No. bosonic frequencies in K1 must be larger than no. bosonic frequencies in K2"
        @assert numK1 >= numK2[2] "No. bosonic frequencies in K1 must be larger than no. fermionic frequencies in K2"
        mK2Ω = MatsubaraMesh(T, numK2[1], Boson)    # bosonic mesh for transfer frequency Ω, 2*numK2[1] - 1 Ω-points
        mK2ν = MatsubaraMesh(T, numK2[2], Fermion)  # fermionic mesh for frequency ν, 2*numK2[2] ν-points
        K2   = MeshFunction(mK2Ω, mK2ν; data_t = Q)
        set!(K2, 0)

        # K3: depends on (Ω, ν, ν'); symmetric in the two fermionic legs (same mesh for both)
        @assert all(numK2 .>= numK3) "Number of frequencies in K2 must be larger than in K3"
        mK3Ω = MatsubaraMesh(T, numK3[1], Boson)    # bosonic mesh for transfer frequency Ω, 2*numK3[1] - 1 Ω-points
        mK3ν = MatsubaraMesh(T, numK3[2], Fermion)  # fermionic mesh for frequencies ν, ν', 2*numK3[2] ν-points
        K3   = MeshFunction(mK3Ω, mK3ν, mK3ν; data_t = Q)
        set!(K3, 0)

        return new{Q}(K1, K2, K3) :: Channel{Q}
    end
end

# --------------------------------------------------------------------------- #
# Getter methods
# --------------------------------------------------------------------------- #

"""
    temperature(γ::AbstractReducibleVertex) -> Float64
 
Return the temperature `T` of the Matsubara grid underlying the reducible vertex,
extracted from the bosonic mesh of K1.
"""
function MatsubaraFunctions.temperature(
    γ :: AbstractReducibleVertex
    ) :: Float64

    return MatsubaraFunctions.temperature(meshes(γ.K1, Val(1)))
end

"""
    numK1(γ::AbstractReducibleVertex) -> Int64
 
Return the number of bosonic Matsubara frequencies in the K1 mesh.
"""
function numK1(
    γ :: AbstractReducibleVertex
    ) :: Int64

    return N(meshes(γ.K1, Val(1)))
end

"""
    numK2(γ::AbstractReducibleVertex) -> NTuple{2, Int64}
 
Return the mesh sizes `(nΩ, nν)` of the K2 vertex, i.e., the number of bosonic and
fermionic Matsubara frequencies, respectively.
"""
function numK2(
    γ :: AbstractReducibleVertex
    ) :: NTuple{2, Int64}

    return N(meshes(γ.K2, Val(1))), N(meshes(γ.K2, Val(2)))
end

"""
    numK3(γ::AbstractReducibleVertex) -> NTuple{2, Int64}
 
Return the mesh sizes `(nΩ, nν)` of the K3 vertex. The same fermionic mesh is used for
both incoming and outgoing fermionic legs ν and ν'.
"""
function numK3(
    γ :: AbstractReducibleVertex
    ) :: NTuple{2, Int64}

    return N(meshes(γ.K3, Val(1))), N(meshes(γ.K3, Val(2)))
end

# --------------------------------------------------------------------------- #
# Setter methods
# --------------------------------------------------------------------------- #

"""
    set!(γ1::AbstractReducibleVertex, γ2::AbstractReducibleVertex) -> Nothing
 
Copy all asymptotic classes (K1, K2, K3) from `γ2` into `γ1` in-place.
Meshes must be compatible.
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

"""
    set!(γ1::AbstractReducibleVertex{Q}, val::Number) -> Nothing
 
Set all data in K1, K2, and K3 to the constant value `val` (cast to type `Q`).
Typically used to zero-initialize a channel: `set!(γ, 0)`.
"""
function MatsubaraFunctions.set!(
    γ1  :: AbstractReducibleVertex{Q},
    val :: Number,
    )   :: Nothing where {Q}

    set!(γ1.K1, Q(val))
    set!(γ1.K2, Q(val))
    set!(γ1.K3, Q(val))

    return nothing
end

# --------------------------------------------------------------------------- #
# Comparison
# --------------------------------------------------------------------------- #

"""
    ==(γ1::AbstractReducibleVertex, γ2::AbstractReducibleVertex) -> Bool
 
Element-wise equality check across all three asymptotic classes K1, K2, K3.
"""
function Base.:(==)(
    γ1 :: AbstractReducibleVertex,
    γ2 :: AbstractReducibleVertex
    )  :: Bool
    return (γ1.K1 == γ2.K1) && (γ1.K2 == γ2.K2) && (γ1.K3 == γ2.K3)
end

# --------------------------------------------------------------------------- #
# Addition
# --------------------------------------------------------------------------- #

"""
    add!(γ1::AbstractReducibleVertex, γ2::AbstractReducibleVertex) -> Nothing
 
In-place addition `γ1 += γ2`, applied to each asymptotic class.
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
    mult_add!(γ1::AbstractReducibleVertex, γ2::AbstractReducibleVertex, val::Number) -> Nothing
 
In-place fused multiply-add `γ1 += val * γ2`, applied to each asymptotic class.
Useful in iterative solvers and self-consistency loops (e.g., mixing updates).
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

# --------------------------------------------------------------------------- #
# Length of channel
# --------------------------------------------------------------------------- #

"""
    length(γ::AbstractReducibleVertex) -> Int64
 
Return the total number of scalar data elements stored across all three asymptotic classes.
This equals `|K1| + |K2| + |K3|` where each term is the product of the respective mesh sizes.
E.g. `|K2| = mK2Ω * mK2ν`
"""
function Base.length(
    γ :: AbstractReducibleVertex
    ) :: Int64

    return length(γ.K1.data) + length(γ.K2.data) + length(γ.K3.data)
end

# --------------------------------------------------------------------------- #
# Maximum absolute value
# --------------------------------------------------------------------------- #

"""
    absmax(γ::AbstractReducibleVertex) -> Float64
 
Return the maximum absolute value across all elements of K1, K2, and K3.
Used as a convergence diagnostic in self-consistency iterations.
"""
function MatsubaraFunctions.absmax(
    γ :: AbstractReducibleVertex
    ) :: Float64

    return max(absmax(γ.K1), absmax(γ.K2), absmax(γ.K3))
end

# --------------------------------------------------------------------------- #
# Flatten into vector / Unflatten from vector
# --------------------------------------------------------------------------- #

"""
    flatten!(γ::AbstractReducibleVertex, x::AbstractVector) -> Nothing
 
Write all vertex data (K1, then K2, then K3 in order) into the preallocated flat vector `x` in-place.
The layout is: `x[1:lenK1] = K1`, `x[lenK1+1:lenK1+lenK2] = K2`, etc.
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
    flatten(γ::AbstractReducibleVertex{Q}) -> Vector{Q}
 
Allocate and return a flat vector containing all vertex data (K1, K2, K3 in order).
See also `flatten!` for the in-place variant.
"""
function MatsubaraFunctions.flatten(
    γ :: AbstractReducibleVertex{Q}
    ) :: Vector{Q} where {Q}

    x = Array{Q}(undef, length(γ))
    flatten!(γ, x)

    return x
end

"""
    unflatten!(γ::AbstractReducibleVertex, x::AbstractVector) -> Nothing
 
Read vertex data back from the flat vector `x` (in the same layout as `flatten!`) and
write it into K1, K2, K3 in-place. Inverse of `flatten!`.
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

# --------------------------------------------------------------------------- #
# Copy
# --------------------------------------------------------------------------- #

"""
    copy(γ::Channel{Q}) -> Channel{Q}
 
Return a copy of the channel, allocating new arrays for K1, K2, and K3.
"""
function Base.:copy(
    γ :: Channel{Q}
    ) :: Channel{Q} where {Q}
    return Channel(copy(γ.K1), copy(γ.K2), copy(γ.K3))
end

# --------------------------------------------------------------------------- #
# Evaluators 
# --------------------------------------------------------------------------- #
#
# The call operator evaluates the full reducible vertex
#
#   Φ_r(Ω, ν, ν') = K1(Ω) + K2(Ω, ν) + K2'(Ω, ν') + K3(Ω, ν, ν')
#
# using the asymptotic decomposition. The individual contributions can be toggled via
# keyword arguments `K1`, `K2`, `K3` (all default to `true`).
#
# Note that K2'(Ω, ν') is stored in the same array as K2 and is evaluated as `K2[Ω, νp]`.
# Hence a single call may contribute both K2(Ω,ν) and K2'(Ω,ν') from the same field.
#
# All evaluations return zero for frequencies outside the stored mesh range, implementing
# the implicit truncation of the asymptotic expansion at the grid boundary.
#
# Four overloads handle combinations of finite and infinite (ν → ∞) fermionic frequencies,
# since InfiniteMatsubaraFrequency propagates analytically through the K-class hierarchy.

"""
    (γ::Channel{Q})(Ω, ν, νp; K1=true, K2=true, K3=true) -> Q
 
Evaluate Φ_r(Ω, ν, ν') for finite Matsubara frequencies Ω (bosonic), ν, ν' (fermionic).
 
The result accumulates contributions from the active asymptotic classes:
- K1(Ω)               if `K1=true` and Ω is within the K1 bosonic mesh.
- K2(Ω,ν) + K2'(Ω,ν') if `K2=true` and the respective frequencies are in-bounds.
- K3(Ω,ν,ν')          if `K3=true` and both fermionic frequencies are in-bounds.
 
Out-of-bounds frequencies return zero, consistent with the high-frequency decay of Φ_r.
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

        # check if Ω is in-bounds for K1. Nested because numK1 >= numK2
        if is_inbounds(Ω, meshes(γ.K2, Val(1)))

            # check if ν, ν' are in-bounds for K2
            ν1_inbounds = is_inbounds(ν, meshes(γ.K2, Val(2)))
            ν2_inbounds = is_inbounds(νp, meshes(γ.K2, Val(2)))

            if ν1_inbounds && ν2_inbounds
                # Both fermionic arguments in range: add K2(Ω,ν) + K2'(Ω,ν') + K3(Ω,ν,ν') if K2=true and K3=true respectively
                K2 && (val += γ.K2[Ω, ν] + γ.K2[Ω, νp])
                K3 && (val += γ.K3(Ω, ν, νp)) # bounds check for Ω, ν, νp handled in MeshFunction()

            elseif ν1_inbounds
                # Only ν in range: add K2(Ω, ν) if K2=true; K3 requires both legs to be in range
                K2 && (val += γ.K2[Ω, ν])

            elseif ν2_inbounds
                # Only ν' in range: add K2'(Ω, ν') if K2=true
                K2 && (val += γ.K2[Ω, νp])
            end
        end
    end

    return val
end

"""
    (γ::Channel{Q})(Ω, ν::InfiniteMatsubaraFrequency, νp; K1, K2, K3) -> Q
 
Evaluate Φ_r(Ω, ∞, ν') — asymptotic limit as the incoming fermionic frequency ν → ∞.
 
In this limit, K2(Ω, ν) and K3(Ω, ν, ν') vanish, leaving only K1(Ω) and K2'(Ω, ν').
This overload is used when computing asymptotic corrections in BSE or SDE kernels.
"""
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

"""
    (γ::Channel{Q})(Ω, ν, νp::InfiniteMatsubaraFrequency; K1, K2, K3) -> Q
 
Evaluate Φ_r(Ω, ν, ∞) — asymptotic limit as the outgoing fermionic frequency ν' → ∞.
 
In this limit, K2'(Ω, ν') and K3(Ω, ν, ν') vanish, leaving only K1(Ω) and K2(Ω, ν).
Symmetric counterpart to the ν → ∞ overload above.
"""
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

 
"""
    (γ::Channel{Q})(Ω, ν::InfiniteMatsubaraFrequency, νp::InfiniteMatsubaraFrequency; K1, K2, K3) -> Q
 
Evaluate Φ_r(Ω, ∞, ∞) — double asymptotic limit ν, ν' → ∞.
 
In this limit, all K2 and K3 contributions vanish, leaving only K1(Ω). This is the
leading high-frequency behavior of the reducible vertex.
"""
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


# --------------------------------------------------------------------------- #
# Reducer
# --------------------------------------------------------------------------- #

"""
    reduce!(γ::Channel) -> Nothing
 
Subtract the lower-order asymptotic contributions from the higher-order classes in-place,
converting from "cumulative" to "pure" (reduced) asymptotic classes:
 
  K2_reduced(Ω, ν)      = K2(Ω, ν) − K1(Ω)
  K3_reduced(Ω, ν, ν')  = K3(Ω, ν, ν') − K1(Ω) − K2_reduced(Ω, ν) − K2'_reduced(Ω, ν')
 
After this operation, each class stores only the increment not captured by the lower
classes.
"""
function reduce!(
    γ :: Channel
    ) :: Nothing

    # Subtract K1(Ω) from K2(Ω, ν) for all ν on the K2 fermionic mesh
    for iΩ in eachindex(meshes(γ.K2, Val(1)))
        Ω = value(meshes(γ.K2, Val(1))[iΩ])
        K1val = γ.K1[Ω]

        for iν in eachindex(meshes(γ.K2, Val(2)))
            ν = value(meshes(γ.K2, Val(2))[iν])
            γ.K2[Ω, ν] -= K1val
        end
    end

    # Subtract K1(Ω) + K2_reduced(Ω, ν) + K2'_reduced(Ω, ν') from K3(Ω, ν, ν') for all ν, ν' on the K3 mesh
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

# --------------------------------------------------------------------------- #
# HDF5 I/O
# --------------------------------------------------------------------------- #

"""
    save!(file::HDF5.File, label::String, γ::AbstractReducibleVertex) -> Nothing
 
Save the three asymptotic classes K1, K2, K3 of a reducible vertex to an HDF5 file
under the group `label/K1`, `label/K2`, and `label/K3`, respectively.
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
    load_channel(::Type{T}, file::HDF5.File, label::String) -> T
 
Load a reducible vertex of concrete type `T <: AbstractReducibleVertex` from an HDF5 file.
The K1, K2, K3 `MeshFunction`s are read from `label/K1`, `label/K2`, `label/K3`
and passed to the `T(K1, K2, K3)` constructor.
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
