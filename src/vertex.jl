# vertex.jl
#
# The full two-particle vertex F is decomposed via the parquet decomposition:
#
#   F = Λ + γ_p + γ_t + γ_a
#
# where Λ is the fully two-particle irreducible (2PI) vertex, and γ_r (r = p, t, a) are
# the channel-reducible vertices in the particle-particle (p), transverse antiparallel (t),
# and antiparallel (a) channels, respectively. Each γ_r is further decomposed in the
# asymptotic (high-frequency) expansion :
#
#   γ_r(Ω, ν, ν') = K1_r(Ω) + K2_r(Ω, ν) + K2'_r(Ω, ν') + K3_r(Ω, ν, ν')
#
# The vertex decomposes into parallel-spin (pSp, F^=) and
# crossed-spin (xSp, F^x) component. The density (dSp) component is:
#   F^D = 2*F^= + F^x
# Crossing symmetry relates xSp evaluations to pSp via channel permutations.


# ---------------------------------------------------------------------------------------------------
# Abstract base type
# ---------------------------------------------------------------------------------------------------

"""
    abstract type AbstractVertex{Q}
 
Abstract base type for all two-particle vertex representations, parameterized by the
scalar field type Q (e.g. ComplexF64).
"""
abstract type AbstractVertex{Q}; end

# Expose the scalar type of a AbstractVertex.
Base.eltype(::Type{<: AbstractVertex{Q}}) where {Q} = Q

# Requires all AbstractVertex subtypes to implement channel_type; throws at runtime
# if a new subtype is added without a corresponding channel_type definition.
channel_type(::Type{<: AbstractVertex}) = error("Not implemented")

# ---------------------------------------------------------------------------------------------------
# Concrete vertex type
# ---------------------------------------------------------------------------------------------------

"""
    struct Vertex{Q, VT} <: AbstractVertex{Q}
 
Concrete representation of the full two-particle vertex F in the parquet decomposition:
 
    F(Ω, ν, ν') = F0(Ω, ν, ν') + γp(Ω_p, ν_p, ν'_p) + γt(Ω_t, ν_t, ν'_t) + γa(Ω_a, ν_a, ν'_a)
 

# Constructors
- `Vertex(F0, γp, γt, γa)`: Construct directly from pre-allocated `Channel` objects and 2PI vertex.
- `Vertex(F0, T, numK1, numK2, numK3)`: Zero-initialize from mesh size parameters and 2PI vertex.
 
Note: all three channels share the same Matsubara mesh sizes (numK1, numK2, numK3), so
their `Channel` objects are constructed from a single template and then copied.
"""
struct Vertex{Q, VT} <: AbstractVertex{Q}
    F0 :: VT
    γp :: Channel{Q}
    γt :: Channel{Q}
    γa :: Channel{Q}

    # Constructor from pre-built Channel objects and 2PI vertex.
    function Vertex(
        F0 :: VT,
        γp :: Channel{Q},
        γt :: Channel{Q},
        γa :: Channel{Q},
        )  :: Vertex{Q} where {Q, VT}

        return new{Q, VT}(F0, γp, γt, γa)
    end

    # Constructor from temperature, mesh sizes and 2PI vertex.
    function Vertex(
        F0    :: VT,
        T     :: Float64,
        numK1 :: Int64,
        numK2 :: NTuple{2, Int64},
        numK3 :: NTuple{2, Int64},
        ) where {VT}

        # ensure correct type is passed down to channels
        Q = eltype(F0)

        # Build a single zero-initialized Channel and replicate it for the three channels.
        γ = Channel(T, numK1, numK2, numK3, Q)
        return new{Q, VT}(F0, γ, copy(γ), copy(γ)) :: Vertex{Q}
    end
end

# Declares that Vertex stores its reducible-vertex channels as Channel{Q} objects;
# used by load_vertex to dispatch the correct load_channel call during HDF5 deserialization.
channel_type(::Type{Vertex}) = Channel

# Expose the scalar type of a Vertex.
Base.eltype(::Type{<: Vertex{Q}}) where {Q} = Q

"""
    Base.show(io::IO, Γ::AbstractVertex) where {Q}
 
Pretty-print the vertex type, bare coupling U, temperature T, and grid sizes.
"""
function Base.show(io::IO, Γ::AbstractVertex{Q}) where {Q}
    print(io, "$(nameof(typeof(Γ))){$Q}, U = $(bare_vertex(Γ)), T = $(temperature(Γ))\n")
    print(io, "F0 : $(Γ.F0)\n")
    print(io, "K1 : $(numK1(Γ))\n")
    print(io, "K2 : $(numK2(Γ))\n")
    print(io, "K3 : $(numK3(Γ))")
end

# ---------------------------------------------------------------------------------------------------
# Getter methods
# ---------------------------------------------------------------------------------------------------

"""
    temperature(F::AbstractVertex) ->  Float64

Return the temperature associated with the vertex's Matsubara frequency grids.
All channels share the same temperature so the choice of channel is arbitrary.
"""
function MatsubaraFunctions.temperature(
    F :: AbstractVertex
    ) :: Float64

    return MatsubaraFunctions.temperature(F.γp)
end

"""
    numK1(F::AbstractVertex) ->  Int64

Return the number of positive bosonic Matsubara frequencies in the K1 mesh.
"""
function numK1(
    F :: AbstractVertex
    ) :: Int64

    return numK1(F.γp)
end

"""
    numK2(F::AbstractVertex) ->  NTuple{2, Int64}

Return the number of positive (bosonic, fermionic) Matsubara frequencies in the K2 mesh.
"""
function numK2(
    F :: AbstractVertex
    ) :: NTuple{2, Int64}

    return numK2(F.γp)
end

"""
    numK3(F::AbstractVertex) ->  NTuple{2, Int64}

Return the number of positive (bosonic, fermionic) Matsubara frequencies used in the K3 asymptotic class.
The same fermionic mesh is used for both incoming and outgoing fermionic legs ν and ν'.
"""
function numK3(
    F :: AbstractVertex
    ) :: NTuple{2, Int64}

    return numK3(F.γp)
end

# ---------------------------------------------------------------------------------------------------
# Setter methods
# ---------------------------------------------------------------------------------------------------

"""
    set!(F1::AbstractVertex, F2::AbstractVertex) ->  Nothing
 
Copy the channel data from `F2` into `F1` (in-place assignment), channel by channel.
Does not copy `F0`; only the reducible parts γ_p, γ_t, γ_a are overwritten.
"""
function MatsubaraFunctions.set!(
    F1 :: AbstractVertex,
    F2 :: AbstractVertex
    )  :: Nothing

    set!(F1.γp, F2.γp)
    set!(F1.γt, F2.γt)
    set!(F1.γa, F2.γa)

    return nothing
end

"""
    set!(F::AbstractVertex, val::Number) ->  Nothing
 
Set all reducible channel data (γ_p, γ_t, γ_a) to the constant value `val` (in-place).
Typically used to zero-initialize a vertex: `set!(F, 0)`.
"""
function MatsubaraFunctions.set!(
    F :: AbstractVertex,
    val :: Number,
    ) :: Nothing

    set!(F.γp, val)
    set!(F.γt, val)
    set!(F.γa, val)

    return nothing
end

# ---------------------------------------------------------------------------------------------------
# Arithmetic
# --------------------------------------------------------------------------------------------------- 

"""
    add!(F1::AbstractVertex, F2::AbstractVertex) -> Nothing
 
In-place addition: F1 += F2, applied channel-wise to γ_p, γ_t, γ_a.
`F0` is not modified.
"""
function MatsubaraFunctions.add!(
    F1 :: AbstractVertex,
    F2 :: AbstractVertex
    )  :: Nothing

    add!(F1.γp, F2.γp)
    add!(F1.γt, F2.γt)
    add!(F1.γa, F2.γa)

    return nothing
end

"""
    mult_add!(F1::AbstractVertex, F2::AbstractVertex, val::Number) -> Nothing
 
In-place multiply-add: F1 += val * F2, applied channel-wise.
`F0` is not modified.
"""
function MatsubaraFunctions.mult_add!(
    F1 :: AbstractVertex,
    F2 :: AbstractVertex,
    val :: Number
    )  :: Nothing

    mult_add!(F1.γp, F2.γp, val)
    mult_add!(F1.γt, F2.γt, val)
    mult_add!(F1.γa, F2.γa, val)

    return nothing
end

# ---------------------------------------------------------------------------------------------------
# Maximum absolute value
# --------------------------------------------------------------------------------------------------- 

"""
    absmax(F::AbstractVertex) -> Float64
 
Return the maximum absolute value across all three reducible channels γ_p, γ_t, γ_a.
Used as a convergence diagnostic in self-consistency iterations.
"""
function MatsubaraFunctions.absmax(
    F :: AbstractVertex
    ) :: Float64

    return max(absmax(F.γp), absmax(F.γt), absmax(F.γa))
end

# ---------------------------------------------------------------------------------------------------
# Comparison
# --------------------------------------------------------------------------------------------------- 

"""
    ==(F1::AbstractVertex{Q}, F2::AbstractVertex{Q}) -> Bool
 
Equality check: returns `true` if `F0`, γ_p, γ_t, and γ_a all compare equal.
"""
function Base.:(==)(
    F1 :: AbstractVertex{Q},
    F2 :: AbstractVertex{Q},
    )  :: Bool where {Q}
    return (F1.F0 == F2.F0) && (F1.γa == F2.γa) && (F1.γp == F2.γp) && (F1.γt == F2.γt)
end

# ---------------------------------------------------------------------------------------------------
# Length of Vertex
# ---------------------------------------------------------------------------------------------------

"""
    length(F::AbstractVertex) -> Int
 
Return the total number of stored data points in the reducible channels
(γ_p + γ_t + γ_a), i.e. the number of independent degrees of freedom
(excluding F0).
"""
function Base.length(
    F :: AbstractVertex,
    ) :: Int
    return length(F.γp) + length(F.γt) + length(F.γa)
end

# ---------------------------------------------------------------------------------------------------
# Flatten into vector / Unflatten from vector
# ---------------------------------------------------------------------------------------------------

"""
    flatten!(F::AbstractVertex, x::AbstractVector) -> Nothing
 
Write all reducible channel data (γ_p, γ_t, γ_a in that order) into the pre-allocated
vector `x` in-place. The vector must have length `length(F)`.
"""
function MatsubaraFunctions.flatten!(
    F :: AbstractVertex,
    x :: AbstractVector
    ) :: Nothing

    offset = 0
    len_γ  = length(F.γp)

    flatten!(F.γp, @view x[1 + offset : offset + len_γ]); offset += len_γ
    flatten!(F.γt, @view x[1 + offset : offset + len_γ]); offset += len_γ
    flatten!(F.γa, @view x[1 + offset : offset + len_γ]); offset += len_γ

    @assert offset == length(x) "Dimension mismatch between vertex and target vector"
    return nothing
end

"""
    flatten(F::AbstractVertex{Q}) -> Vector{Q}
 
Return a freshly allocated vector containing all reducible channel data
(γ_p, γ_t, γ_a concatenated).
"""
function MatsubaraFunctions.flatten(
    F :: AbstractVertex{Q}
    ) :: Vector{Q} where {Q}

    xp = flatten(F.γp)
    xt = flatten(F.γt)
    xa = flatten(F.γa)

    return vcat(xp, xt, xa)
end

"""
    unflatten!(F::AbstractVertex, x::AbstractVector) -> Nothing
 
Read back reducible channel data from a flat vector `x` into the vertex (inverse of
`flatten!`) in-place. The layout must match: γ_p first, then γ_t, then γ_a, each of length
`length(F.γp)`.
"""
function MatsubaraFunctions.unflatten!(
    F :: AbstractVertex,
    x :: AbstractVector
    ) :: Nothing

    offset = 0
    len_γ  = length(F.γp)

    unflatten!(F.γp, @view x[1 + offset : offset + len_γ]); offset += len_γ
    unflatten!(F.γt, @view x[1 + offset : offset + len_γ]); offset += len_γ
    unflatten!(F.γa, @view x[1 + offset : offset + len_γ]); offset += len_γ

    @assert offset == length(x) "Dimension mismatch between vertex and target vector"
    return nothing
end

# ---------------------------------------------------------------------------------------------------
# Copy
# ---------------------------------------------------------------------------------------------------

"""
    copy(F::Vertex{Q}) -> Vertex{Q}
 
Return a deep copy of the vertex, including independent copies of all three channels and F0.
"""
function Base.:copy(
    F :: Vertex{Q}
    ) :: Vertex{Q} where {Q}

    return Vertex(copy(F.F0), copy(F.γp), copy(F.γt), copy(F.γa))
end

# ---------------------------------------------------------------------------------------------------
# Evaluators
# ---------------------------------------------------------------------------------------------------
# The full vertex is evaluated by summing contributions from F0 and all three
# reducible channels. The channel arguments are optional booleans so that
# partial contributions (e.g. only K1 + K2) can be selected.

"""
    (F::Vertex{Q})(Ω, ν, νp, ::Type{Ch}, ::Type{pSp}; F0, γp, γt, γa) -> Q
 
Evaluate the full vertex in the **parallel spin (pSp)** component, for a given channel
representation `Ch` (one of `pCh`, `tCh`, `aCh`).
 
The frequencies (Ω, ν, νp) are given in the convention of channel `Ch`, and are
internally converted to the native convention of each reducible channel before
evaluation via `convert_frequency`.
 
The decomposition evaluated is:
    F(Ω,ν,ν') = F0(Ω,ν,ν',Ch,pSp) + γ_p(...) + γ_t(...) + γ_a(...)
 
# Arguments
- `Ω`  : bosonic (transfer) Matsubara frequency
- `ν`  : incoming fermionic Matsubara frequency
- `νp` : outgoing fermionic Matsubara frequency
- `Ch` : channel tag (`pCh`, `tCh`, or `aCh`) specifying the frequency convention
- `pSp`: parallel spin component dispatch tag
 
# Keyword arguments (default `true`)
- `F0` : include the bare vertex contribution
- `γp` : include the particle-particle reducible channel
- `γt` : include the transversal particle-hole reducible channel
- `γa` : include the antiparallel particle-hole reducible channel
"""
@inline function (F :: Vertex{Q})(
    Ω  :: MatsubaraFrequency,
    ν  :: MatsubaraFrequency,
    νp :: MatsubaraFrequency,
       :: Type{Ch},
       :: Type{pSp}
    ;
    F0 :: Bool = true,
    γp :: Bool = true,
    γt :: Bool = true,
    γa :: Bool = true
    )  :: Q where {Q, Ch <: ChannelTag}

    val = zero(Q)

    if F0
        val += F.F0(Ω, ν, νp, Ch, pSp)
    end

    if γp
        val += F.γp(convert_frequency(Ω, ν, νp, Ch, pCh)...)
    end

    if γt
        val += F.γt(convert_frequency(Ω, ν, νp, Ch, tCh)...)
    end

    if γa
        val += F.γa(convert_frequency(Ω, ν, νp, Ch, aCh)...)
    end

    return val
end

"""
    (F::Vertex{Q})(Ω, ν, νp, ::Type{Ch}, ::Type{pSp}; F0, γp, γt, γa) -> Q
 
Evaluate the full vertex in the **parallel spin (pSp)** component for the special case
where either `ν` or `νp` is an `InfiniteMatsubaraFrequency` (representing the
high-frequency / asymptotic limit ν → ∞).
 
In this limit, only the reducible vertex in the *same* channel `Ch` as the input
convention is non-zero (all cross-channel contributions vanish by asymptotic power
counting).
"""
@inline function (F :: Vertex{Q})(
    Ω  :: MatsubaraFrequency,
    ν  :: Union{MatsubaraFrequency, InfiniteMatsubaraFrequency},
    νp :: Union{MatsubaraFrequency, InfiniteMatsubaraFrequency},
       :: Type{Ch},
       :: Type{pSp}
    ;
    F0 :: Bool = true,
    γp :: Bool = true,
    γt :: Bool = true,
    γa :: Bool = true
    )  :: Q where {Q, Ch <: ChannelTag}

    val = zero(Q)

    if F0
        val += F.F0(Ω, ν, νp, Ch, pSp)
    end

    if Ch === pCh && γp
        val += F.γp(Ω, ν, νp)
    end

    if Ch === tCh && γt
        val += F.γt(Ω, ν, νp)
    end

    if Ch === aCh && γa
        val += F.γa(Ω, ν, νp)
    end

    return val
end


"""
    (F::Vertex{Q})(Ω, ν, νp, ::Type{Ch}, ::Type{xSp}; F0, γp, γt, γa) -> Q
 
Evaluate the full vertex in the **crossed spin (xSp)** component.
 
The crossed component is not stored independently but is computed on the fly using
crossing symmetry relations (see types.jl). The mapping depends on the channel `Ch`:
 
- `pCh` (particle-particle): Γ^x_p(Ω, ν, ν') = -Γ^p_p(Ω, ν, Ω - ν')
- `tCh` (transversal ph):    Γ^x_t(Ω, ν, ν') = -Γ^p_a(Ω, ν, ν')   [t ↔ a swap]
- `aCh` (antiparallel ph):   Γ^x_a(Ω, ν, ν') = -Γ^p_t(Ω, ν, ν')   [t ↔ a swap]
"""
@inline function (F :: Vertex{Q})(
    Ω  :: MatsubaraFrequency,
    ν  :: Union{MatsubaraFrequency, InfiniteMatsubaraFrequency},
    νp :: Union{MatsubaraFrequency, InfiniteMatsubaraFrequency},
       :: Type{Ch},
       :: Type{xSp}
    ;
    F0 :: Bool = true,
    γp :: Bool = true,
    γt :: Bool = true,
    γa :: Bool = true
    )  :: Q where {Q, Ch <: ChannelTag}

    if Ch === pCh
        # Crossing: exchange ν' → Ω - ν', and swap the t/a channel roles
        return -F(Ω, ν, Ω - νp, pCh, pSp; F0 = F0, γp = γp, γt = γa, γa = γt)

    elseif Ch === tCh
        # Crossing: t-channel crossed component equals negative of a-channel parallel component
        return -F(Ω, ν, νp, aCh, pSp; F0 = F0, γp = γp, γt = γa, γa = γt)

    elseif Ch === aCh
        # Crossing: a-channel crossed component equals negative of t-channel parallel component
        return -F(Ω, ν, νp, tCh, pSp; F0 = F0, γp = γp, γt = γa, γa = γt)

    else
        throw(ArgumentError("Invalid channel $Ch"))
    end
end

"""
    (F::Vertex{Q})(Ω, ν, νp, ::Type{Ch}, ::Type{dSp}; F0, γp, γt, γa) -> Q
 
Evaluate the full vertex in the **density spin (dSp)** component.
 
The density component is a linear combination of the parallel and crossed spin components:
    dSp = 2 * pSp + xSp
(see types.jl).
"""
@inline function (F :: Vertex{Q})(
    Ω  :: MatsubaraFrequency,
    ν  :: Union{MatsubaraFrequency, InfiniteMatsubaraFrequency},
    νp :: Union{MatsubaraFrequency, InfiniteMatsubaraFrequency},
       :: Type{Ch},
       :: Type{dSp}
    ;
    F0 :: Bool = true,
    γp :: Bool = true,
    γt :: Bool = true,
    γa :: Bool = true
    )  :: Q where {Q, Ch <: ChannelTag}

    val = zero(Q)

    # dSp = 2 * pSp + xSp
    val += F(Ω, ν, νp, Ch, pSp; F0, γp, γt, γa) * 2

    val += F(Ω, ν, νp, Ch, xSp; F0, γp, γt, γa)

    return val
end

# ---------------------------------------------------------------------------------------------------
# Bare vertex accessors
# ---------------------------------------------------------------------------------------------------

"""
    bare_vertex(F::AbstractVertex) -> Q
 
Return the bare vertex value (e.g. the Hubbard U), delegating to F.F0.
"""
@inline bare_vertex(F :: AbstractVertex) =  bare_vertex(F.F0)

"""
    bare_vertex(F::AbstractVertex, ::Type{Sp}) -> Q
 
Return the bare vertex in spin component `Sp`, delegating to F.F0.
"""
@inline bare_vertex(F :: AbstractVertex, :: Type{Sp}) where {Sp <: SpinTag} = bare_vertex(F.F0, Sp)

@inline bare_vertex(F :: AbstractVertex, :: Type{Ch}, :: Type{Sp}) where {Ch <: ChannelTag, Sp <: SpinTag} = bare_vertex(F, Sp)  # TODO: remove

# # build full vertex in given frequency convention and spin component
# function mk_vertex(
#     F   :: Vertex,
#     gΩ  :: MatsubaraGrid,
#     gν  :: MatsubaraGrid,
#     gνp :: MatsubaraGrid,
#         :: Type{CT},
#         :: Type{ST}
#     ;
#     F0  :: Bool = true,
#     γp  :: Bool = true,
#     γt  :: Bool = true,
#     γa  :: Bool = true
#     )   :: MatsubaraFunction{3, 1, 4, Float64} where {CT <: ChannelTag, ST <: SpinTag}

#     f = MatsubaraFunction((gΩ, gν, gνp), 1, Float64)

#     Threads.@threads for i in eachindex(f.data)
#         f[i] = F(first(to_Matsubara(f, i))..., CT, ST; F0 = F0, γp = γp, γt = γt, γa = γa)
#     end

#     return f
# end

# ---------------------------------------------------------------------------------------------------
# Reducer 
# ---------------------------------------------------------------------------------------------------

"""
    reduce!(F::AbstractVertex) -> Nothing
 
Apply the asymptotic reduction to all three channels in-place.
 
This subtracts the lower-order (K1, K2) asymptotic contributions from the higher-order
classes so that each class Kn stores only the *remainder* not captured by lower classes.
Specifically, after `reduce!`:
- K2_reduced[Ω, ν] stores K2[Ω, ν] - K1[Ω]
- K3_reduced[Ω, ν, ν'] stores K3[Ω, ν, ν'] - K1[Ω] - K2_reduced[Ω, ν] - K2_reduced[Ω, ν']
"""
function reduce!(
    F :: AbstractVertex
    ) :: Nothing

    reduce!(F.γp)
    reduce!(F.γt)
    reduce!(F.γa)

    return nothing
end

# --------------------------------------------------------------------------- #
# HDF5 I/O
# --------------------------------------------------------------------------- #

"""
    save!(file::HDF5.File, label::String, F::AbstractVertex) -> Nothing
 
Save all components of the vertex to an HDF5 file under the given `label` prefix.
Stores `F0`, `γp`, `γt`, and `γa` in separate subgroups.
"""
function MatsubaraFunctions.save!(
    file  :: HDF5.File,
    label :: String,
    F     :: AbstractVertex
    )     :: Nothing

    MatsubaraFunctions.save!(file, label * "/F0", F.F0)
    MatsubaraFunctions.save!(file, label * "/γp", F.γp)
    MatsubaraFunctions.save!(file, label * "/γt", F.γt)
    MatsubaraFunctions.save!(file, label * "/γa", F.γa)

    return nothing
end

"""
    load_vertex(::Type{T}, file::HDF5.File, label::String) -> T
 
Load a vertex of concrete type `T <: AbstractVertex` from an HDF5 file.
 
The `F0` subgroup is inspected first: if it has a `"U"` attribute it is a bare
`RefVertex`; otherwise the function attempts to load it as a nested `Vertex`,
`NL_Vertex`, `NL2_Vertex`, `NL3_Vertex`, or `NL2_MBEVertex` (tried in sequence).
The three reducible channels are loaded via `load_channel`.
"""
function load_vertex(
          :: Type{T},
    file  :: HDF5.File,
    label :: String
    )     :: T where {T <: AbstractVertex}

    if haskey(attributes(file[label * "/F0"]), "U")
        F0 = load_refvertex(file, label * "/F0")
    else
        try
            F0 = load_vertex(Vertex, file, label * "/F0")
        catch
            try
                F0 = load_vertex(NL_Vertex, file, label * "/F0")
            catch
                try
                    F0 = load_vertex(NL2_Vertex, file, label * "/F0")
                catch
                    try
                        F0 = load_vertex(NL3_Vertex, file, label * "/F0")
                    catch
                        F0 = load_vertex(NL2_MBEVertex, file, label * "/F0")
                    end
                end
            end
        end
    end
    γp = load_channel(channel_type(T), file, label * "/γp")
    γt = load_channel(channel_type(T), file, label * "/γt")
    γa = load_channel(channel_type(T), file, label * "/γa")

    return T(F0, γp, γt, γa)
end
