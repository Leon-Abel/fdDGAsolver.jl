# =============================================================================
# The full two-particle vertex in the parquet decomposition (local version).
#
# A Vertex is a vertex F0, which is not split further, plus the reducible
# vertices of the three channels (Lihm 2025, Eq. (1)):
#
#   F = F0 + Φ_p + Φ_t + Φ_a
#
# F0 is what the channels are added on top of:
#   - In the parquet approximation (PA) it is the bare vertex U (a RefVertex),
#     which stands in for the fully irreducible vertex Λ.
#   - In finite-difference (fd) runs it is the reference vertex f. Then
#     F = f + Φ̃_p + Φ̃_t + Φ̃_a, and the channels hold the differences
#     Φ̃_r = Φ_r(target) - Φ_r(reference) (Lihm 2025, text below Eq. (5)).
#     f is itself a vertex: for DMFT input, a RefVertex core plus the DMFT
#     channels (K1 and K2).
# Because of this nesting, several functions (e.g. the SDE) walk down
# F → F.F0 → F.F0.F0 → ….
#
# Each Φ_r is a Channel (channel.jl) in its own channel's frequency
# parametrization. Evaluating F in channel r converts the arguments of the
# other two channels (convention.jl, Eq. (S6)). Only the parallel spin
# component F^= is stored; the crossed one F^× follows from crossing symmetry
# (Eq. (S4a)), and the density combination 2F^= + F^× from both (types.jl).
#
# Only the channels are unknowns of the solver: set!, add!, flatten, length,
# … act on Φ_p, Φ_t, Φ_a and leave F0 untouched.
#
# Paper ↔ code: Φ_r ↔ the fields `γp`, `γt`, `γa`; reference f ↔ `F0`;
# ω ↔ Ω; ν′ ↔ νp. The paper's F₀ (the bare vertex U) is bare_vertex(F), not
# the field F0.
# =============================================================================


# -----------------------------------------------------------------------------
# Abstract type
# -----------------------------------------------------------------------------

"""
    AbstractVertex{Q}

Supertype of all full vertices with number type `Q` (e.g. `ComplexF64`): the
local [`Vertex`](@ref), the lattice `NL_Vertex`, `NL2_Vertex`, `NL3_Vertex`
and the MBE vertices. All have the fields `F0`, `γp`, `γt`, `γa`.
"""
abstract type AbstractVertex{Q}; end

# number type of the stored data
Base.eltype(::Type{<: AbstractVertex{Q}}) where {Q} = Q

# Which channel type a vertex type uses (Channel for Vertex). load_vertex needs
# it to read the channels back. New vertex types must define it, otherwise this
# fallback throws an error.
channel_type(::Type{<: AbstractVertex}) = error("Not implemented")

# -----------------------------------------------------------------------------
# Vertex struct
# -----------------------------------------------------------------------------

"""
    Vertex{Q, VT} <: AbstractVertex{Q}

Full two-particle vertex without momentum dependence,
`F = F0 + Φ_p + Φ_t + Φ_a` (Lihm 2025, Eq. (1)).

- `F0` (type `VT`): the part the channels are added to. A `RefVertex` (the bare
  U in the PA, or U plus a core) or another vertex, e.g. the reference f in fd
  runs. Then the channels hold `Φ̃_r`.
- `γp`, `γt`, `γa`: the reducible vertices `Φ_p`, `Φ_t`, `Φ_a` as [`Channel`](@ref)s.

Constructors:
- `Vertex(F0, γp, γt, γa)`: from existing channels.
- `Vertex(F0, T, numK1, numK2, numK3)`: three zero channels with the given box
  sizes; the number type is taken from `F0`.
"""
struct Vertex{Q, VT} <: AbstractVertex{Q}
    F0 :: VT
    γp :: Channel{Q}
    γt :: Channel{Q}
    γa :: Channel{Q}

    # Constructor from existing channels.
    function Vertex(
        F0 :: VT,
        γp :: Channel{Q},
        γt :: Channel{Q},
        γa :: Channel{Q},
        )  :: Vertex{Q} where {Q, VT}

        return new{Q, VT}(F0, γp, γt, γa)
    end

    # Constructor from temperature and box sizes; all channels start at zero.
    function Vertex(
        F0    :: VT,
        T     :: Float64,
        numK1 :: Int64,
        numK2 :: NTuple{2, Int64},
        numK3 :: NTuple{2, Int64},
        ) where {VT}

        # ensure correct type is passed down to channels
        Q = eltype(F0)

        # Build one zero channel and copy it for the three channels.
        γ = Channel(T, numK1, numK2, numK3, Q)
        return new{Q, VT}(F0, γ, copy(γ), copy(γ)) :: Vertex{Q}
    end
end

# A Vertex stores its channels as Channel objects (used by load_vertex).
channel_type(::Type{Vertex}) = Channel

# number type of the stored data
Base.eltype(::Type{<: Vertex{Q}}) where {Q} = Q

# Printing: type, U, temperature, F0 and the box sizes.
function Base.show(io::IO, Γ::AbstractVertex{Q}) where {Q}
    print(io, "$(nameof(typeof(Γ))){$Q}, U = $(bare_vertex(Γ)), T = $(temperature(Γ))\n")
    print(io, "F0 : $(Γ.F0)\n")
    print(io, "K1 : $(numK1(Γ))\n")
    print(io, "K2 : $(numK2(Γ))\n")
    print(io, "K3 : $(numK3(Γ))")
end

# -----------------------------------------------------------------------------
# Getters
# -----------------------------------------------------------------------------
# All three channels use the same meshes, so it is enough to ask one of them.

"""
    temperature(F::AbstractVertex) -> Float64

Temperature T = 1/β of the Matsubara meshes.
"""
function MatsubaraFunctions.temperature(
    F :: AbstractVertex
    ) :: Float64

    return MatsubaraFunctions.temperature(F.γp)
end

"""
    numK1(F::AbstractVertex) -> Int64

Number of positive bosonic frequencies in the K1 box.
"""
function numK1(
    F :: AbstractVertex
    ) :: Int64

    return numK1(F.γp)
end

"""
    numK2(F::AbstractVertex) -> (nΩ, nν)

Number of positive bosonic and fermionic frequencies in the K2 box.
"""
function numK2(
    F :: AbstractVertex
    ) :: NTuple{2, Int64}

    return numK2(F.γp)
end

"""
    numK3(F::AbstractVertex) -> (nΩ, nν)

Number of positive bosonic and fermionic frequencies in the K3 box. ν and ν′
share the same fermionic mesh.
"""
function numK3(
    F :: AbstractVertex
    ) :: NTuple{2, Int64}

    return numK3(F.γp)
end

# -----------------------------------------------------------------------------
# Setters
# -----------------------------------------------------------------------------

"""
    set!(F1::AbstractVertex, F2::AbstractVertex) -> nothing
    set!(F::AbstractVertex, val::Number) -> nothing

Overwrite the three channels `Φ_p`, `Φ_t`, `Φ_a` of `F1` with those of `F2`, or set
all their entries to `val` (e.g. `set!(F, 0)`). `F0` is not changed.
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

# set! with a number, see the docstring above
function MatsubaraFunctions.set!(
    F :: AbstractVertex,
    val :: Number,
    ) :: Nothing

    set!(F.γp, val)
    set!(F.γt, val)
    set!(F.γa, val)

    return nothing
end

# -----------------------------------------------------------------------------
# Arithmetic
# -----------------------------------------------------------------------------

"""
    add!(F1::AbstractVertex, F2::AbstractVertex) -> nothing

In place `F1 += F2` for the three channels. `F0` is not changed.
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
    mult_add!(F1::AbstractVertex, F2::AbstractVertex, val) -> nothing

In place `F1 += val * F2` for the three channels. `F0` is not changed.
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

# -----------------------------------------------------------------------------
# Maximum absolute value
# -----------------------------------------------------------------------------

"""
    absmax(F::AbstractVertex) -> Float64

Largest absolute value in the three channels. Used to monitor convergence.
"""
function MatsubaraFunctions.absmax(
    F :: AbstractVertex
    ) :: Float64

    return max(absmax(F.γp), absmax(F.γt), absmax(F.γa))
end

# -----------------------------------------------------------------------------
# Comparison
# -----------------------------------------------------------------------------

"""
    ==(F1::AbstractVertex, F2::AbstractVertex) -> Bool

True if `F0` and all three channels are equal.
"""
function Base.:(==)(
    F1 :: AbstractVertex{Q},
    F2 :: AbstractVertex{Q},
    )  :: Bool where {Q}
    return (F1.F0 == F2.F0) && (F1.γa == F2.γa) && (F1.γp == F2.γp) && (F1.γt == F2.γt)
end

# -----------------------------------------------------------------------------
# Size and conversion to a flat vector
# -----------------------------------------------------------------------------
# The solvers treat the three channels as one long vector of unknowns; F0 is
# fixed and not part of it.

"""
    length(F::AbstractVertex) -> Int

Number of stored values in the three channels, i.e. the number of unknowns
(F0 not included).
"""
function Base.length(
    F :: AbstractVertex,
    ) :: Int
    return length(F.γp) + length(F.γt) + length(F.γa)
end

"""
    flatten!(F::AbstractVertex, x) -> nothing

Write the three channels into the vector `x`, in the order `Φ_p`, `Φ_t`, `Φ_a`.
`x` must have length `length(F)`.
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
    flatten(F::AbstractVertex) -> Vector

The three channels in one new vector, in the order `Φ_p`, `Φ_t`, `Φ_a`.
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
    unflatten!(F::AbstractVertex, x) -> nothing

Read the three channels back from the vector `x`, written by `flatten!`.
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

# -----------------------------------------------------------------------------
# Copy
# -----------------------------------------------------------------------------

"""
    copy(F::Vertex) -> Vertex

Independent copy of the vertex, including a copy of `F0`. The copy therefore
no longer shares its reference vertex with the original (in a solver,
`S.F.F0 === S.F0`).
"""
function Base.:copy(
    F :: Vertex{Q}
    ) :: Vertex{Q} where {Q}

    return Vertex(copy(F.F0), copy(F.γp), copy(F.γt), copy(F.γa))
end

# -----------------------------------------------------------------------------
# Evaluation
# -----------------------------------------------------------------------------
# A vertex is called like a function. The value is the sum of F0 and the three
# channels, each evaluated at the arguments converted to its own
# parametrization (Lihm 2025, Eq. (S6)). The keywords `F0`, `γp`, `γt`, `γa`
# switch single parts off, e.g. to get the irreducible vertex of a channel.

"""
    (F::Vertex)(Ω, ν, νp, ::Type{Ch}, ::Type{Sp}; F0 = true, γp = true, γt = true, γa = true) -> value

Value of the full vertex at frequencies (Ω, ν, νp) given in channel `Ch`'s
parametrization, for spin component `Sp`:

    F(Ω, ν, ν′) = F0 + Φ_p + Φ_t + Φ_a    (each Φ_r at converted arguments)

The keywords switch the parts on or off. For example, `F0 = false` together with
`γa = false` gives the sum of the p and t channels. In fd runs that is
`Ĩ_a` = `Φ̃_p` + `Φ̃_t`, as used in the BSE kernels; in the PA it is `I_a` - U.

- `pSp`: the sum above.
- `xSp`: from crossing symmetry (Lihm 2025, Eq. (S4a)), see the methods below.
- `dSp`: `2·pSp + xSp`; in the t channel this is the density vertex `F_D`.

If `ν` or `νp` is [`νInf`](@ref), the other two channels vanish (their bosonic
argument becomes infinite), so only F0 and channel `Ch` itself contribute:
this gives the limits `F_012` and `F_01̄2` of Eq. (S22).
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

    # reference / bare part
    if F0
        val += F.F0(Ω, ν, νp, Ch, pSp)
    end

    # each channel at the arguments converted to its own parametrization
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

# Method for pSp with ν or ν′ = νInf: only F0 and channel Ch itself contribute.
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


# Method for xSp: crossing symmetry (Lihm 2025, Eq. (S4a)). Exchanging the two
# outgoing legs maps the crossed component to minus the parallel one and swaps
# the roles of the a and t channels (p stays p), hence γt = γa, γa = γt below.
#   p channel:  F^×_p(Ω, ν, ν′) = -F^=_p(Ω, ν, Ω - ν′)
#   t channel:  F^×_t(Ω, ν, ν′) = -F^=_a(Ω, ν′, ν)
#   a channel:  F^×_a(Ω, ν, ν′) = -F^=_t(Ω, ν′, ν)
# In the t and a lines the code leaves out the swap ν ↔ ν′, which is allowed
# by the ν ↔ ν′ symmetry of the vertex (docs, Sec. 1.5).
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

# Method for dSp = 2·pSp + xSp (types.jl); the density vertex in the t channel.
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

# -----------------------------------------------------------------------------
# Bare vertex
# -----------------------------------------------------------------------------

"""
    bare_vertex(F::AbstractVertex) -> U
    bare_vertex(F::AbstractVertex, ::Type{Sp}) -> value
    bare_vertex(F::AbstractVertex, ::Type{Ch}, ::Type{Sp}) -> value

The bare vertex F₀ (Lihm 2025, Eq. (S3)), passed down through `F0` to the
RefVertex at the bottom of the vertex: `U`, or its value in spin component `Sp`
(+U for pSp and dSp, -U for xSp). The three-argument form ignores the channel.
"""
@inline bare_vertex(F :: AbstractVertex) =  bare_vertex(F.F0)

# bare_vertex with a spin component, see the docstring above
@inline bare_vertex(F :: AbstractVertex, :: Type{Sp}) where {Sp <: SpinTag} = bare_vertex(F.F0, Sp)

@inline bare_vertex(F :: AbstractVertex, :: Type{Ch}, :: Type{Sp}) where {Ch <: ChannelTag, Sp <: SpinTag} = bare_vertex(F, Sp)  # TODO: remove

# Old helper (unused): tabulate F on a frequency grid, written for an older
# MatsubaraFunctions interface.
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

# -----------------------------------------------------------------------------
# From cumulative to separate classes
# -----------------------------------------------------------------------------

"""
    reduce!(F::AbstractVertex) -> nothing

Apply [`reduce!`](@ref) to each of the three channels: turn cumulative classes
(K2 holding K1 + K2, K3 holding the whole `Φ_r`) into the separate classes of
Lihm 2025, Eq. (S7). Not used by the solvers; the tests use it.
"""
function reduce!(
    F :: AbstractVertex
    ) :: Nothing

    reduce!(F.γp)
    reduce!(F.γt)
    reduce!(F.γa)

    return nothing
end

# -----------------------------------------------------------------------------
# HDF5 input and output
# -----------------------------------------------------------------------------

"""
    save!(file, label, F::AbstractVertex) -> nothing

Write a vertex to the HDF5 group `label`: `F0` (recursively, if it is a
vertex itself) and the channels as subgroups `F0`, `γp`, `γt`, `γa`.
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
    load_vertex(T, file, label) -> T

Read a vertex of type `T` (e.g. `Vertex`, `NL2_Vertex`) written by `save!` from
the HDF5 group `label`.

`F0` can be a RefVertex or a nested vertex. If the group `F0` carries the
attribute `U` it is a RefVertex. Otherwise the vertex types `Vertex`,
`NL_Vertex`, `NL2_Vertex`, `NL3_Vertex` and `NL2_MBEVertex` are tried in that
order. The channels are read with `load_channel` and the channel type of `T`.
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
