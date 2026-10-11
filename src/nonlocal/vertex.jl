# =============================================================================
# The lattice vertex of the s-wave solver NL, and the evaluation shared by all
# lattice vertices (NL, NL2, NL3).
#
# As for the local Vertex (vertex.jl; Lihm 2025, Eq. (1)), now with momenta:
#
#   F(Ω, ν, ν′; q, k, k′) = F0 + Φ_p + Φ_t + Φ_a
#
# Evaluating F in channel r converts the frequencies and momenta of the other
# two channels (convention.jl, Eq. (S6)). The crossed spin component comes from
# crossing symmetry, which now also acts on the momenta: in the p channel
# (ν′, k′) → (Ω - ν′, q - k′).
#
# F0 (the reference f in fd runs, e.g. the local DMFT vertex) is always
# evaluated in full. The keyword switches `γp`, `γt`, `γa` act only on this
# vertex's own channels, and `F0` decides whether F0 is included at all.
#
# s-wave shortcut (NL only): if a fermionic momentum is the marker kSW
# (averaged, nonlocal/swave.jl), the channel the vertex is evaluated in keeps
# its bosonic momentum q. The other two channels then have a bosonic momentum
# that contains the averaged one, so they are averaged over their momentum.
# This is how "the r-reducible vertex loses its momentum dependence when
# converted to different channels" (Lihm 2025, text below Eq. (S35)).
#
# Paper ↔ code: q ↔ P; k, k′ ↔ k, kp; Φ_r ↔ the fields `γp`, `γt`, `γa`.
# =============================================================================

"""
    AbstractNonlocalVertex{Q} <: AbstractVertex{Q}

Supertype of the lattice vertices `NL_Vertex`, `NL2_Vertex` and `NL3_Vertex`.
They share the evaluation in this file.
"""
abstract type AbstractNonlocalVertex{Q} <: AbstractVertex{Q} end

"""
    NL_Vertex{Q, VT} <: AbstractNonlocalVertex{Q}

Full lattice vertex of the s-wave solver NL, `F = F0 + Φ_p + Φ_t + Φ_a`, with
the channels stored as [`NL_Channel`](@ref) (bosonic momentum only).

- `F0` (type `VT`): the vertex the channels are added to: a `RefVertex` (bare U)
  in the PA, or the reference f in fd runs (e.g. the local DMFT `Vertex`, which
  ignores momenta).
- `γp`, `γt`, `γa`: the channels `Φ_p`, `Φ_t`, `Φ_a` (`Φ̃_r` in fd runs).

Constructors: `NL_Vertex(F0, γp, γt, γa)`, or
`NL_Vertex(F0, T, numK1, numK2, numK3, meshK)` with three zero channels on the
bosonic momentum mesh `meshK`.
"""
struct NL_Vertex{Q, VT} <: AbstractNonlocalVertex{Q}
    F0 :: VT
    γp :: NL_Channel{Q}
    γt :: NL_Channel{Q}
    γa :: NL_Channel{Q}

    # from existing channels
    function NL_Vertex(
        F0 :: VT,
        γp :: NL_Channel{Q},
        γt :: NL_Channel{Q},
        γa :: NL_Channel{Q},
        )  :: NL_Vertex{Q} where {Q, VT}

        return new{Q, VT}(F0, γp, γt, γa)
    end

    # three zero channels with the given boxes and momentum mesh
    function NL_Vertex(
        F0    :: VT,
        T     :: Float64,
        numK1 :: Int64,
        numK2 :: NTuple{2, Int64},
        numK3 :: NTuple{2, Int64},
        meshK :: KMesh,
        ) where {VT}

        Q = eltype(F0)

        γ = NL_Channel(T, numK1, numK2, numK3, meshK, Q)
        return new{Q, VT}(F0, γ, copy(γ), copy(γ)) :: NL_Vertex{Q}
    end
end

# An NL_Vertex stores its channels as NL_Channel objects (used by load_vertex).
channel_type(::Type{NL_Vertex}) = NL_Channel

# Printing: type, U, temperature, F0 and the box sizes.
function Base.show(io::IO, Γ::NL_Vertex{Q}) where {Q}
    print(io, "$(nameof(typeof(Γ))){$Q}, U = $(bare_vertex(Γ)), T = $(temperature(Γ))\n")
    print(io, "F0 : $(Γ.F0)\n")
    print(io, "K1 : $(numK1(Γ))\n")
    print(io, "K2 : $(numK2(Γ))\n")
    print(io, "K3 : $(numK3(Γ))")
end

# -----------------------------------------------------------------------------
# Getters and copy
# -----------------------------------------------------------------------------

"""
    get_P_mesh(F::AbstractNonlocalVertex) -> KMesh

Mesh of the bosonic momentum q (code: P) of the vertex channels.
"""
function get_P_mesh(
    F :: AbstractNonlocalVertex
    ) :: KMesh

    return get_P_mesh(F.γp)
end

"""
    numP(F::AbstractNonlocalVertex) -> Int64

Number of points `N_q` of the bosonic momentum mesh.
"""
function numP(
    F :: AbstractNonlocalVertex
    ) :: Int64

    return length(get_P_mesh(F))
end

"""
    copy(F::NL_Vertex) -> NL_Vertex

Independent copy, including a copy of `F0`.
"""
function Base.:copy(
    F :: NL_Vertex{Q}
    ) :: NL_Vertex{Q} where {Q}

    return NL_Vertex(copy(F.F0), copy(F.γp), copy(F.γt), copy(F.γa))
end

# -----------------------------------------------------------------------------
# Evaluation
# -----------------------------------------------------------------------------

"""
    (F::AbstractNonlocalVertex)(Ω, ν, νp, P, k, kp, ::Type{Ch}, ::Type{Sp}; F0 = true, γp = true, γt = true, γa = true) -> value

Value of the lattice vertex at frequencies (Ω, ν, νp) and momenta (P, k, kp)
given in channel `Ch`'s parametrization, for spin component `Sp`:

    F = F0 + Φ_p + Φ_t + Φ_a    (each Φ_r at converted frequencies and momenta)

- `pSp`: the sum above. `F0` is evaluated in full; the keywords `γp`, `γt`, `γa`
  switch this vertex's own channels off.
- `xSp`: crossing symmetry, `-F^=` at (Ω, ν, Ω - νp; P, k, P - kp) in the p
  channel, and with the a and t channels exchanged in the a and t channels.
- `dSp`: `2·pSp + xSp`.
- `ν` or `νp` = [`νInf`](@ref): only F0 and channel `Ch` itself contribute.
- For `NL_Vertex`, `k` and/or `kp` may be [`kSW`](@ref): channel `Ch` keeps
  its momentum `P` and the other two channels are averaged over theirs
  (s-wave approximation). `NL2_Vertex` has its own methods for this
  (nonlocal_2/vertex.jl). `NL3_Vertex` has none: with `kSW` and finite
  frequencies it lands in the `νInf` method below, which keeps only `F0` and
  channel `Ch`.
"""
@inline function (F :: AbstractNonlocalVertex{Q})(
    Ω  :: MatsubaraFrequency,
    ν  :: MatsubaraFrequency,
    νp :: MatsubaraFrequency,
    P  :: BrillouinPoint,
    k  :: BrillouinPoint,
    kp :: BrillouinPoint,
       :: Type{Ch},
       :: Type{pSp}
    ;
    F0 :: Bool = true,
    γp :: Bool = true,
    γt :: Bool = true,
    γa :: Bool = true
    )  :: Q where {Q, Ch <: ChannelTag}

    val = zero(Q)

    # reference part, evaluated in full (the switches are not passed on)
    if F0
        val += F.F0(Ω, ν, νp, P, k, kp, Ch, pSp)
    end

    # each channel at frequencies and momenta converted to its own parametrization
    if γp
        val += F.γp(convert_frequency(Ω, ν, νp, Ch, pCh)...,
                    convert_momentum( P, k, kp, Ch, pCh)...)
    end

    if γt
        val += F.γt(convert_frequency(Ω, ν, νp, Ch, tCh)...,
                    convert_momentum( P, k, kp, Ch, tCh)...)
    end

    if γa
        val += F.γa(convert_frequency(Ω, ν, νp, Ch, aCh)...,
                    convert_momentum( P, k, kp, Ch, aCh)...)
    end

    return val
end



# Special cases where either ν or νp is an InfiniteMatsubaraFrequency, see the docstring above.
@inline function (F :: AbstractNonlocalVertex{Q})(
    Ω  :: MatsubaraFrequency,
    ν  :: Union{MatsubaraFrequency, InfiniteMatsubaraFrequency},
    νp :: Union{MatsubaraFrequency, InfiniteMatsubaraFrequency},
    P  :: BrillouinPoint,
    k  :: Union{BrillouinPoint, SWaveBrillouinPoint},
    kp :: Union{BrillouinPoint, SWaveBrillouinPoint},
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
        val += F.F0(Ω, ν, νp, P, k, kp, Ch, pSp)
    end

    # This function is called only if either ν or νp is an InfiniteMatsubaraFrequency.
    # Otherwise, the specific case of having all MatsubaraFrequency's is called.

    # If ν or νp is an InfiniteMatsubaraFrequency, reducible vertices is nonzero
    # only for the same channel evaluated.

    if Ch === pCh && γp
        val += F.γp(Ω, ν, νp, P, k, kp)
    end

    if Ch === tCh && γt
        val += F.γt(Ω, ν, νp, P, k, kp)
    end

    if Ch === aCh && γa
        val += F.γa(Ω, ν, νp, P, k, kp)
    end

    return val
end


# Crossed spin component (crossing symmetry, Lihm 2025, Eq. (S4a)), see the
# docstring above. In the p channel the outgoing legs are exchanged:
# (ν′, k′) → (Ω - ν′, P - k′). In the a and t channels the two channels swap
# roles, as in vertex.jl.
@inline function (F :: AbstractNonlocalVertex{Q})(
    Ω  :: MatsubaraFrequency,
    ν  :: Union{MatsubaraFrequency, InfiniteMatsubaraFrequency},
    νp :: Union{MatsubaraFrequency, InfiniteMatsubaraFrequency},
    P  :: BrillouinPoint,
    k  :: Union{BrillouinPoint, SWaveBrillouinPoint},
    kp :: Union{BrillouinPoint, SWaveBrillouinPoint},
       :: Type{Ch},
       :: Type{xSp}
    ;
    F0 :: Bool = true,
    γp :: Bool = true,
    γt :: Bool = true,
    γa :: Bool = true
    )  :: Q where {Q, Ch <: ChannelTag}

    if Ch === pCh
        return -F(Ω, ν, Ω - νp, P, k, P - kp, pCh, pSp; F0 = F0, γp = γp, γt = γa, γa = γt)

    elseif Ch === tCh
        return -F(Ω, ν, νp, P, k, kp, aCh, pSp; F0 = F0, γp = γp, γt = γa, γa = γt)

    elseif Ch === aCh
        return -F(Ω, ν, νp, P, k, kp, tCh, pSp; F0 = F0, γp = γp, γt = γa, γa = γt)

    else
        throw(ArgumentError("Invalid channel $Ch"))
    end
end

# Density component dSp = 2·pSp + xSp, see the docstring above.
@inline function (F :: AbstractNonlocalVertex{Q})(
    Ω  :: MatsubaraFrequency,
    ν  :: Union{MatsubaraFrequency, InfiniteMatsubaraFrequency},
    νp :: Union{MatsubaraFrequency, InfiniteMatsubaraFrequency},
    P  :: BrillouinPoint,
    k  :: Union{BrillouinPoint, SWaveBrillouinPoint},
    kp :: Union{BrillouinPoint, SWaveBrillouinPoint},
       :: Type{Ch},
       :: Type{dSp}
    ;
    F0 :: Bool = true,
    γp :: Bool = true,
    γt :: Bool = true,
    γa :: Bool = true
    )  :: Q where {Q, Ch <: ChannelTag}

    val = zero(Q)

    val += F(Ω, ν, νp, P, k, kp, Ch, pSp; F0, γp, γt, γa) * 2

    val += F(Ω, ν, νp, P, k, kp, Ch, xSp; F0, γp, γt, γa)

    return val
end

# -----------------------------------------------------------------------------
# s-wave evaluation (NL_Vertex with averaged fermionic momenta)
# -----------------------------------------------------------------------------
# With k and/or k′ averaged (kSW), the channel Ch the vertex is evaluated in
# keeps its bosonic momentum P, and the other two channels are evaluated at kSW,
# i.e. averaged over their own bosonic momentum. This needs a single average
# per channel instead of a sum over the averaged momentum.

# k averaged, see the docstring above
@inline function (F :: NL_Vertex{Q})(
    Ω  :: MatsubaraFrequency,
    ν  :: MatsubaraFrequency,
    νp :: MatsubaraFrequency,
    P  :: BrillouinPoint,
    k  :: SWaveBrillouinPoint,
    kp :: BrillouinPoint,
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
        val += F.F0(Ω, ν, νp, P, k, kp, Ch, pSp)
    end

    # k isa SWaveBrillouinPoint, sum over the k momentum.
    # We use the fact that NL_Vertex has only bosonic momentum dependence.

    if γp
        if Ch === pCh
            # Vertices in the same channel as Ch does not have any k dependence.
            # So we just use ordinary evaluation.
            val += F.γp(convert_frequency(Ω, ν, νp, Ch, pCh)..., P)
        else
            # Vertices in the different channel will be integrated over the bosonic momentum.
            val += F.γp(convert_frequency(Ω, ν, νp, Ch, pCh)..., SWaveBrillouinPoint())
        end
    end

    if γt
        if Ch === tCh
            val += F.γt(convert_frequency(Ω, ν, νp, Ch, tCh)..., P)
        else
            val += F.γt(convert_frequency(Ω, ν, νp, Ch, tCh)..., SWaveBrillouinPoint())
        end
    end

    if γa
        if Ch === aCh
            val += F.γa(convert_frequency(Ω, ν, νp, Ch, aCh)..., P)
        else
            val += F.γa(convert_frequency(Ω, ν, νp, Ch, aCh)..., SWaveBrillouinPoint())
        end
    end

    return val
end

# k′ averaged, see the docstring above
@inline function (F :: NL_Vertex{Q})(
    Ω  :: MatsubaraFrequency,
    ν  :: MatsubaraFrequency,
    νp :: MatsubaraFrequency,
    P  :: BrillouinPoint,
    k  :: BrillouinPoint,
    kp :: SWaveBrillouinPoint,
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
        val += F.F0(Ω, ν, νp, P, k, kp, Ch, pSp)
    end

    # kp isa SWaveBrillouinPoint, sum over the kp momentum.
    # We use the fact that NL_Vertex has only bosonic momentum dependence.

    if γp
        if Ch === pCh
            # Vertices in the same channel as Ch does not have any k dependence.
            # So we just use ordinary evaluation.
            val += F.γp(convert_frequency(Ω, ν, νp, Ch, pCh)..., P)
        else
            # Vertices in the different channel will be integrated over the bosonic momentum.
            val += F.γp(convert_frequency(Ω, ν, νp, Ch, pCh)..., SWaveBrillouinPoint())
        end
    end

    if γt
        if Ch === tCh
            val += F.γt(convert_frequency(Ω, ν, νp, Ch, tCh)..., P)
        else
            val += F.γt(convert_frequency(Ω, ν, νp, Ch, tCh)..., SWaveBrillouinPoint())
        end
    end

    if γa
        if Ch === aCh
            val += F.γa(convert_frequency(Ω, ν, νp, Ch, aCh)..., P)
        else
            val += F.γa(convert_frequency(Ω, ν, νp, Ch, aCh)..., SWaveBrillouinPoint())
        end
    end

    return val
end

# k and k′ averaged, see the docstring above
@inline function (F :: NL_Vertex{Q})(
    Ω  :: MatsubaraFrequency,
    ν  :: MatsubaraFrequency,
    νp :: MatsubaraFrequency,
    P  :: BrillouinPoint,
    k  :: SWaveBrillouinPoint,
    kp :: SWaveBrillouinPoint,
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
        val += F.F0(Ω, ν, νp, P, k, kp, Ch, pSp)
    end

    # k and kp are SWaveBrillouinPoint, sum over the k and kp momentum.
    # We use the fact that NL_Vertex has only bosonic momentum dependence.

    if γp
        if Ch === pCh
            # Vertices in the same channel as Ch does not have any k dependence.
            # So we just use ordinary evaluation.
            val += F.γp(convert_frequency(Ω, ν, νp, Ch, pCh)..., P)
        else
            # Vertices in the different channel will be integrated over the bosonic momentum.
            val += F.γp(convert_frequency(Ω, ν, νp, Ch, pCh)..., SWaveBrillouinPoint())
        end
    end

    if γt
        if Ch === tCh
            val += F.γt(convert_frequency(Ω, ν, νp, Ch, tCh)..., P)
        else
            val += F.γt(convert_frequency(Ω, ν, νp, Ch, tCh)..., SWaveBrillouinPoint())
        end
    end

    if γa
        if Ch === aCh
            val += F.γa(convert_frequency(Ω, ν, νp, Ch, aCh)..., P)
        else
            val += F.γa(convert_frequency(Ω, ν, νp, Ch, aCh)..., SWaveBrillouinPoint())
        end
    end

    return val
end
