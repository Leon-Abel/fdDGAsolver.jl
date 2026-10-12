# =============================================================================
# Multi-boson exchange (MBE) form of the vertex.
#
# The K3 class of each channel contains a part that factorizes into two K2
# classes joined by a boson propagator (Lihm 2025, Eqs. (S24)-(S25)):
#
#   K3_r = M_r + K2_r W_r⁻¹ K̄2_r,        W_r = U_r + K1_r,
#
# with U_r the bare vertex in that channel. The single-boson exchange (SBE)
# part S_r = U_r + K1_r + K2_r + K̄2_r + K2_r W_r⁻¹ K̄2_r describes one boson (a
# collective fluctuation) exchanged between the electrons; M_r is the
# multi-boson rest, typically more compact in frequency than K3_r. An MBE
# vertex (MBEVertex, NL2_MBEVertex) stores M_r in the K3 slot of its channels
# and evaluates each channel as
#
#   Φ_r = K1 + K2 + K̄2 + K2 K̄2 / (U_r + K1) + M_r.
#
# Its fd equation for M_r is Lihm 2025, Eq. (S31): the K3 kernels with the MBE
# caches below. The U-closed strategies :fdPA_new and :scPA_new (solve.jl) are
# meant for MBE vertices.
#
# The SBE factorization holds in the spin channels in which the BSE is
# diagonal. In the t channel these are the density D and magnetic M
# combinations, so the t channel is evaluated from its D and M parts
# (U_D = U, U_M = -U). The p channel uses the pSp form, which equals the
# singlet form when the triplet K1 and K2 vanish (as for the local vertex).
#
# Class tags (ClassTag) select one asymptotic class of a vertex:
#   K1Cl   K1(Ω)            K2Cl   K2(Ω, ν)         K2pCl  K̄2(Ω, ν′)
#   K3Cl   K3(Ω, ν, ν′)     ΛCl    the core of the RefVertex (without U), i.e.
#                                  the part that is not split into channels
# Evaluating a vertex at a class sums that class over the nested levels
# F → F.F0 → …, so the MBE formula acts on the total classes.
#
# Also here: conversion between the asymptotic and the MBE form
# (asymptotic_to_mbe, mbe_to_asymptotic) and the K3 caches of MBE solvers.
#
# Paper ↔ code: ω ↔ Ω; ν′ ↔ νp (or ω); q ↔ P; K̄2 ↔ K2' (class K2pCl);
# S_r ↔ ∇; M_r ↔ the K3 slot of an MBE channel; Φ_r ↔ the fields γp, γt, γa.
# =============================================================================

# -----------------------------------------------------------------------------
# Class tags
# -----------------------------------------------------------------------------

"""
    abstract type ClassTag end

Tag for one asymptotic class of a vertex (see the file header). Passed in
place of the spin tag to evaluate a single class.
"""
abstract type ClassTag end

"""
    struct K1Cl <: ClassTag end

K1 class: K1(Ω), depends on the bosonic frequency only.
"""
struct K1Cl <: ClassTag end

"""
    struct K2Cl <: ClassTag end

K2 class: K2(Ω, ν), depends on Ω and the first fermionic frequency ν.
"""
struct K2Cl <: ClassTag end

"""
    struct K2pCl <: ClassTag end

K̄2 class (K2'): K̄2(Ω, ν′), depends on Ω and the second fermionic frequency
ν′. It is read from the same K2 array (K̄2 = K2 by the symmetry ν ↔ ν′).
"""
struct K2pCl <: ClassTag end

"""
    struct K3Cl <: ClassTag end

K3 class: K3(Ω, ν, ν′). In an MBE vertex this slot holds the multi-boson rest
`M_r`.
"""
struct K3Cl <: ClassTag end

"""
    struct ΛCl <: ClassTag end

Λ class: the part of the vertex that is not split into channels, i.e. the
core of the RefVertex (without U). It is zero in the PA. For DMFT input it
holds `(Λ - U) + ∑_r K3_r` (refvertex.jl); after [`asymptotic_to_mbe`](@ref) it
holds `(Λ - U) + ∑_r M_r`.
"""
struct ΛCl <: ClassTag end

# Maps each asymptotic class to its counterpart under the crossing (exchange) symmetry.
# K2 ↔ K2' because crossing swaps ν ↔ ν'; all other classes are self-conjugate.
# Not used anywhere. (BSE_templates.jl defines an unrelated _crossing for
# frequencies and momenta.)
_crossing(::Type{K1Cl})  = K1Cl
_crossing(::Type{K2Cl})  = K2pCl
_crossing(::Type{K2pCl}) = K2Cl
_crossing(::Type{K3Cl})  = K3Cl
_crossing(::Type{ΛCl})   = ΛCl

# -----------------------------------------------------------------------------
# Class evaluators: one class of one channel or of a RefVertex
# -----------------------------------------------------------------------------

"""
    (γ::Channel)(Ω, ν, ν', P, k, k', ::Type{Cl})
    (γ::Channel)(Ω, ν, ν', ::Type{Cl})

Evaluates the channel `γ` at the given asymptotic class `Cl`.
Momentum arguments are accepted but ignored for the local `Channel` type.
Returns zero for the `ΛCl` class (that part lives in the RefVertex).
"""
@inline function (γ :: Channel{Q})(
    Ω  :: Union{MatsubaraFrequency, InfiniteMatsubaraFrequency},
    ν  :: Union{MatsubaraFrequency, InfiniteMatsubaraFrequency},
    νp :: Union{MatsubaraFrequency, InfiniteMatsubaraFrequency},
       :: BrillouinPoint,
       :: BrillouinPoint,
       :: BrillouinPoint,
       :: Type{Cl},
    )  :: Q where {Q, Cl <: ClassTag}

    if Cl === K1Cl
        return γ.K1(Ω)

    elseif Cl === K2Cl
        return γ.K2(Ω, ν)

    elseif Cl === K2pCl
        return γ.K2(Ω, νp)

    elseif Cl === K3Cl
        return γ.K3(Ω, ν, νp)

    elseif Cl === ΛCl
        return zero(Q)

    else
        throw(ArgumentError("Invalid class tag $Cl"))
    end

end

# Channel class evaluator without momenta, see the docstring above.
@inline function (γ :: Channel{Q})(
    Ω  :: Union{MatsubaraFrequency, InfiniteMatsubaraFrequency},
    ν  :: Union{MatsubaraFrequency, InfiniteMatsubaraFrequency},
    νp :: Union{MatsubaraFrequency, InfiniteMatsubaraFrequency},
       :: Type{Cl},
    )  :: Q where {Q, Cl <: ClassTag}

    if Cl === K1Cl
        return γ.K1(Ω)

    elseif Cl === K2Cl
        return γ.K2(Ω, ν)

    elseif Cl === K2pCl
        return γ.K2(Ω, νp)

    elseif Cl === K3Cl
        return γ.K3(Ω, ν, νp)

    elseif Cl === ΛCl
        return zero(Q)

    else
        throw(ArgumentError("Invalid class tag $Cl"))
    end

end

"""
    (γ::ChannelViewX2X)(Ω, ν, ν', ::Type{Cl})

Evaluates the `ChannelViewX2X` channel at the given asymptotic class `Cl`.
Unlike `Channel`, this view stores K2 and K2' separately: with the momenta
fixed, they are slices of the same K2 array at different fermionic momenta
(k and k′; `fixed_momentum_view.jl`).
"""
@inline function (γ :: ChannelViewX2X{Q})(
    Ω  :: Union{MatsubaraFrequency, InfiniteMatsubaraFrequency},
    ν  :: Union{MatsubaraFrequency, InfiniteMatsubaraFrequency},
    νp :: Union{MatsubaraFrequency, InfiniteMatsubaraFrequency},
       :: Type{Cl},
    )  :: Q where {Q, Cl <: ClassTag}

    if Cl === K1Cl
        return γ.K1(Ω)

    elseif Cl === K2Cl
        return γ.K2(Ω, ν)

    # K2' stored separately (slice at the second fermionic momentum)
    elseif Cl === K2pCl
        return γ.K2p(Ω, νp)

    elseif Cl === K3Cl
        return γ.K3(Ω, ν, νp)

    elseif Cl === ΛCl
        return zero(Q)

    else
        throw(ArgumentError("Invalid class tag $Cl"))
    end

end


"""
    (γ::NL2_Channel)(Ω, ν, ν', P, k, k', ::Type{Cl})

Evaluates the nonlocal channel `γ` at the given asymptotic class `Cl`.
The bosonic momentum `P` enters K1–K3; the fermionic momenta `k`, `k'` enter K2, K2'.
K3 is s-wave (no fermionic momenta). Returns zero for the `ΛCl` class.
"""
@inline function (γ :: NL2_Channel{Q})(
    Ω  :: Union{MatsubaraFrequency, InfiniteMatsubaraFrequency},
    ν  :: Union{MatsubaraFrequency, InfiniteMatsubaraFrequency},
    νp :: Union{MatsubaraFrequency, InfiniteMatsubaraFrequency},
    P  :: BrillouinPoint,
    k  :: Union{BrillouinPoint, SWaveBrillouinPoint},
    kp :: Union{BrillouinPoint, SWaveBrillouinPoint},
       :: Type{Cl},
    )  :: Q where {Q, Cl <: ClassTag}

    if Cl === K1Cl
        return γ.K1(Ω, P)

    elseif Cl === K2Cl
        return γ.K2(Ω, ν, P, k)

    elseif Cl === K2pCl
        return γ.K2(Ω, νp, P, kp)

    elseif Cl === K3Cl
        return γ.K3(Ω, ν, νp, P)

    elseif Cl === ΛCl
        return zero(Q)

    else
        throw(ArgumentError("Invalid class tag $Cl"))
    end

end

"""
    (F::RefVertex)(Ω, ν, ν', P, k, k', ::Type{pCh}, ::Type{Cl})
    (F::RefVertex)(Ω, ν, ν', P, k, k', ::Type{tCh}, ::Type{Cl})
    (F::RefVertex)(Ω, ν, ν', P, k, k', ::Type{aCh}, ::Type{Cl})

Class evaluator of the RefVertex, parallel spin component, in the
parametrization of the p (parallel), t (transverse) or a (antiparallel)
channel. For `ΛCl` it returns the core, read from the p- or t-channel box,
without U (the MBE evaluator adds U itself). All other classes return zero,
because the RefVertex is not split into channels. Momenta are ignored.
"""
@inline function (F :: RefVertex{Q})(
    Ω  :: Union{MatsubaraFrequency, InfiniteMatsubaraFrequency},
    ν  :: Union{MatsubaraFrequency, InfiniteMatsubaraFrequency},
    νp :: Union{MatsubaraFrequency, InfiniteMatsubaraFrequency},
       :: BrillouinPoint,
       :: Union{BrillouinPoint, SWaveBrillouinPoint},
       :: Union{BrillouinPoint, SWaveBrillouinPoint},
       :: Type{pCh},
       :: Type{Cl},
    ; kwargs...
    )  :: Q where {Q, Cl <: ClassTag}

    if Cl === ΛCl
        return F.Fp_p(Ω, ν, νp)
    else
        return zero(Q)
    end
end

# t channel: core from the t-channel box, see the docstring above.
@inline function (F :: RefVertex{Q})(
    Ω  :: Union{MatsubaraFrequency, InfiniteMatsubaraFrequency},
    ν  :: Union{MatsubaraFrequency, InfiniteMatsubaraFrequency},
    νp :: Union{MatsubaraFrequency, InfiniteMatsubaraFrequency},
       :: BrillouinPoint,
       :: Union{BrillouinPoint, SWaveBrillouinPoint},
       :: Union{BrillouinPoint, SWaveBrillouinPoint},
       :: Type{tCh},
       :: Type{Cl},
    ; kwargs...
    )  :: Q where {Q, Cl <: ClassTag}

    if Cl === ΛCl
        return F.Ft_p(Ω, ν, νp)
    else
        return zero(Q)
    end
end




# a channel: crossing symmetry with the t channel, as in refvertex.jl,
#   Λ^=_a(Ω, ν, ν′) = -Λ^×_t(Ω, ν′, ν)
# (the swap ν ↔ ν′ and the minus sign). See the docstring above.
@inline function (F :: RefVertex{Q})(
    Ω  :: Union{MatsubaraFrequency, InfiniteMatsubaraFrequency},
    ν  :: Union{MatsubaraFrequency, InfiniteMatsubaraFrequency},
    νp :: Union{MatsubaraFrequency, InfiniteMatsubaraFrequency},
       :: BrillouinPoint,
       :: Union{BrillouinPoint, SWaveBrillouinPoint},
       :: Union{BrillouinPoint, SWaveBrillouinPoint},
       :: Type{aCh},
       :: Type{Cl},
    ; kwargs...
    )  :: Q where {Q, Cl <: ClassTag}

    if Cl === ΛCl
        return -F.Ft_x(Ω, νp, ν)
    else
        return zero(Q)
    end
end


# -----------------------------------------------------------------------------
# MBE vertex types
# -----------------------------------------------------------------------------

"""
    abstract type AbstractMBEVertex{Q} <: AbstractVertex{Q}

Abstract base type for MBE vertices.  Concrete subtypes store a reference vertex `F0`
and three channel-reducible contributions `γp`, `γt`, `γa`.
"""
abstract type AbstractMBEVertex{Q} <: AbstractVertex{Q} end

"""
    struct MBEVertex{Q, VT} <: AbstractMBEVertex{Q}

Local (momentum-independent) MBE vertex.  Holds a reference vertex `F0` of type `VT`
and three local `Channel{Q}` objects for the p, t, and a channels. The K3 slots
of the channels hold the multi-boson rest `M_r` (see the file header).
"""
struct MBEVertex{Q, VT} <: AbstractMBEVertex{Q}
    F0 :: VT
    γp :: Channel{Q}
    γt :: Channel{Q}
    γa :: Channel{Q}

    # build MBEVertex from reference Vertex and pre-defined channels
    function MBEVertex(
        F0 :: VT,
        γp :: Channel{Q},
        γt :: Channel{Q},
        γa :: Channel{Q},
        ) where {Q, VT}

        return new{Q, VT}(F0, γp, γt, γa)
    end

    # build MBEVertex from reference Vertex and frequency-box sizes
    # channels are initialized empty
    function MBEVertex(
        F0    :: VT,
        T     :: Float64,
        numK1 :: Int64,
        numK2 :: NTuple{2, Int64},
        numK3 :: NTuple{2, Int64},
        ) where {VT}

        Q = eltype(F0)

        γ = Channel(T, numK1, numK2, numK3, Q)
        return new{Q, VT}(F0, γ, copy(γ), copy(γ))
    end
end

# An MBEVertex stores its channels as Channel objects (used by load_vertex).
channel_type(::Type{MBEVertex}) = Channel

"""
    (F::Union{Channel,RefVertex,Vertex})(Ω, ν, ν', P, k, k', ::Type{Ch}, ::Type{Cl})

Evaluates the vertex `F` at the given channel `Ch` and asymptotic class `Cl` by
adding the reference vertex contribution `F.F0` and the channel-reducible
part `γr` for channel `r = Ch`. In practice this serves the local `Vertex` (the
RefVertex and the channels have their own methods above); `F.F0` is evaluated
the same way, so the class is summed over all nested levels.
"""
@inline function (F :: Union{Channel{Q}, RefVertex{Q}, Vertex{Q}})(
    Ω  :: Union{MatsubaraFrequency, InfiniteMatsubaraFrequency},
    ν  :: Union{MatsubaraFrequency, InfiniteMatsubaraFrequency},
    νp :: Union{MatsubaraFrequency, InfiniteMatsubaraFrequency},
    P  :: BrillouinPoint,
    k  :: Union{BrillouinPoint, SWaveBrillouinPoint},
    kp :: Union{BrillouinPoint, SWaveBrillouinPoint},
       :: Type{Ch},
       :: Type{Cl},
    )  :: Q where {Q, Ch <: ChannelTag, Cl <: ClassTag}

    # evaluate the Vertex `F` and its RefVertex `F.F0`
    # at the pSp spin component, channel `Ch` and asymptotic class `Cl`

    val = F.F0(Ω, ν, νp, P, k, kp, Ch, Cl)

    if Ch === pCh
        val += F.γp(Ω, ν, νp, P, k, kp, Cl)
    end

    if Ch === tCh
        val += F.γt(Ω, ν, νp, P, k, kp, Cl)
    end

    if Ch === aCh
        val += F.γa(Ω, ν, νp, P, k, kp, Cl)
    end

    return val
end

"""
    (F::AbstractVertex)(Ω, ν, ν', P, k, k', ::Type{Ch}, ::Type{Cl})

Generic fallback — same structure as the `Channel`/`RefVertex`/`Vertex`
overload above, but accepts any `AbstractVertex` subtype and allows `SWaveBrillouinPoint`
for the bosonic momentum `P`.
"""
@inline function (F :: AbstractVertex{Q})(
    Ω  :: Union{MatsubaraFrequency, InfiniteMatsubaraFrequency},
    ν  :: Union{MatsubaraFrequency, InfiniteMatsubaraFrequency},
    νp :: Union{MatsubaraFrequency, InfiniteMatsubaraFrequency},
    P  :: Union{BrillouinPoint, SWaveBrillouinPoint},
    k  :: Union{BrillouinPoint, SWaveBrillouinPoint},
    kp :: Union{BrillouinPoint, SWaveBrillouinPoint},
       :: Type{Ch},
       :: Type{Cl},
    )  :: Q where {Q, Ch <: ChannelTag, Cl <: ClassTag}

    # evaluate the Vertex `F` and its RefVertex `F.F0`
    # at the pSp spin component, channel `Ch` and asymptotic class `Cl`

    val = F.F0(Ω, ν, νp, P, k, kp, Ch, Cl)

    if Ch === pCh
        val += F.γp(Ω, ν, νp, P, k, kp, Cl)
    end

    if Ch === tCh
        val += F.γt(Ω, ν, νp, P, k, kp, Cl)
    end

    if Ch === aCh
        val += F.γa(Ω, ν, νp, P, k, kp, Cl)
    end

    return val
end

"""
    (F::MBEVertex)(Ω, ν, ν', ::Type{Ch}, ::Type{Sp}; F0, γp, γt, γa)

Convenience wrapper for the local `MBEVertex`: delegates to the momentum-resolved
evaluator with all momenta set to the zero point `k0`.
"""
function (F::MBEVertex{Q})(
    Ω  :: Union{MatsubaraFrequency, InfiniteMatsubaraFrequency},
    ν  :: Union{MatsubaraFrequency, InfiniteMatsubaraFrequency},
    νp :: Union{MatsubaraFrequency, InfiniteMatsubaraFrequency},
       :: Type{Ch},
       :: Type{Sp},
    ;
    F0 :: Bool = true,
    γp :: Bool = true,
    γt :: Bool = true,
    γa :: Bool = true,
    )  :: Q where {Q, Ch <: ChannelTag, Sp <: SpinTag}

    return F(Ω, ν, νp, k0, k0, k0, Ch, Sp; F0, γp, γt, γa)
end


# -----------------------------------------------------------------------------
# Evaluation of an MBE vertex
# -----------------------------------------------------------------------------

"""
    (F::AbstractMBEVertex)(Ω, ν, ν', P, k, k', ::Type{Ch}, ::Type{pSp}; F0, γp, γt, γa)

Evaluates the full MBE vertex in the parallel-spin (pSp) component using:

    F = U + γp + γt + γa + Λ

Each channel contribution follows the SBE formula

    γr = K1 + K2 + K2' + K2·K2'/(U + K1) + K3

converted to the target channel `Ch` via frequency/momentum transformations.
Here K3 is the stored `M_r`, and each class is the total over the nested levels
`F → F.F0 → …`, so the formula acts on the total vertex. Λ is the core of the
RefVertex at the bottom (class `ΛCl`).

The t-channel contribution is split into density (D) and magnetic (M) components,
in which the SBE formula holds (with U and -U):
`Φ_t^= = (Φ_D - Φ_M)/2`, with `Φ_M = -Φ_a^=` and `Φ_D = 2Φ_t^= - Φ_a^=` (crossing).

The keywords `γp`, `γt`, `γa` switch channels off. `F0 = false` subtracts
`F.F0` evaluated with the same switches, so it returns the difference between
F and its reference vertex, e.g. `Ĩ_r` = (F - f) with channel r off. This is
how the fd (irreducible) vertices are computed for MBE vertices.
"""
function (F::AbstractMBEVertex{Q})(
    Ω  :: Union{MatsubaraFrequency, InfiniteMatsubaraFrequency},
    ν  :: Union{MatsubaraFrequency, InfiniteMatsubaraFrequency},
    νp :: Union{MatsubaraFrequency, InfiniteMatsubaraFrequency},
    P  :: BrillouinPoint,
    k  :: BrillouinPoint,
    kp :: BrillouinPoint,
       :: Type{Ch},
       :: Type{Sp},
    ;
    F0 :: Bool = true,
    γp :: Bool = true,
    γt :: Bool = true,
    γa :: Bool = true,
    )  :: Q where {Q, Ch <: ChannelTag, Sp <: pSp}

    val = zero(Q)

    # bare vertex U (pSp)
    U = bare_vertex(F, Sp)

    val += U

    # p channel: total classes at the converted arguments, SBE formula
    if γp
        ωs = convert_frequency(Ω, ν, νp, Ch, pCh)
        ks = convert_momentum( P, k, kp, Ch, pCh)
        K1  = F(ωs..., ks..., pCh, K1Cl)
        K2  = F(ωs..., ks..., pCh, K2Cl)
        K2p = F(ωs..., ks..., pCh, K2pCl)
        K3  = F(ωs..., ks..., pCh, K3Cl)
        val += K1 + K2 + K2p + K2 * K2p / (U + K1) + K3
    end

    if γt
        # tCh, pSp =  1/2 * (tCh, D) - 1/2 * (tCh, M)
        # (tCh, M) = - (aCh, pSp)
        # (tCh, D) = 2 * (tCh, pSp) - (aCh, pSp)
        ωs = convert_frequency(Ω, ν, νp, Ch, tCh)
        ks = convert_momentum( P, k, kp, Ch, tCh)

        # Magnetic channel
        K1  = - F(ωs..., ks..., aCh, K1Cl)
        K2  = - F(ωs..., ks..., aCh, K2Cl)
        K2p = - F(ωs..., ks..., aCh, K2pCl)
        K3  = - F(ωs..., ks..., aCh, K3Cl)
        val -= (K1 + K2 + K2p + K2 * K2p / (-U + K1) + K3) / 2

        # Density channel
        K1  = 2 * F(ωs..., ks..., tCh, K1Cl)  + K1
        K2  = 2 * F(ωs..., ks..., tCh, K2Cl)  + K2
        K2p = 2 * F(ωs..., ks..., tCh, K2pCl) + K2p
        K3  = 2 * F(ωs..., ks..., tCh, K3Cl)  + K3
        val += (K1 + K2 + K2p + K2 * K2p / (U + K1) + K3) / 2
    end

    # a channel: as for p
    if γa
        ωs = convert_frequency(Ω, ν, νp, Ch, aCh)
        ks = convert_momentum( P, k, kp, Ch, aCh)
        K1  = F(ωs..., ks..., aCh, K1Cl)
        K2  = F(ωs..., ks..., aCh, K2Cl)
        K2p = F(ωs..., ks..., aCh, K2pCl)
        K3  = F(ωs..., ks..., aCh, K3Cl)

        val += K1 + K2 + K2p + K2 * K2p / (U + K1) + K3
    end

    # Add 2-particle irreducible contribution
    # (the core of the RefVertex, class ΛCl, without U)
    val += F.F0(Ω, ν, νp, P, k, kp, Ch, ΛCl)

    # F0 = false: difference to the reference vertex, with the same switches
    if F0 == false
        val -= F.F0(Ω, ν, νp, P, k, kp, Ch, Sp; γp, γt, γa)
    end

    return val
end

"""
    (F::AbstractMBEVertex)(Ω, ν, ν', P, k, k', ::Type{Ch}, ::Type{xSp}; F0, γp, γt, γa)

Evaluates the vertex in the crossed spin component (xSp), by crossing symmetry
(Lihm 2025, Eq. (S4a)): `-F^=` at (Ω, ν, Ω - ν′; P, k, P - k′) in the p channel,
and with the a and t channels exchanged in the a and t channels. The switches
for t and a swap accordingly, as in nonlocal/vertex.jl.
"""
@inline function (F :: AbstractMBEVertex{Q})(
    Ω  :: Union{MatsubaraFrequency, InfiniteMatsubaraFrequency},
    ν  :: Union{MatsubaraFrequency, InfiniteMatsubaraFrequency},
    νp :: Union{MatsubaraFrequency, InfiniteMatsubaraFrequency},
    P  :: BrillouinPoint,
    k  :: BrillouinPoint,
    kp :: BrillouinPoint,
       :: Type{Ch},
       :: Type{xSp}
    ;
    F0 :: Bool = true,
    γp :: Bool = true,
    γt :: Bool = true,
    γa :: Bool = true
    )  :: Q where {Q, Ch <: ChannelTag}

    # Use crossing symmetry to map xSp to pSp.

    if Ch === pCh
        return -F(Ω, ν, Ω - νp, P, k, P - kp, pCh, pSp; F0, γp, γt = γa, γa = γt)

    elseif Ch === tCh
        return -F(Ω, ν, νp, P, k, kp, aCh, pSp; F0, γp, γt = γa, γa = γt)

    elseif Ch === aCh
        return -F(Ω, ν, νp, P, k, kp, tCh, pSp; F0, γp, γt = γa, γa = γt)

    else
        throw(ArgumentError("Invalid channel tag $Ch"))
    end
end


# evaluators for density spin component
"""
    (F::AbstractMBEVertex)(Ω, ν, ν', P, k, k', ::Type{Ch}, ::Type{dSp}; F0, γp, γt, γa)

Evaluates the vertex in the density spin component (dSp) using
    dSp = 2·pSp + xSp.
"""
@inline function (F :: AbstractMBEVertex{Q})(
    Ω  :: Union{MatsubaraFrequency, InfiniteMatsubaraFrequency},
    ν  :: Union{MatsubaraFrequency, InfiniteMatsubaraFrequency},
    νp :: Union{MatsubaraFrequency, InfiniteMatsubaraFrequency},
    P  :: BrillouinPoint,
    k  :: BrillouinPoint,
    kp :: BrillouinPoint,
       :: Type{Ch},
       :: Type{dSp}
    ;
    F0 :: Bool = true,
    γp :: Bool = true,
    γt :: Bool = true,
    γa :: Bool = true
    )  :: Q where {Q, Ch <: ChannelTag}

    return ( 2 * F(Ω, ν, νp, P, k, kp, Ch, pSp; F0, γp, γt, γa)
               + F(Ω, ν, νp, P, k, kp, Ch, xSp; F0, γp, γt, γa) )
end



"""
    (F::AbstractMBEVertex)(Ω, ν, ν', P, k, kp::SWaveBrillouinPoint, ::Type{Ch}, ::Type{Sp})
    (F::AbstractMBEVertex)(Ω, ν, ν', P, k::SWaveBrillouinPoint, kp, ::Type{Ch}, ::Type{Sp})
    (F::AbstractMBEVertex)(Ω, ν, ν', P, k::SWaveBrillouinPoint, kp::SWaveBrillouinPoint, ::Type{Ch}, ::Type{Sp})

Evaluates the vertex with one or both fermionic momenta averaged over the full Brillouin
zone (s-wave projection).  A `SWaveBrillouinPoint` argument triggers averaging over the
momentum mesh returned by `get_P_mesh(F)`. The average is an explicit sum over
the vertex values: the MBE formula is not linear in K2, so the classes cannot
be averaged first. With both momenta averaged this is a double sum, which is
expensive.
"""
function (F::AbstractMBEVertex{Q})(
    Ω  :: Union{MatsubaraFrequency, InfiniteMatsubaraFrequency},
    ν  :: Union{MatsubaraFrequency, InfiniteMatsubaraFrequency},
    νp :: Union{MatsubaraFrequency, InfiniteMatsubaraFrequency},
    P  :: BrillouinPoint,
    k  :: BrillouinPoint,
    kp :: SWaveBrillouinPoint,
       :: Type{Ch},
       :: Type{Sp},
    ;
    F0 :: Bool = true,
    γp :: Bool = true,
    γt :: Bool = true,
    γa :: Bool = true,
    )  :: Q where {Q, Ch <: ChannelTag, Sp <: SpinTag}

    val = zero(Q)

    mK = get_P_mesh(F)

    for q in mK
        val += F(Ω, ν, νp, P, k, value(q), Ch, Sp; F0, γp, γt, γa)
    end

    return val / length(mK)
end

# k averaged, see the docstring above.
function (F::AbstractMBEVertex{Q})(
    Ω  :: Union{MatsubaraFrequency, InfiniteMatsubaraFrequency},
    ν  :: Union{MatsubaraFrequency, InfiniteMatsubaraFrequency},
    νp :: Union{MatsubaraFrequency, InfiniteMatsubaraFrequency},
    P  :: BrillouinPoint,
    k  :: SWaveBrillouinPoint,
    kp :: BrillouinPoint,
       :: Type{Ch},
       :: Type{Sp},
    ;
    F0 :: Bool = true,
    γp :: Bool = true,
    γt :: Bool = true,
    γa :: Bool = true,
    )  :: Q where {Q, Ch <: ChannelTag, Sp <: SpinTag}

    val = zero(Q)

    mK = get_P_mesh(F)

    for q in mK
        val += F(Ω, ν, νp, P, value(q), kp, Ch, Sp; F0, γp, γt, γa)
    end

    return val / length(mK)
end

# k and k′ averaged, see the docstring above.
function (F::AbstractMBEVertex{Q})(
    Ω  :: Union{MatsubaraFrequency, InfiniteMatsubaraFrequency},
    ν  :: Union{MatsubaraFrequency, InfiniteMatsubaraFrequency},
    νp :: Union{MatsubaraFrequency, InfiniteMatsubaraFrequency},
    P  :: BrillouinPoint,
    k  :: SWaveBrillouinPoint,
    kp :: SWaveBrillouinPoint,
       :: Type{Ch},
       :: Type{Sp},
    ;
    F0 :: Bool = true,
    γp :: Bool = true,
    γt :: Bool = true,
    γa :: Bool = true,
    )  :: Q where {Q, Ch <: ChannelTag, Sp <: SpinTag}

    val = zero(Q)

    mK = get_P_mesh(F)

    for qp in mK, q in mK
        val += F(Ω, ν, νp, P, value(q), value(qp), Ch, Sp; F0, γp, γt, γa)
    end

    return val / length(mK)^2
end


"""
    Base.copy(F::MBEVertex) -> MBEVertex

Returns a deep copy of `F`, with all fields copied independently.
"""
function Base.:copy(
    F :: MBEVertex{Q}
    ) :: MBEVertex{Q} where {Q}

    return MBEVertex(copy(F.F0), copy(F.γp), copy(F.γt), copy(F.γa))
end


"""
    get_reducible_vertex(F::AbstractVertex, ::Type{Ch})

Returns the channel-reducible part `γr` of the vertex `F` for channel `r = Ch`,
i.e. the field `γa`, `γp` or `γt`. Works for every vertex type with these
fields.
"""
function get_reducible_vertex(
    F :: AbstractVertex,
      :: Type{Ch},
    ) where {Ch <: ChannelTag}

    if Ch === aCh
        return F.γa
    elseif Ch === pCh
        return F.γp
    elseif Ch === tCh
        return F.γt
    else
        throw(ArgumentError("Invalid channel tag $Ch"))
    end
end



# -----------------------------------------------------------------------------
# K3 caches of the local MBE solver
# -----------------------------------------------------------------------------

"""
    build_K3_cache!(S::ParquetSolver{Q, <:MBEVertex})

Precomputes and caches the vertex combinations needed to update K3 in the MBE parquet
equations. Three classes of cached quantities are filled:

1. `cache_F0*` — channel-U-irreducible vertex of the reference system `S.F0`:
   ``T_r = I_r - U + M_r``
2. `cache_Γ*`  — irreducible finite-difference vertex of the target system `S.F`
   (evaluated with `F0=false` and the corresponding `γr=false`)
3. `cache_Fp/Fa/Ft` — channel-U-irreducible vertex of the target system `S.F`:
   ``T_r = I_r - U + M_r``

`T_r = F - S_r` is the vertex minus its single-boson exchange part `S_r` in channel
r (f - `s_r` for the reference). These take the place of `f - f_012` and `F - F_01̄2`
of the plain caches (`build_K3_cache.jl`), and with them the K3 kernels give the
multi-boson part `M̃_r` (Lihm 2025, Eq. (S31)). For an MBE vertex `F0 = false`
means `F - F.F0`, so `cache_Γ*` is `Ĩ_r`.

Caches indexed by `(Ω, ω, ν')` are used for bubble contractions from the left;
caches indexed by `(Ω, ν, ω)` are used for contractions from the right.
The t-channel entries are converted from the pSp component to the density (dSp)
component before being stored.  All caches are reduced across MPI ranks.
"""
function build_K3_cache!(
    S :: ParquetSolver{Q, <: MBEVertex}
    ) where {Q}

    U = bare_vertex(S.F)

    set!(S.cache_Γpx, 0)
    set!(S.cache_F0p, 0)
    set!(S.cache_F0a, 0)
    set!(S.cache_F0t, 0)
    set!(S.cache_Γpp, 0)
    set!(S.cache_Γa,  0)
    set!(S.cache_Γt,  0)
    set!(S.cache_Fp,  0)
    set!(S.cache_Fa,  0)
    set!(S.cache_Ft,  0)

    # Vertices multiplied by bubbles to the left (by ω)
    # Γpx : Target, irreducible vertex in the p channel, xSp component.

    Threads.@threads for i in mpi_split(1 : length(S.cache_Γpx.data))
        Ω, ω, νp = value.(MatsubaraFunctions.to_meshes(S.cache_Γpx, i))

        S.cache_Γpx[i] = S.F( Ω, ω, νp, pCh, xSp; F0=false, γp=false)

        # (Iᵣ - U) + Mᵣ
        # (p channel in xSp: the bare part is -U, and M_p^×(Ω, ν̃, ν′) = -M_p^=(Ω, ν̃, Ω - ν′))
        S.cache_F0p[i] = S.F0(Ω, ω, νp, pCh, xSp; γp = false) + U + S.F0(Ω, ω, Ω - νp, k0, k0, k0, pCh, K3Cl) * -1
        S.cache_F0a[i] = S.F0(Ω, ω, νp, aCh, pSp; γa = false) - U + S.F0(Ω, ω, νp, k0, k0, k0, aCh, K3Cl)
        S.cache_F0t[i] = S.F0(Ω, ω, νp, tCh, pSp; γt = false) - U + S.F0(Ω, ω, νp, k0, k0, k0, tCh, K3Cl)

        # Convert from pSp (parallel spin component) to dSp (density component)
        # using the relation dSp = 2 * pSp + xSp = 2 * pSp - pSp(a) (crossing symmetry)
        S.cache_F0t[i] = 2 * S.cache_F0t[i] - S.cache_F0a[i]
    end

    mpi_allreduce!(S.cache_Γpx)
    mpi_allreduce!(S.cache_F0p)
    mpi_allreduce!(S.cache_F0a)
    mpi_allreduce!(S.cache_F0t)


    # Vertices multiplied by bubbles from the right (by ω)

    Threads.@threads for i in mpi_split(1 : length(S.cache_Γpp.data))
        Ω, ν, ω = value.(MatsubaraFunctions.to_meshes(S.cache_Γpp, i))

        # r-irreducible vertex in each channel r = p, a, t
        S.cache_Γpp[i] = S.F(Ω, ν, ω, pCh, pSp; F0=false, γp=false)
        S.cache_Γa[i]  = S.F(Ω, ν, ω, aCh, pSp; F0=false, γa=false)
        S.cache_Γt[i]  = S.F(Ω, ν, ω, tCh, pSp; F0=false, γt=false)

        # (Iᵣ - U) + Mᵣ
        S.cache_Fp[i]  = S.F(Ω, ν, ω, pCh, pSp; γp = false) - U + S.F(Ω, ν, ω, k0, k0, k0, pCh, K3Cl)
        S.cache_Fa[i]  = S.F(Ω, ν, ω, aCh, pSp; γa = false) - U + S.F(Ω, ν, ω, k0, k0, k0, aCh, K3Cl)
        S.cache_Ft[i]  = S.F(Ω, ν, ω, tCh, pSp; γt = false) - U + S.F(Ω, ν, ω, k0, k0, k0, tCh, K3Cl)

        # Convert from pSp (parallel spin component) to dSp (density component)
        # using the relation dSp = 2 * pSp + xSp = 2 * pSp - pSp(a) (crossing symmetry)
        S.cache_Γt[i] = S.cache_Γt[i] * 2 - S.cache_Γa[i]
        S.cache_Ft[i] = S.cache_Ft[i] * 2 - S.cache_Fa[i]
    end

    mpi_allreduce!(S.cache_Γpp)
    mpi_allreduce!(S.cache_Γa)
    mpi_allreduce!(S.cache_Γt)
    mpi_allreduce!(S.cache_Fp)
    mpi_allreduce!(S.cache_Fa)
    mpi_allreduce!(S.cache_Ft)

    return nothing
end


# -----------------------------------------------------------------------------
# Conversion between the asymptotic and the MBE form
# -----------------------------------------------------------------------------

"""
    asymptotic_to_mbe(F::Vertex) -> MBEVertex
    asymptotic_to_mbe(F::NL2_Vertex) -> NL2_MBEVertex

Converts a vertex stored in the asymptotic (K1/K2/K2'/K3) representation to the MBE
representation by removing the single-boson-exchange (SBE) contribution from K3 (or
from the reference vertex `F0` when K3 is smaller than `F0`), so that the residual K3
is the multi-boson rest `M_r` beyond the SBE part `U + K1 + K2 + K2' + K2·K2'/(U + K1)`.

The second case is the one taken for the DMFT vertex (one-point K3 box, see
`load_triqs.jl`): there the SBE terms of all channels are taken out of the core,
which keeps the total vertex unchanged.

NOTE: the K3 case (first case) is exact only if `F.F0` is a bare RefVertex (zero
core, as in the PA). Otherwise the evaluation of ∇ also contains the core (and,
for a nested `F.F0`, its channels), which then is subtracted from K3 in every
channel.
"""
function asymptotic_to_mbe(F :: Vertex)
    F_mbe = MBEVertex(copy(F.F0), copy(F.γp), copy(F.γt), copy(F.γa))

    U = bare_vertex(F_mbe)

    if numK3(F.γa) >= numK3(F.F0)
        # Subtract SBE contribution from K3
        # (∇ = U + K1 + K2 + K2' + K2·K2'/(U + K1), plus the core of F.F0, see the NOTE)
        for Ch in (aCh, pCh, tCh)
            γ = get_reducible_vertex(F_mbe, Ch)

            for i in eachindex(γ.K3.data)
                Ω, ν, ω = value.(to_meshes(γ.K3, i))
                K1 = γ.K1(Ω)
                K2 = γ.K2(Ω, ν)
                K2p = γ.K2(Ω, ω)
                ∇ = F_mbe(Ω, ν, ω, Ch, pSp; γp = (Ch === pCh), γt = (Ch === tCh), γa = (Ch === aCh)) - γ.K3(Ω, ν, ω)
                γ.K3.data[i] -= ∇ - (U + K1 + K2 + K2p)
            end
        end

    else
        # Subtract SBE contribution from RefVertex F0
        # Since F0 is the total vertex, not channel reducible, we need to subtract SBE
        # contributions in all channels.
        for (Λ, Ch, Sp) in [(F_mbe.F0.Fp_p, pCh, pSp),
                            (F_mbe.F0.Fp_x, pCh, xSp),
                            (F_mbe.F0.Ft_p, tCh, pSp),
                            (F_mbe.F0.Ft_x, tCh, xSp),]
            for i in eachindex(Λ.data)
                Ω, ν, ω = value.(to_meshes(Λ, i))

                # MBE minus asymptotic form of the channels: the SBE products
                # K2·K2'/(U + K1) of all three channels (D/M form for t)
                ∇ = F_mbe(Ω, ν, ω, Ch, Sp; F0 = false) - F(Ω, ν, ω, Ch, Sp; F0 = false)
                Λ[i] -= ∇
            end
        end
    end

    return F_mbe
end


"""
    mbe_to_asymptotic(F_mbe::MBEVertex) -> Vertex
    mbe_to_asymptotic(F_mbe::NL2_MBEVertex) -> NL2_Vertex

Inverse of `asymptotic_to_mbe`: converts an MBE vertex back to the asymptotic
representation by reabsorbing the SBE contribution into K3.

NOTE: K3 is set to the single-channel MBE value minus `U + K1 + K2 + K2'`, which
also contains the core of `F.F0`. So this inverts the K3 case of
`asymptotic_to_mbe` only for a zero core, and never the core case.
"""
function mbe_to_asymptotic(F_mbe :: MBEVertex)
    F = Vertex(copy(F_mbe.F0), copy(F_mbe.γp), copy(F_mbe.γt), copy(F_mbe.γa))

    # Subtract SBE contribution from K3
    # (K3 = single-channel MBE value - (U + K1 + K2 + K2'), i.e. M_r + K2·K2'/(U + K1))
    U = bare_vertex(F)
    for Ch in (aCh, pCh, tCh)
        γ = get_reducible_vertex(F, Ch)

        for i in eachindex(γ.K3.data)
            Ω, ν, ω = value.(to_meshes(γ.K3, i))
            K1 = γ.K1(Ω)
            K2 = γ.K2(Ω, ν)
            K2p = γ.K2(Ω, ω)
            K123 = F_mbe(Ω, ν, ω, Ch, pSp; γp = (Ch === pCh), γt = (Ch === tCh), γa = (Ch === aCh))
            γ.K3.data[i] = K123 - (U + K1 + K2 + K2p)
        end
    end

    return F
end



# -----------------------------------------------------------------------------
# Lattice MBE vertex (solver NL2)
# -----------------------------------------------------------------------------

"""
    struct NL2_MBEVertex{Q, VT} <: AbstractMBEVertex{Q}

Nonlocal (momentum-resolved) MBE vertex.  Holds a reference vertex `F0` and three
`NL2_Channel{Q}` objects that store K1(Ω, P), K2(Ω, ν, P, k), and K3(Ω, ν, ν', P)
on a Brillouin-zone mesh. The K3 slot holds the s-wave multi-boson rest `M_r`.

Constructors: `NL2_MBEVertex(F0, γp, γt, γa)`, or
`NL2_MBEVertex(F0, T, numK1, numK2, numK3, meshK)` with three zero channels.
"""
struct NL2_MBEVertex{Q, VT} <: AbstractMBEVertex{Q}
    F0 :: VT
    γp :: NL2_Channel{Q}
    γt :: NL2_Channel{Q}
    γa :: NL2_Channel{Q}

    # from existing channels
    function NL2_MBEVertex(
        F0 :: VT,
        γp :: NL2_Channel{Q},
        γt :: NL2_Channel{Q},
        γa :: NL2_Channel{Q},
        ) where {Q, VT}

        return new{Q, VT}(F0, γp, γt, γa)
    end

    # three zero channels with the given boxes and momentum mesh
    function NL2_MBEVertex(
        F0    :: VT,
        T     :: Float64,
        numK1 :: Int64,
        numK2 :: NTuple{2, Int64},
        numK3 :: NTuple{2, Int64},
        meshK :: KMesh,
        ) where {VT}

        Q = eltype(F0)

        γ = NL2_Channel(T, numK1, numK2, numK3, meshK, Q)
        return new{Q, VT}(F0, γ, copy(γ), copy(γ))
    end
end

# An NL2_MBEVertex stores its channels as NL2_Channel objects (used by load_vertex).
channel_type(::Type{NL2_MBEVertex}) = NL2_Channel

"""
    copy(F::NL2_MBEVertex) -> NL2_MBEVertex

Independent copy, including a copy of `F0`.
"""
function Base.:copy(
    F :: NL2_MBEVertex{Q}
    ) :: NL2_MBEVertex{Q} where {Q}

    return NL2_MBEVertex(copy(F.F0), copy(F.γp), copy(F.γt), copy(F.γa))
end

"""
    get_P_mesh(F::NL2_MBEVertex) -> KMesh

Returns the Brillouin-zone momentum mesh used by the nonlocal channels of `F`.
"""
function get_P_mesh(
    F :: NL2_MBEVertex
    ) :: KMesh

    return get_P_mesh(F.γp)
end

"""
    numP(F::NL2_MBEVertex) -> Int64

Returns the number of momentum points in the Brillouin-zone mesh of `F`.
"""
function numP(
    F :: NL2_MBEVertex
    ) :: Int64

    return length(get_P_mesh(F))
end


"""
    asymptotic_to_mbe(F::NL2_Vertex) -> NL2_MBEVertex

Nonlocal counterpart of `asymptotic_to_mbe(::Vertex)`.  Converts a momentum-resolved
vertex from the asymptotic representation to the MBE representation by removing the
SBE contribution from K3 in each channel, leaving the multi-boson rest `M_r`.
Fermionic momenta are s-wave projected (`kSW`) when evaluating K2 and K2'.
Since K3 is s-wave, the subtracted SBE term is the average of
`K2(k)·K2'(k′)/(U + K1)` over k and k′. Only the K3 case exists here; the NOTE
of the local method applies.
"""
function asymptotic_to_mbe(F :: NL2_Vertex)
    F_mbe = NL2_MBEVertex(copy(F.F0), copy(F.γp), copy(F.γt), copy(F.γa))

    # Subtract SBE contribution from K3
    U = bare_vertex(F_mbe)
    for Ch in (aCh, pCh, tCh)
        γ = get_reducible_vertex(F_mbe, Ch)

        for i in eachindex(γ.K3.data)
            Ω, ν, ω, P = value.(to_meshes(γ.K3, i))
            K1 = γ.K1(Ω, P)
            K2 = γ.K2(Ω, ν, P, kSW)
            K2p = γ.K2(Ω, ω, P, kSW)
            ∇ = F_mbe(Ω, ν, ω, P, kSW, kSW, Ch, pSp; γp = (Ch === pCh), γt = (Ch === tCh), γa = (Ch === aCh)) - γ.K3(Ω, ν, ω, P)
            γ.K3.data[i] -= ∇ - (U + K1 + K2 + K2p)
        end
    end

    return F_mbe
end


"""
    mbe_to_asymptotic(F_mbe::NL2_MBEVertex) -> NL2_Vertex

Nonlocal counterpart of `mbe_to_asymptotic(::MBEVertex)`.  Converts a momentum-resolved
MBE vertex back to the asymptotic representation by reabsorbing the SBE contribution
into K3: `K3 ← K123 - (U + K1 + K2 + K2')`, with `K123` the single-channel MBE value
averaged over k and k′ (the NOTE of the local method applies).
"""
function mbe_to_asymptotic(F_mbe :: NL2_MBEVertex)
    F = NL2_Vertex(copy(F_mbe.F0), copy(F_mbe.γp), copy(F_mbe.γt), copy(F_mbe.γa))

    # Subtract SBE contribution from K3
    U = bare_vertex(F)
    for Ch in (aCh, pCh, tCh)
        γ = get_reducible_vertex(F, Ch)

        for i in eachindex(γ.K3.data)
            Ω, ν, ω, P = value.(to_meshes(γ.K3, i))
            K1 = γ.K1(Ω, P)
            K2 = γ.K2(Ω, ν, P, kSW)
            K2p = γ.K2(Ω, ω, P, kSW)
            K123 = F_mbe(Ω, ν, ω, P, kSW, kSW, Ch, pSp; γp = (Ch === pCh), γt = (Ch === tCh), γa = (Ch === aCh))
            γ.K3.data[i] = K123 - (U + K1 + K2 + K2p)
        end
    end

    return F
end



# -----------------------------------------------------------------------------
# Unused: spin-resolved view of an MBE vertex, and its class evaluators
# -----------------------------------------------------------------------------

"""
    struct MBEVertexViewX2X{Q, VT, CT} <: AbstractVertex{Q}

A fixed-momentum view of an MBE vertex (compare `VertexViewX2X`,
`fixed_momentum_view.jl`) that keeps the pSp and xSp spin components of each
channel in separate `ChannelViewX2X` fields, and the a channel split into its
parts from the t and the a channel. Not used anywhere in the package: no
function builds or evaluates it.
Fields:
- `γpp`/`γpx` : p-channel, pSp and xSp components
- `γtp`/`γtx` : t-channel, pSp and xSp components
- `γatp`/`γatx`: a-channel contribution from the t-channel (via crossing), pSp and xSp
- `γaap`/`γaax`: a-channel contribution from the a-channel, pSp and xSp
"""
struct MBEVertexViewX2X{Q, VT, CT} <: AbstractVertex{Q}
    F0  :: VT
    γpp :: CT
    γtp :: CT
    γatp :: CT
    γaap :: CT
    γpx :: CT
    γtx :: CT
    γatx :: CT
    γaax :: CT

    function MBEVertexViewX2X(
        F0  :: VT,
        γpp :: CT,
        γtp :: CT,
        γatp :: CT,
        γaap :: CT,
        γpx :: CT,
        γtx :: CT,
        γatx :: CT,
        γaax :: CT,
        ) where {VT, CT <: ChannelViewX2X{Q}} where {Q}

        return new{Q, VT, CT}(F0, γpp, γtp, γatp, γaap, γpx, γtx, γatx, γaax)
    end
end

# Temperature and box sizes, from the p-channel view.
function MatsubaraFunctions.temperature(F :: MBEVertexViewX2X) :: Float64
    return MatsubaraFunctions.temperature(F.γpp)
end

numK1(F :: MBEVertexViewX2X) :: Int64 = numK1(F.γpp)
numK2(F :: MBEVertexViewX2X) :: NTuple{2, Int64} = numK2(F.γpp)
numK3(F :: MBEVertexViewX2X) :: NTuple{2, Int64} = numK3(F.γpp)


"""
    (F::MBEVertex)(Ω, ν, ν', ::Type{Ch}, ::Type{Ch_k}, ::Type{Sp}, ::Type{Cl})

Evaluates the `MBEVertex` at asymptotic class `Cl`, spin component `Sp`, and channel `Ch`,
with a secondary channel tag `Ch_k` forwarded to the reference vertex `F.F0`:
the class of `F.F0` plus `+γr` for pSp or `-γr` for xSp, with r = Ch. Not used
anywhere in the package.

NOTE: `-γr` is not in general the crossed spin component of channel r. Crossing
maps t ↔ a, and in the p channel ν′ → Ω - ν′ (Lihm 2025, Eq. (S4a)).
"""
@inline function (F :: MBEVertex{Q})(
    Ω  :: Union{MatsubaraFrequency, InfiniteMatsubaraFrequency},
    ν  :: Union{MatsubaraFrequency, InfiniteMatsubaraFrequency},
    νp :: Union{MatsubaraFrequency, InfiniteMatsubaraFrequency},
       :: Type{Ch},
       :: Type{Ch_k},
       :: Type{Sp},
       :: Type{Cl},
    )  :: Q where {Q, Ch <: ChannelTag, Ch_k <: ChannelTag, Sp <: SpinTag, Cl <: ClassTag}

    # Recursively evaluate the Vertex `F` and its RefVertex `F.F0`
    # at the `Sp` spin component, channel `Ch` and asymptotic class `Cl`

    val = F.F0(Ω, ν, νp, Ch, Ch_k, Sp, Cl)

    Ch === pCh && Sp == pSp && (val += F.γp(Ω, ν, νp, Cl))
    Ch === pCh && Sp == xSp && (val -= F.γp(Ω, ν, νp, Cl))

    Ch === tCh && Sp == pSp && (val += F.γt(Ω, ν, νp, Cl))
    Ch === tCh && Sp == xSp && (val -= F.γt(Ω, ν, νp, Cl))

    Ch === aCh && Sp == pSp && (val += F.γa(Ω, ν, νp, Cl))
    Ch === aCh && Sp == xSp && (val -= F.γa(Ω, ν, νp, Cl))

    return val
end



"""
    (F::RefVertex)(Ω, ν, ν', ::Type{Ch}, ::Type{Ch_k}, ::Type{Sp}, ::Type{Cl})

Returns the Λ-class contribution (the core, without U) of the reference vertex
for all (Ch, Sp) channel–spin combinations, using crossing relations to relate the
a-channel and xSp components to the p/t-channel data stored in `F`.
Returns zero for all non-Λ asymptotic classes. Used only by the unused method above.
"""
@inline function (F :: RefVertex{Q})(
    Ω  :: Union{MatsubaraFrequency, InfiniteMatsubaraFrequency},
    ν  :: Union{MatsubaraFrequency, InfiniteMatsubaraFrequency},
    νp :: Union{MatsubaraFrequency, InfiniteMatsubaraFrequency},
       :: Type{Ch},
       :: Type{Ch_k},
       :: Type{Sp},
       :: Type{Cl},
    ; kwargs...
    )  :: Q where {Q, Ch <: ChannelTag, Ch_k <: ChannelTag, Sp <: SpinTag, Cl <: ClassTag}

    if Cl === ΛCl
        if Ch === pCh && Sp === pSp
            return F.Fp_p(Ω, ν, νp)

        elseif Ch === tCh && Sp === pSp
            return F.Ft_p(Ω, ν, νp)

        elseif Ch === aCh && Sp === pSp
            return -F.Ft_x(Ω, νp, ν)

        elseif Ch === pCh && Sp === xSp
            return -F.Fp_p(Ω, ν, Ω - νp)

        elseif Ch === tCh && Sp === xSp
            return F.Ft_x(Ω, ν, νp)

        elseif Ch === aCh && Sp === xSp
            return -F.Ft_p(Ω, νp, ν)

        end
    else
        return zero(Q)
    end
end



# -----------------------------------------------------------------------------
# K3 caches of the NL2 MBE solver
# -----------------------------------------------------------------------------

"""
    build_K3_cache!(S::NL2_ParquetSolver{Q, <:NL2_MBEVertex})

Nonlocal counterpart of `build_K3_cache!(::ParquetSolver)`.  Precomputes the vertex
combinations needed to update K3 for a momentum-resolved MBE parquet solver.
The same three cache classes are filled (see the local overload for details), but all
quantities now carry an explicit bosonic momentum `P` and fermionic momenta are
s-wave projected (`kSW`).

`cache_Γ*` is written out as `F - F.F0`, each with channel r switched off. This
is the same `Ĩ_r` as in the local version, where `F0 = false` does that
subtraction inside the MBE evaluator. All caches are reduced across MPI ranks.
The s-wave evaluation of an MBE vertex sums over k and k′ explicitly, so this
is expensive.
"""
function build_K3_cache!(
    S :: NL2_ParquetSolver{Q, <: NL2_MBEVertex}
    ) where {Q}

    U = bare_vertex(S.F)

    set!(S.cache_Γpx, 0)
    set!(S.cache_F0p, 0)
    set!(S.cache_F0a, 0)
    set!(S.cache_F0t, 0)
    set!(S.cache_Γpp, 0)
    set!(S.cache_Γa,  0)
    set!(S.cache_Γt,  0)
    set!(S.cache_Fp,  0)
    set!(S.cache_Fa,  0)
    set!(S.cache_Ft,  0)

    # Vertices multiplied by bubbles to the left (by ω)
    # Γpx : Target, irreducible vertex in the p channel, xSp component.

    Threads.@threads for i in mpi_split(1 : length(S.cache_Γpx.data))
        Ω, ω, νp, P = value.(MatsubaraFunctions.to_meshes(S.cache_Γpx, i))

        S.cache_Γpx[i] = S.F( Ω, ω, νp, P, kSW, kSW, pCh, xSp; γp=false) - S.F.F0( Ω, ω, νp, P, kSW, kSW, pCh, xSp; γp=false)

        # (Iᵣ - U) + Mᵣ
        # (p channel in xSp: the bare part is -U, and M_p^×(Ω, ν̃, ν′) = -M_p^=(Ω, ν̃, Ω - ν′))
        S.cache_F0p[i] = S.F0(Ω, ω, νp, P, kSW, kSW, pCh, xSp; γp = false) + U + S.F0(Ω, ω, Ω - νp, P, kSW, kSW, pCh, K3Cl) * -1
        S.cache_F0a[i] = S.F0(Ω, ω, νp, P, kSW, kSW, aCh, pSp; γa = false) - U + S.F0(Ω, ω, νp, P, kSW, kSW, aCh, K3Cl)
        S.cache_F0t[i] = S.F0(Ω, ω, νp, P, kSW, kSW, tCh, pSp; γt = false) - U + S.F0(Ω, ω, νp, P, kSW, kSW, tCh, K3Cl)

        # Convert from pSp (parallel spin component) to dSp (density component)
        # using the relation dSp = 2 * pSp + xSp = 2 * pSp - pSp(a) (crossing symmetry)
        S.cache_F0t[i] = 2 * S.cache_F0t[i] - S.cache_F0a[i]
    end

    mpi_allreduce!(S.cache_Γpx)
    mpi_allreduce!(S.cache_F0p)
    mpi_allreduce!(S.cache_F0a)
    mpi_allreduce!(S.cache_F0t)


    # Vertices multiplied by bubbles from the right (by ω)

    Threads.@threads for i in mpi_split(1 : length(S.cache_Γpp.data))
        Ω, ν, ω, P = value.(MatsubaraFunctions.to_meshes(S.cache_Γpp, i))

        # r-irreducible vertex in each channel r = p, a, t
        S.cache_Γpp[i] = S.F(Ω, ν, ω, P, kSW, kSW, pCh, pSp; γp=false) - S.F.F0(Ω, ν, ω, P, kSW, kSW, pCh, pSp; γp=false)
        S.cache_Γa[i]  = S.F(Ω, ν, ω, P, kSW, kSW, aCh, pSp; γa=false) - S.F.F0(Ω, ν, ω, P, kSW, kSW, aCh, pSp; γa=false)
        S.cache_Γt[i]  = S.F(Ω, ν, ω, P, kSW, kSW, tCh, pSp; γt=false) - S.F.F0(Ω, ν, ω, P, kSW, kSW, tCh, pSp; γt=false)

        # (Iᵣ - U) + Mᵣ
        S.cache_Fp[i]  = S.F(Ω, ν, ω, P, kSW, kSW, pCh, pSp; γp = false) - U + S.F(Ω, ν, ω, P, kSW, kSW, pCh, K3Cl)
        S.cache_Fa[i]  = S.F(Ω, ν, ω, P, kSW, kSW, aCh, pSp; γa = false) - U + S.F(Ω, ν, ω, P, kSW, kSW, aCh, K3Cl)
        S.cache_Ft[i]  = S.F(Ω, ν, ω, P, kSW, kSW, tCh, pSp; γt = false) - U + S.F(Ω, ν, ω, P, kSW, kSW, tCh, K3Cl)

        # Convert from pSp (parallel spin component) to dSp (density component)
        # using the relation dSp = 2 * pSp + xSp = 2 * pSp - pSp(a) (crossing symmetry)
        S.cache_Γt[i] = S.cache_Γt[i] * 2 - S.cache_Γa[i]
        S.cache_Ft[i] = S.cache_Ft[i] * 2 - S.cache_Fa[i]
    end

    mpi_allreduce!(S.cache_Γpp)
    mpi_allreduce!(S.cache_Γa)
    mpi_allreduce!(S.cache_Γt)
    mpi_allreduce!(S.cache_Fp)
    mpi_allreduce!(S.cache_Fa)
    mpi_allreduce!(S.cache_Ft)

    return nothing
end
