# =============================================================================
# Frequency and momentum conventions of the three channels, and how to convert
# between them.
#
# A two-particle vertex has four legs, but only three independent frequencies,
# because the sum of all four frequencies vanishes. Each channel r = p, t, a
# writes them as one bosonic frequency Ω and two fermionic frequencies ν, ν′
# (Lihm 2025, Eq. (S5) and Fig. S1). Ω is the frequency carried by the pair of
# lines that is cut in that channel. With this choice the BSE of channel r is
# diagonal in Ω (Eqs. (S9)).
#
# The reducible vertex Φ_r is always stored in its own channel's
# parametrization. Evaluating the full vertex F in channel r therefore means
# converting the arguments to the other two channels (Eq. (S6)). That is what
# this file does. Momenta follow the same rules: one bosonic momentum and two
# fermionic momenta k, k′.
#
# Paper ↔ code: ω ↔ Ω (the drawings below keep the paper's ω), ν′ ↔ νp,
# q ↔ P (called Q in convert_momentum), k′ ↔ kp.
# =============================================================================

# -----------------------------------------------------------------------------
# Frequency conventions
# -----------------------------------------------------------------------------
# Leg labels: F(ν_1, ν_2, ν_3, ν_4). Arrows into F are incoming legs (frequency
# -ν_2 and -ν_4), arrows out of F are outgoing legs (ν_1 and ν_3), and
# ν_1 + ν_2 + ν_3 + ν_4 = 0.
#
#  -ν_2 ↘      ↗ ν_1
#         ┌---┐
#         | F |
#         └---┘
#  -ν_4 ↗      ↘ ν_3
#
# The three channel parametrizations (Lihm 2025, Eq. (S5)):
#
#   parallel channel       transverse channel       antiparallel channel
#  ω-ν ↘      ↗ ν'       ω+ν ↘      ↗ ν              ν ↘      ↗ ν'
#        ┌---┐                 ┌---┐                      ┌---┐
#        | F |                 | F |                      | F |
#        └---┘                 └---┘                      └---┘
#    ν ↗      ↘ ω-ν'     ν' ↗       ↘ ω+ν'        ω+ν' ↗      ↘ ω+ν
#
#   F(ω, ν, ν′)_p = F(ν′, ν - ω, ω - ν′, -ν)
#   F(ω, ν, ν′)_t = F(ν,  -ν - ω, ν′ + ω, -ν′)
#   F(ω, ν, ν′)_a = F(ν′, -ν, ν + ω, -ν′ - ω)
#
# In the p channel ω is the total frequency of the two incoming particles; in
# the a and t channels it is the frequency transferred between a particle and a
# hole.

"""
    _convert_channel(A, b, bp, Ch_from, Ch_to) -> (A, b, bp)

Re-express the three arguments of a vertex, given in the parametrization of
channel `Ch_from`, in the parametrization of channel `Ch_to` (Lihm 2025,
Eq. (S5)).

`A` is the bosonic argument and `b`, `bp` are the two fermionic ones. They can be
frequencies (Ω, ν, ν′) or momenta (P, k, k′): conservation of frequency and of
momentum has the same form, so the same rules apply.
"""
@inline function _convert_channel(
    A,
    b,
    bp,
    :: Type{Ch_from},
    :: Type{Ch_to}
    ) where {Ch_from <: ChannelTag, Ch_to <: ChannelTag}

    # no change of channel
    if Ch_from === Ch_to
        return A, b, bp

    # from parallel to transverse channel
    # ω_t  = -ν_2 - ν_1 = ω_p - ν_p - ν'_p
    # ν_t  = ν_1        = ν'_p
    # ν'_t = -ν_4       = ν_p
    elseif Ch_from === pCh && Ch_to === tCh
        return A - b - bp, bp, b

    # from parallel to antiparallel channel
    # ω_a  = -ν_4 - ν_1 = ν_p - ν'_p
    # ν_a  = -ν_2       = ω_p - ν_p
    # ν'_a = ν_1        = ν'_p
    elseif Ch_from === pCh && Ch_to === aCh
        return b - bp, A - b, bp

    # from transverse to parallel channel
    # ω_p  = -ν_2 - ν_4 = ω_t + ν_t + ν'_t
    # ν_p  = -ν_4       = ν'_t
    # ν'_p = ν_1        = ν_t
    elseif Ch_from === tCh && Ch_to === pCh
        return A + b + bp, bp, b

    # from transverse to antiparallel channel
    # ω_a  = -ν_4 - ν_1 = ν'_t - ν_t
    # ν_a  = -ν_2       = ω_t + ν_t
    # ν'_a = ν_1        = ν_t
    elseif Ch_from === tCh && Ch_to === aCh
        return bp - b, A + b, b

    # from antiparallel to parallel channel
    # ω_p  = -ν_2 - ν_4 = ω_a + ν_a + ν'_a
    # ν_p  = -ν_4       = ω_a + ν'_a
    # ν'_p = ν_1        = ν'_a
    elseif Ch_from === aCh && Ch_to === pCh
        return A + bp + b, A + bp, bp

    # from antiparallel to transverse channel
    # ω_t  = -ν_2 - ν_1 = ν_a - ν'_a
    # ν_t  = ν_1        = ν'_a
    # ν'_t = -ν_4       = ω_a + ν'_a
    elseif Ch_from === aCh && Ch_to === tCh
        return b - bp, bp, A + bp

    else
        throw(ArgumentError("Cannot convert from $Ch_from to $Ch_to !"))
    end

end

# -----------------------------------------------------------------------------
# Public conversion functions
# -----------------------------------------------------------------------------

"""
    convert_frequency(Ω, ν, νp, Ch_from, Ch_to) -> (Ω, ν, νp)

Re-express the frequencies of a vertex, given in channel `Ch_from`'s
parametrization (bosonic `Ω`, fermionic `ν`, `νp`), in channel `Ch_to`'s
parametrization (Lihm 2025, Eq. (S5)).

This is how a reducible vertex of one channel is evaluated inside the full
vertex written in another channel, as in Eq. (S6). If `ν` or `νp` is
[`νInf`](@ref), the converted bosonic frequency of another channel is infinite
as well, so that channel drops out. This is correct: in the high-frequency
limits of the channel-r vertex only channel r's own K1 and K2 survive
(Eq. (S22)).
"""
@inline function convert_frequency(
    Ω  :: MatsubaraFrequency{Boson},
    ν  :: Union{InfiniteMatsubaraFrequency, MatsubaraFrequency{Fermion}},
    νp :: Union{InfiniteMatsubaraFrequency, MatsubaraFrequency{Fermion}},
       :: Type{Ch_from},
       :: Type{Ch_to}
    ) where {Ch_from <: ChannelTag, Ch_to <: ChannelTag}

    _convert_channel(Ω, ν, νp, Ch_from, Ch_to)
end

"""
    convert_momentum(P, k, kp, Ch_from, Ch_to) -> (P, k, kp)

Re-express the momenta of a vertex from channel `Ch_from`'s parametrization to
channel `Ch_to`'s. `P` is the bosonic momentum (paper: q) and `k`, `kp` are the fermionic ones. The
rules are the same as for frequencies, see [`convert_frequency`](@ref).
"""
@inline function convert_momentum(
    P  :: BrillouinPoint,
    k  :: BrillouinPoint,
    kp :: BrillouinPoint,
       :: Type{Ch_from},
       :: Type{Ch_to}
    ) where {Ch_from <: ChannelTag, Ch_to <: ChannelTag}

    _convert_channel(P, k, kp, Ch_from, Ch_to)
end
