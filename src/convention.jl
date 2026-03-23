# convention.jl
#
#   - Channel-to-channel transformation rules for frequencies and momenta.
#   - Interface for mapping between p-channel (parallel), t-channel (transversal), 
#     and a-channel (antiparallel) representations.

# conversion between different frequency/momentum representations
#----------------------------------------------------------------------------------------------#
# Core logic for transforming triplet indices (A, b, bp). 
# A represents the transfer frequency/momentum (ω or q).
# b and bp represent the fermionic leg frequencies/momenta (ν, ν' or k, k').

# frequency convention
# -ν_2 ↘      ↗ ν_1
#        ┌---┐
#        | F |
#        └---┘
# -ν_4 ↗      ↘ ν_3
#
# parallel channel       tranvserse channel       antiparallel channel         
#  ω-ν ↘      ↗ ν'       ω+ν ↘      ↗ ν              ν ↘      ↗ ν'       
#        ┌---┐                 ┌---┐                      ┌---┐           
#        | F |                 | F |                      | F |          
#        └---┘                 └---┘                      └---┘           
#    ν ↗      ↘ ω-ν'     ν' ↗       ↘ ω+ν'        ω+ν' ↗      ↘ ω+ν    

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

"""
    function convert_frequency(
        Ω  :: MatsubaraFrequency{Boson},
        ν  :: Union{InfiniteMatsubaraFrequency, MatsubaraFrequency{Fermion}},
        νp :: Union{InfiniteMatsubaraFrequency, MatsubaraFrequency{Fermion}},
           :: Type{Ch_from},
           :: Type{Ch_to}
        ) where {Ch_from <: ChannelTag, Ch_to <: ChannelTag}

Convert frequencies in the `Ch_from` channel representation to `Ch_to` channel representation
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
    function convert_momentum(
        Q  :: BrillouinPoint,
        k  :: BrillouinPoint,
        kp :: BrillouinPoint,
           :: Type{Ch_from},
           :: Type{Ch_to}
        ) where {Ch_from <: ChannelTag, Ch_to <: ChannelTag}

Convert momenta in the `Ch_from` channel representation to `Ch_to` channel representation
"""
@inline function convert_momentum(
    Q  :: BrillouinPoint,
    k  :: BrillouinPoint,
    kp :: BrillouinPoint,
       :: Type{Ch_from},
       :: Type{Ch_to}
    ) where {Ch_from <: ChannelTag, Ch_to <: ChannelTag}

    _convert_channel(Q, k, kp, Ch_from, Ch_to)
end
