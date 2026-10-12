# =============================================================================
# The lattice vertex of the solver NL2, and its evaluation with averaged
# fermionic momenta.
#
# As for NL_Vertex (nonlocal/vertex.jl), F = F0 + Φ_p + Φ_t + Φ_a (Lihm 2025,
# Eq. (1)), but the channels are NL2_Channel: K2(Ω, ν; q, k) keeps the
# fermionic momentum k of its leg, while K1 and K3 depend on the bosonic
# momentum q only (nonlocal_2/channel.jl). Evaluation with all momenta given,
# at νInf, and for xSp and dSp uses the shared methods of nonlocal/vertex.jl.
#
# This file adds the s-wave evaluation used by the BSE kernels: F^= at fixed
# bosonic momentum q of channel Ch, averaged over k and/or k′ (marker kSW).
#   - Channel Ch itself: its K2 (or K̄2) is averaged over the averaged leg.
#   - The other two channels: the averaged momentum enters their bosonic
#     momentum, so their K1 and K3 are averaged over it (kSW). Their K2 + K̄2
#     depend on two momenta. As k runs over the Brillouin zone at fixed q and
#     k′, both change together, so this is not a plain average over either
#     one. It is done as an explicit sum, 1/N_k ∑_k (K2 + K̄2 at the momenta
#     converted from (q, k, k′)).
#   - k and k′ both averaged: the other two channels are averaged over all
#     their momenta (kSW).
#
# In fd runs F0 (the reference f) is itself an NL2_Vertex that wraps the local
# DMFT vertex with zero channels (fdDGAsolver.jl, script/common/runs.jl).
#
# Paper ↔ code: q ↔ P; Φ_r ↔ the fields `γp`, `γt`, `γa`. The evaluators here
# are written (Ω, ν, ω, P, k, q), with ω = ν′ and q = k′ (not the paper's q).
# =============================================================================

"""
    NL2_Vertex{Q, VT} <: AbstractNonlocalVertex{Q}

Full lattice vertex of the solver NL2, `F = F0 + Φ_p + Φ_t + Φ_a`, with the
channels stored as [`NL2_Channel`](@ref) (K2 keeps the fermionic momentum).

- `F0` (type `VT`): the vertex the channels are added to: a `RefVertex` (bare U)
  in the PA, or the reference f in fd runs.
- `γp`, `γt`, `γa`: the channels `Φ_p`, `Φ_t`, `Φ_a` (`Φ̃_r` in fd runs).

Constructors: `NL2_Vertex(F0, γp, γt, γa)`, or
`NL2_Vertex(F0, T, numK1, numK2, numK3, meshK)` with three zero channels on the
momentum mesh `meshK`.
"""
struct NL2_Vertex{Q, VT} <: AbstractNonlocalVertex{Q}
    F0 :: VT
    γp :: NL2_Channel{Q}
    γt :: NL2_Channel{Q}
    γa :: NL2_Channel{Q}

    # from existing channels
    function NL2_Vertex(
        F0 :: VT,
        γp :: NL2_Channel{Q},
        γt :: NL2_Channel{Q},
        γa :: NL2_Channel{Q},
        )  :: NL2_Vertex{Q} where {Q, VT}

        return new{Q, VT}(F0, γp, γt, γa)
    end

    # three zero channels with the given boxes and momentum mesh
    function NL2_Vertex(
        F0    :: VT,
        T     :: Float64,
        numK1 :: Int64,
        numK2 :: NTuple{2, Int64},
        numK3 :: NTuple{2, Int64},
        meshK :: KMesh,
        ) where {VT}

        Q = eltype(F0)

        γ = NL2_Channel(T, numK1, numK2, numK3, meshK, Q)
        return new{Q, VT}(F0, γ, copy(γ), copy(γ)) :: NL2_Vertex{Q}
    end
end

# An NL2_Vertex stores its channels as NL2_Channel objects (used by load_vertex).
channel_type(::Type{NL2_Vertex}) = NL2_Channel

# Printing: type, U, temperature, F0 and the box sizes.
function Base.show(io::IO, Γ::NL2_Vertex{Q}) where {Q}
    print(io, "$(nameof(typeof(Γ))){$Q}, U = $(bare_vertex(Γ.F0)), T = $(temperature(Γ))\n")
    print(io, "F0 : $(Γ.F0)\n")
    print(io, "K1 : $(numK1(Γ))\n")
    print(io, "K2 : $(numK2(Γ))\n")
    print(io, "K3 : $(numK3(Γ))")
end

"""
    copy(F::NL2_Vertex) -> NL2_Vertex

Independent copy, including a copy of `F0`.
"""
function Base.:copy(
    F :: NL2_Vertex{Q}
    ) :: NL2_Vertex{Q} where {Q}

    return NL2_Vertex(copy(F.F0), copy(F.γp), copy(F.γt), copy(F.γa))
end

# -----------------------------------------------------------------------------
# s-wave evaluation (averaged fermionic momenta)
# -----------------------------------------------------------------------------

"""
    (F::NL2_Vertex)(Ω, ν, ω, P, k, q, ::Type{Ch}, ::Type{pSp}; F0 = true, γp = true, γt = true, γa = true) -> value

F^= in channel `Ch`'s parametrization at the bosonic momentum `P`, averaged
over the fermionic momentum `k` and/or `q` (= k′) when it is
[`kSW`](@ref) (s-wave approximation, see the file header). `ω` is the second
fermionic frequency ν′. Three methods: `k` averaged, `q` averaged, or both. The
keywords work as for [`AbstractNonlocalVertex`](@ref); `F0` is evaluated in
full at the same arguments.
"""
@inline function (F :: NL2_Vertex{Q})(
    Ω  :: MatsubaraFrequency,
    ν  :: MatsubaraFrequency,
    ω  :: MatsubaraFrequency,
    P  :: BrillouinPoint,
    k  :: SWaveBrillouinPoint,
    q  :: BrillouinPoint,
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
        val += F.F0(Ω, ν, ω, P, k, q, Ch, pSp)
    end

    # k is averaged (kSW), k′ = q is given

    if γp
        if Ch === pCh
            # Same channel: evaluate directly. Its K2 is averaged over k,
            # its K̄2 is taken at k′.
            val += F.γp(Ω, ν, ω, P, k, q)
        else
            # Other channel: its bosonic momentum contains k, so its K1 and K3
            # are averaged over it.
            ωs = convert_frequency(Ω, ν, ω, Ch, pCh)
            val += F.γp(ωs..., kSW, kSW, kSW; K1 = true, K2 = false, K3 = true)

            # Its K2 + K̄2 depend on two momenta that both change with k:
            # average over k by an explicit sum.
            for k_int in get_P_mesh(F)
                ks = convert_momentum(P, value(k_int), q, Ch, pCh)
                val += F.γp(ωs..., ks...; K1 = false, K2 = true, K3 = false) / numP(F)
            end
        end
    end

    if γt
        if Ch === tCh
            val += F.γt(Ω, ν, ω, P, k, q)
        else
            ωs = convert_frequency(Ω, ν, ω, Ch, tCh)
            val += F.γt(ωs..., kSW, kSW, kSW; K1 = true, K2 = false, K3 = true)

            for k_int in get_P_mesh(F)
                ks = convert_momentum(P, value(k_int), q, Ch, tCh)
                val += F.γt(ωs..., ks...; K1 = false, K2 = true, K3 = false) / numP(F)
            end
        end
    end

    if γa
        if Ch === aCh
            val += F.γa(Ω, ν, ω, P, k, q)
        else
            ωs = convert_frequency(Ω, ν, ω, Ch, aCh)
            val += F.γa(ωs..., kSW, kSW, kSW; K1 = true, K2 = false, K3 = true)

            for k_int in get_P_mesh(F)
                ks = convert_momentum(P, value(k_int), q, Ch, aCh)
                val += F.γa(ωs..., ks...; K1 = false, K2 = true, K3 = false) / numP(F)
            end
        end
    end

    return val
end

# k′ (= q) averaged, k given: the same as above with the roles of k and k′
# exchanged, see the docstring above.
@inline function (F :: NL2_Vertex{Q})(
    Ω  :: MatsubaraFrequency,
    ν  :: MatsubaraFrequency,
    ω  :: MatsubaraFrequency,
    P  :: BrillouinPoint,
    k  :: BrillouinPoint,
    q  :: SWaveBrillouinPoint,
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
        val += F.F0(Ω, ν, ω, P, k, q, Ch, pSp)
    end

    # k′ = q is averaged (kSW), k is given

    if γp
        if Ch === pCh
            # Same channel: evaluate directly. Its K2 is taken at k, its K̄2
            # is averaged over k′.
            val += F.γp(Ω, ν, ω, P, k, q)
        else
            # Other channel: K1 and K3 averaged over its bosonic momentum
            ωs = convert_frequency(Ω, ν, ω, Ch, pCh)
            val += F.γp(ωs..., kSW, kSW, kSW; K1 = true, K2 = false, K3 = true)

            # K2 + K̄2: average over k′ by an explicit sum
            for q_int in get_P_mesh(F)
                ks = convert_momentum(P, k, value(q_int), Ch, pCh)
                val += F.γp(ωs..., ks...; K1 = false, K2 = true, K3 = false) / numP(F)
            end
        end
    end

    if γt
        if Ch === tCh
            val += F.γt(Ω, ν, ω, P, k, q)
        else
            ωs = convert_frequency(Ω, ν, ω, Ch, tCh)
            val += F.γt(ωs..., kSW, kSW, kSW; K1 = true, K2 = false, K3 = true)

            for q_int in get_P_mesh(F)
                ks = convert_momentum(P, k, value(q_int), Ch, tCh)
                val += F.γt(ωs..., ks...; K1 = false, K2 = true, K3 = false) / numP(F)
            end
        end
    end

    if γa
        if Ch === aCh
            val += F.γa(Ω, ν, ω, P, k, q)
        else
            ωs = convert_frequency(Ω, ν, ω, Ch, aCh)
            val += F.γa(ωs..., kSW, kSW, kSW; K1 = true, K2 = false, K3 = true)

            for q_int in get_P_mesh(F)
                ks = convert_momentum(P, k, value(q_int), Ch, aCh)
                val += F.γa(ωs..., ks...; K1 = false, K2 = true, K3 = false) / numP(F)
            end
        end
    end

    return val
end

# k and k′ (= q) both averaged, see the docstring above.
@inline function (F :: NL2_Vertex{Q})(
    Ω  :: MatsubaraFrequency,
    ν  :: MatsubaraFrequency,
    ω  :: MatsubaraFrequency,
    P  :: BrillouinPoint,
    k  :: SWaveBrillouinPoint,
    q  :: SWaveBrillouinPoint,
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
        val += F.F0(Ω, ν, ω, P, k, q, Ch, pSp)
    end

    # k and k′ = q both averaged. The momenta of another channel (bosonic for
    # K1 and K3; bosonic and fermionic for K2) are independent combinations of
    # k and k′, so that channel is averaged over all of them.

    if γp
        if Ch === pCh
            # Same channel: K2 and K̄2 averaged over their fermionic momentum
            val += F.γp(Ω, ν, ω, P, k, q)
        else
            # Other channel: averaged over all its momenta
            val += F.γp(convert_frequency(Ω, ν, ω, Ch, pCh)..., kSW, kSW, kSW)
        end
    end

    if γt
        if Ch === tCh
            val += F.γt(Ω, ν, ω, P, k, q)
        else
            val += F.γt(convert_frequency(Ω, ν, ω, Ch, tCh)..., kSW, kSW, kSW)
        end
    end

    if γa
        if Ch === aCh
            val += F.γa(Ω, ν, ω, P, k, q)
        else
            val += F.γa(convert_frequency(Ω, ν, ω, Ch, aCh)..., kSW, kSW, kSW)
        end
    end

    return val
end
