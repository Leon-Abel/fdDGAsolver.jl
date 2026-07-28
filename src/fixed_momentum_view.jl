# fixed_momentum_view.jl
#
# Implements momentum-fixed views of the reducible channels and full vertex.
# A "fixed-momentum view" slices the momentum indices out of a non-local channel or vertex,
# producing a lightweight callable that only requires Matsubara frequency arguments.
# This avoids repeated momentum look-ups during the frequency summations in the BSE/parquet loops.
#
# CHANNEL EVALUATOR CONVENTION:
# Both NL_Channel and NL2_Channel store K1, K2, K2p, K3 on grids that include momentum axes.
# After slicing the momentum axes, the resulting ChannelViewX2X evaluates as
#   γ(Ω, ν, ν') = K1[Ω] + K2[Ω, ν] + K2p[Ω, ν'] + K3[Ω, ν, ν']
# where each term is included only if the corresponding frequency is in-bounds on its mesh.
# Out-of-bounds frequencies are treated as the ν → ∞ limit (contribution = 0).
#
# CROSSED CHANNEL CONVENTION:
# VertexViewX2X stores six channel views: three "direct" (γpp, γtp, γap) and three "crossed"
# (γpx, γtx, γax). The crossed views are pre-evaluated at the exchanged momentum so that the
# xSp evaluator can apply crossing symmetry without an additional momentum conversion at call time.


"""
    ChannelViewX2X{Q, T1, T2, T3} <: AbstractReducibleVertex{Q}

A momentum-fixed view of a single reducible channel (NL_Channel or NL2_Channel).

Stores array views of K1, K2, K2p, and K3 with all momentum indices already sliced out,
so only the Matsubara frequency axes remain. `Q` is the element type (e.g. ComplexF64).
"""
struct ChannelViewX2X{Q, T1, T2, T3} <: AbstractReducibleVertex{Q}
    K1  :: T1   # view of K1 at fixed P:         K1[Ω]
    K2  :: T2   # view of K2 at fixed (P, k):    K2[Ω, ν]
    K2p :: T2   # view of K2 at fixed (P, q):    K2p[Ω, ν']
    K3  :: T3   # view of K3 at fixed P:         K3[Ω, ν, ν']

    function ChannelViewX2X(
        K1  :: T1,
        K2  :: T2,
        K2p :: T2,
        K3  :: T3
        ) where {T1 <: MeshFunction{1, Q}, T2 <: MeshFunction{2, Q}, T3 <: MeshFunction{3, Q}} where {Q}

        return new{Q, T1, T2, T3}(K1, K2, K2p, K3)
    end
end


"""
    VertexViewX2X{Q, VT, CT} <: AbstractVertex{Q}

A momentum-fixed view of the full two-particle vertex F(Ω, ν, ν'; P, k, q, Ch, Sp).

Stores a view of the fully irreducible part `F0` and six momentum-fixed channel views:
- `γpp`, `γtp`, `γap`: parallel spin channel contributions in the p-, t-, and a-channels
- `γpx`, `γtx`, `γax`: crossed spin channel contributions in the p-, t-, and a-channels

`Q` is the element type (e.g. ComplexF64).
"""
struct VertexViewX2X{Q, VT, CT} <: AbstractVertex{Q}
    F0  :: VT   # fully irreducible vertex (local or non-local view)
    γpp :: CT   # p-channel reducible part, parallel spin
    γtp :: CT   # t-channel reducible part, parallel spin
    γap :: CT   # a-channel reducible part, parallel spin
    γpx :: CT   # p-channel reducible part, crossed spin
    γtx :: CT   # t-channel reducible part, crossed spin
    γax :: CT   # a-channel reducible part, crossed spin

    function VertexViewX2X(
        F0  :: VT,
        γpp :: CT,
        γtp :: CT,
        γap :: CT,
        γpx :: CT,
        γtx :: CT,
        γax :: CT,
        ) where {VT, CT <: ChannelViewX2X{Q}} where {Q}

        return new{Q, VT, CT}(F0, γpp, γtp, γap, γpx, γtx, γax)
    end
end

function MatsubaraFunctions.temperature(F :: VertexViewX2X) :: Float64
    return MatsubaraFunctions.temperature(F.γpp)
end

numK1(F :: VertexViewX2X) :: Int64            = numK1(F.γpp)
numK2(F :: VertexViewX2X) :: NTuple{2, Int64} = numK2(F.γpp)
numK3(F :: VertexViewX2X) :: NTuple{2, Int64} = numK3(F.γpp)


"""
    fixed_momentum_view(γ::NL_Channel, P, k, q) -> ChannelViewX2X

Returns a `ChannelViewX2X` of `γ` with the bosonic transfer momentum `P` fixed.

`NL_Channel` stores K2 and K2p as the same function of `(Ω, ν, P)`, so both views
are identical slices at `iP`. The fermionic momenta `k` and `q` are unused.
"""
@inline function fixed_momentum_view(
    γ :: NL_Channel,
    P :: BrillouinPoint,
    k :: BrillouinPoint,
    q :: BrillouinPoint,
    )
    iP = MatsubaraFunctions.mesh_index_bc(P, get_P_mesh(γ))

    # slice out the P axis from each asymptotic class
    K1  = MeshFunction(γ.K1.meshes[1:1], view(γ.K1, :, iP))
    K2  = MeshFunction(γ.K2.meshes[1:2], view(γ.K2, :, :, iP))
    K2p = MeshFunction(γ.K2.meshes[1:2], view(γ.K2, :, :, iP))   # same as K2: NL_Channel has no k-dependence
    K3  = MeshFunction(γ.K3.meshes[1:3], view(γ.K3, :, :, :, iP))
    ChannelViewX2X(K1, K2, K2p, K3)
end


"""
    fixed_momentum_view(γ::NL2_Channel, P, k, q) -> ChannelViewX2X

Returns a `ChannelViewX2X` of `γ` with all three momentum arguments `(P, k, q)` fixed.

`NL2_Channel` stores K2 as a function of `(Ω, ν, P, k)` and K2p as a function of
`(Ω, ν, P, q)`, so K2 is sliced at `(iP, ik)` and K2p at `(iP, iq)` independently.
K3 depends only on `P` (not on the fermionic momenta).
"""
@inline function fixed_momentum_view(
    γ :: NL2_Channel,
    P :: BrillouinPoint,
    k :: BrillouinPoint,
    q :: BrillouinPoint,
    )
    iP = MatsubaraFunctions.mesh_index_bc(P, get_P_mesh(γ))
    ik = MatsubaraFunctions.mesh_index_bc(k, get_P_mesh(γ))
    iq = MatsubaraFunctions.mesh_index_bc(q, get_P_mesh(γ))

    K1  = MeshFunction(γ.K1.meshes[1:1], view(γ.K1, :, iP))
    K2  = MeshFunction(γ.K2.meshes[1:2], view(γ.K2, :, :, iP, ik))    # direct fermionic momentum k
    K2p = MeshFunction(γ.K2.meshes[1:2], view(γ.K2, :, :, iP, iq))    # primed fermionic momentum q
    K3  = MeshFunction(γ.K3.meshes[1:3], view(γ.K3, :, :, :, iP))
    ChannelViewX2X(K1, K2, K2p, K3)
end

"""
    fixed_momentum_view(γ::Union{Vertex,RefVertex}, P, k, kp, ::Type{Ch}) -> γ

Returns `γ` unchanged. Local vertices carry no momentum dependence,
so there is nothing to fix; they are their own fixed-momentum view.
"""
@inline function fixed_momentum_view(
    γ :: Union{Vertex, RefVertex},
    P  :: BrillouinPoint,
    k  :: BrillouinPoint,
    kp :: BrillouinPoint,
       :: Type{Ch}
    ) where {Ch <: ChannelTag}
    γ
end


"""
    fixed_momentum_view(F::Union{NL_Vertex,NL2_Vertex}, P, k, kp, ::Type{Ch}) -> VertexViewX2X

Returns a `VertexViewX2X` of the full non-local vertex `F` with momenta `(P, k, kp)` fixed
in channel `Ch`.

The direct channel momenta are obtained by converting `(P, k, kp)` from channel `Ch` into
each of the three channels (p, t, a) via `convert_momentum`. The crossed channel momenta
are computed from the exchanged external leg using the crossing relations:
- pCh: crossing replaces `kp → P - kp`
- tCh: crossing maps the momentum conventions from tCh to aCh (and vice versa)
- aCh: crossing maps the momentum conventions from aCh to tCh (and vice versa)
"""
@inline function fixed_momentum_view(
    F  :: Union{NL_Vertex, NL2_Vertex},
    P  :: BrillouinPoint,
    k  :: BrillouinPoint,
    kp :: BrillouinPoint,
       :: Type{Ch}
    ) where {Ch <: ChannelTag}

    F0 = fixed_momentum_view(F.F0, P, k, kp, Ch)

    # convert (P, k, kp) from channel Ch into each channel's own momentum convention
    ks_p = convert_momentum(P, k, kp, Ch, pCh)
    ks_t = convert_momentum(P, k, kp, Ch, tCh)
    ks_a = convert_momentum(P, k, kp, Ch, aCh)

    γpp = fixed_momentum_view(F.γp, ks_p...)
    γtp = fixed_momentum_view(F.γt, ks_t...)
    γap = fixed_momentum_view(F.γa, ks_a...)

    # Crossed channel views: pre-compute momenta at the exchanged external leg so the xSp
    # evaluator can apply crossing without an additional momentum conversion at runtime.
    if Ch === pCh
        # pCh crossing: exchange the two outgoing legs, sending kp → P - kp
        γpx = fixed_momentum_view(F.γp, convert_momentum(P, k, P - kp, pCh, pCh)...)
        γtx = fixed_momentum_view(F.γt, convert_momentum(P, k, P - kp, pCh, tCh)...)
        γax = fixed_momentum_view(F.γa, convert_momentum(P, k, P - kp, pCh, aCh)...)
    elseif Ch === tCh
        # tCh crossing: related to aCh by exchange of external legs
        γpx = fixed_momentum_view(F.γp, convert_momentum(P, k, kp, aCh, pCh)...)
        γtx = fixed_momentum_view(F.γt, convert_momentum(P, k, kp, aCh, tCh)...)
        γax = fixed_momentum_view(F.γa, convert_momentum(P, k, kp, aCh, aCh)...)
    elseif Ch === aCh
        # aCh crossing: related to tCh by exchange of external legs
        γpx = fixed_momentum_view(F.γp, convert_momentum(P, k, kp, tCh, pCh)...)
        γtx = fixed_momentum_view(F.γt, convert_momentum(P, k, kp, tCh, tCh)...)
        γax = fixed_momentum_view(F.γa, convert_momentum(P, k, kp, tCh, aCh)...)
    end

    VertexViewX2X(F0, γpp, γtp, γap, γpx, γtx, γax)
end


"""
    (γ::ChannelViewX2X{Q})(Ω, ν, ν'; K1, K2, K3) -> Q

Evaluates the asymptotic decomposition of the channel at fixed momentum:
    γ(Ω, ν, ν') = K1[Ω] + K2[Ω, ν] + K2p[Ω, ν'] + K3[Ω, ν, ν']

Each term is included only if its frequency arguments are within the stored mesh;
out-of-bounds frequencies are treated as ν → ∞ and contribute 0. Individual
asymptotic classes can be toggled via the keyword flags `K1`, `K2`, `K3`.
"""
@inline function (γ :: ChannelViewX2X{Q})(
    Ω  :: Union{MatsubaraFrequency, InfiniteMatsubaraFrequency},
    ν  :: Union{MatsubaraFrequency, InfiniteMatsubaraFrequency},
    νp :: Union{MatsubaraFrequency, InfiniteMatsubaraFrequency}
    ;
    K1 :: Bool = true,
    K2 :: Bool = true,
    K3 :: Bool = true,
    )  :: Q where {Q}

    val = zero(Q)

    if is_inbounds(Ω, meshes(γ.K1, Val(1)))
        K1 && (val += γ.K1[Ω])

        if is_inbounds(Ω, meshes(γ.K2, Val(1)))
            ν1_inbounds = is_inbounds(ν,  meshes(γ.K2, Val(2)))
            ν2_inbounds = is_inbounds(νp, meshes(γ.K2, Val(2)))

            if ν1_inbounds && ν2_inbounds
                # both fermionic frequencies in-bounds: K2, K2p, and K3 all contribute
                K2 && (val += γ.K2[Ω, ν] + γ.K2p[Ω, νp])
                K3 && (val += γ.K3(Ω, ν, νp))

            elseif ν1_inbounds
                # only ν in-bounds: K2 contributes; K2p and K3 vanish (ν' → ∞ limit)
                K2 && (val += γ.K2[Ω, ν])

            elseif ν2_inbounds
                # only ν' in-bounds: K2p contributes; K2 and K3 vanish (ν → ∞ limit)
                K2 && (val += γ.K2p[Ω, νp])
            end
        end
    end

    return val
end


"""
    (F::VertexViewX2X{Q})(Ω, ν, ν', ::Type{Ch}, ::Type{pSp}; F0, γp, γt, γa) -> Q

Evaluates the physical (particle) spin component of the momentum-fixed full vertex:
    F_pSp(Ω, ν, ν') = F0_pSp + γ_p + γ_t + γ_a

Each channel contribution is routed through the appropriate frequency conversion from `Ch`
into the native channel convention. Components can be toggled via the keyword flags.
"""
@inline function (F :: VertexViewX2X{Q})(
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

    if γp
        val += F.γpp(convert_frequency(Ω, ν, νp, Ch, pCh)...)
    end

    if γt
        val += F.γtp(convert_frequency(Ω, ν, νp, Ch, tCh)...)
    end

    if γa
        val += F.γap(convert_frequency(Ω, ν, νp, Ch, aCh)...)
    end

    return val
end

"""
    (F::VertexViewX2X{Q})(Ω, ν, ν', ::Type{Ch}, ::Type{xSp}; F0, γp, γt, γa) -> Q

Evaluates the exchange spin component of the momentum-fixed full vertex using crossing symmetry.
The crossing relation maps F_xSp to a sum over the channel views pre-stored with the exchanged
momentum (`γpx`, `γtx`, `γax`), with a global sign flip relative to the pSp evaluator:
    F_xSp = F0_xSp - [γ_px + γ_ax + γ_tx]
The channel-to-channel crossing map (p↔p, t↔a) and the corresponding frequency conversions
depend on the evaluation channel `Ch`.
"""
@inline function (F :: VertexViewX2X{Q})(
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

    val = zero(Q)

    if F0
        val += F.F0(Ω, ν, νp, Ch, xSp)
    end

    # The crossed channel views (γpx, γtx, γax) already encode the exchanged momentum.
    # The frequency argument is converted from Ch into the crossed channel's convention.
    # The sign is negative: xSp = pSp - 2*(exchange diagrams).
    if Ch === pCh
        γp && (val -= F.γpx(convert_frequency(Ω, ν, Ω - νp, pCh, pCh)...))
        γt && (val -= F.γax(convert_frequency(Ω, ν, Ω - νp, pCh, aCh)...))
        γa && (val -= F.γtx(convert_frequency(Ω, ν, Ω - νp, pCh, tCh)...))

    elseif Ch === tCh
        γp && (val -= F.γpx(convert_frequency(Ω, ν, νp, aCh, pCh)...))
        γt && (val -= F.γax(convert_frequency(Ω, ν, νp, aCh, aCh)...))
        γa && (val -= F.γtx(convert_frequency(Ω, ν, νp, aCh, tCh)...))

    elseif Ch === aCh
        γp && (val -= F.γpx(convert_frequency(Ω, ν, νp, tCh, pCh)...))
        γt && (val -= F.γax(convert_frequency(Ω, ν, νp, tCh, aCh)...))
        γa && (val -= F.γtx(convert_frequency(Ω, ν, νp, tCh, tCh)...))

    end

    return val
end


"""
    (F::VertexViewX2X{Q})(Ω, ν, ν', ::Type{Ch}, ::Type{dSp}; F0, γp, γt, γa) -> Q

Evaluates the density spin component of the momentum-fixed full vertex:
    F_dSp = 2 * F_pSp + F_xSp
"""
@inline function (F :: VertexViewX2X{Q})(
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

    val += F(Ω, ν, νp, Ch, pSp; F0, γp, γt, γa) * 2

    val += F(Ω, ν, νp, Ch, xSp; F0, γp, γt, γa)

    return val
end
