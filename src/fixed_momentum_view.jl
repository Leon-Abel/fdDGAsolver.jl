# =============================================================================
# Fixed-momentum views of lattice channels and vertices (NL and NL2).
#
# A view fixes the momenta (q, k, k′) of a lattice channel or vertex and keeps
# only the frequency dependence. It is a light object that reads the stored
# arrays at those momenta, so the frequency sums of a BSE need no momentum
# look-ups. The NL2 BSE kernels used to evaluate their vertices this way (the
# commented-out lines in nonlocal_2/BSEa/); now the views are used only in the
# tests.
#
# Channel view (ChannelViewX2X): with the momenta fixed, the reducible vertex
# of one channel (Lihm 2025, Eq. (S7)) reads
#
#   Φ_r(Ω, ν, ν′) = K1(Ω) + K2(Ω, ν) + K̄2(Ω, ν′) + K3(Ω, ν, ν′)
#
# where K̄2 is the K2 array read at the second fermionic momentum (field K2p).
# A class whose frequency lies outside its box contributes zero.
#
# Vertex view (VertexViewX2X): the three channels at the momenta converted
# from channel Ch (convention.jl), plus the same three channels at the momenta
# of the crossed vertex. The second set gives the crossed spin component xSp
# by crossing symmetry (Lihm 2025, Eq. (S4a)) without converting momenta at
# every call.
#
# Paper ↔ code: q ↔ P; k′ ↔ kp (vertex) or q (channel); ν′ ↔ νp.
# =============================================================================

# -----------------------------------------------------------------------------
# Types
# -----------------------------------------------------------------------------


"""
    ChannelViewX2X{Q, T1, T2, T3} <: AbstractReducibleVertex{Q}

A momentum-fixed view of a single reducible channel (`NL_Channel` or `NL2_Channel`).

Stores array views of K1, K2, K2p, and K3 with all momentum indices already sliced out,
so only the Matsubara frequency axes remain. `Q` is the element type (e.g. ComplexF64).
`K2p` is the class K̄2(Ω, ν′): a view of the same K2 array, at the second
fermionic momentum.
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

Stores the vertex `F0` the channels are added to (a `RefVertex` in the PA, the
reference f in fd runs; local vertices are used as they are) and six
momentum-fixed channel views:
- `γpp`, `γtp`, `γap`: `Φ_p`, `Φ_t`, `Φ_a` at the momenta of F, for the parallel
  spin component pSp;
- `γpx`, `γtx`, `γax`: the same channels at the momenta of the crossed vertex,
  for the crossed spin component xSp.

`Q` is the element type (e.g. ComplexF64).
"""
struct VertexViewX2X{Q, VT, CT} <: AbstractVertex{Q}
    F0  :: VT   # vertex the channels are added to (local, or a view)
    γpp :: CT   # p channel at the momenta of F
    γtp :: CT   # t channel at the momenta of F
    γap :: CT   # a channel at the momenta of F
    γpx :: CT   # p channel at the momenta of the crossed vertex
    γtx :: CT   # t channel at the momenta of the crossed vertex
    γax :: CT   # a channel at the momenta of the crossed vertex

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

# Temperature and box sizes, from the p-channel view.
function MatsubaraFunctions.temperature(F :: VertexViewX2X) :: Float64
    return MatsubaraFunctions.temperature(F.γpp)
end

numK1(F :: VertexViewX2X) :: Int64            = numK1(F.γpp)
numK2(F :: VertexViewX2X) :: NTuple{2, Int64} = numK2(F.γpp)
numK3(F :: VertexViewX2X) :: NTuple{2, Int64} = numK3(F.γpp)


# -----------------------------------------------------------------------------
# Building views
# -----------------------------------------------------------------------------

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

`NL2_Channel` stores one K2 array over `(Ω, ν, P, k)`. The view K2 is sliced
from it at `(iP, ik)` and K2p (K̄2) at `(iP, iq)`, with `q` the second
fermionic momentum k′. K3 depends only on `P` (not on the fermionic momenta).
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

These are the momentum parts of the crossing relations used for xSp in
nonlocal/vertex.jl. `F.F0` is turned into a view as well; a local `F0` is used
as it is. There is no method for MBE vertices.
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
    # (In the t and a channels the same momenta are read in the other
    # channel's parametrization, as in the xSp method of nonlocal/vertex.jl.)
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


# -----------------------------------------------------------------------------
# Evaluation
# -----------------------------------------------------------------------------

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

Evaluates the parallel spin component F^= (pSp) of the momentum-fixed full vertex:
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

Evaluates the crossed spin component F^× (xSp) of the momentum-fixed full vertex using
crossing symmetry (Lihm 2025, Eq. (S4a)): F^× is minus F^= at the crossed point.
The crossing relation maps `F_xSp` to a sum over the channel views pre-stored with the exchanged
momentum (`γpx`, `γtx`, `γax`), with a global sign flip relative to the pSp evaluator:
    F_xSp = F0_xSp - [γ_px + γ_ax + γ_tx]
The channel-to-channel crossing map (p↔p, t↔a) and the corresponding frequency conversions
depend on the evaluation channel `Ch`. Because crossing exchanges t and a, the
flag `γt` switches the a-channel term and `γa` the t-channel term, as in
nonlocal/vertex.jl.
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
    # The sign is negative: F^× = -F^= at the crossed point (crossing symmetry).
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
