# =============================================================================
# Bethe-Salpeter equation for the K2 class (local solver).
#
# K2_r(Ω, ν) is the part of Φ_r that is left when the right outer frequency
# goes to infinity, minus K1: Φ_r(Ω, ν, ∞) = K1_r(Ω) + K2_r(Ω, ν). It is tied
# to the three-leg (Hedin) vertex of channel r (Lihm 2025, Eq. (S19c)).
#
# From the mirrored fd-BSE (Lihm 2025, Eq. (5); docs/physics_and_notation.md,
# Sec. 1.2)
#
#   Φ̃_r = F Π̃_r f + F Π_r Ĩ_r + F Π_r Ĩ_r π_r f + Ĩ_r π_r f
#
# one gets the K2 equation (Eq. (S29b), mirrored):
#
#   K̃2_r(Ω, ν) = 1/β ∑_ν̃ [F - F_01̄2](Ω, ν, ν̃) [ Π̃_r(Ω, ν̃) f_012(Ω, ν̃)
#                                              + Π_r(Ω, ν̃) FL_012(Ω, ν̃) ]
#                + FL_012(Ω, ν)
#
#   F - F_01̄2   the target vertex minus its ν → ∞ limit. F_01̄2 alone gives K1,
#               so the difference gives the ν-dependent part, i.e. K2.
#   FL_012      the K2 class of the left part Ĩ_r π_r f (BSE_L_K2!).
#
# The term F Π_r Ĩ_r does not reach K2: Ĩ_r is made of the other two channels,
# which vanish when the right outer frequency goes to infinity.
#
# Paper ↔ code: ω (bosonic) ↔ Ω; ν̃ (summed) ↔ ω; f, π ↔ F0, Π0;
# 1/β ∑_ν̃ ↔ temperature(F) * sum. The channel, spin and sign come from the
# wrappers in BSE_templates.jl.
# =============================================================================

"""
    BSE_L_K2!(K2, F0, F, Π0, Π, SG, sign, ::Type{Ch}, ::Type{Sp}, is_mfRG = Val(false); mode) -> nothing

Compute the K2 class of the left part of the fd-BSE and write it into `K2`:

    FL_012(Ω, ν) = 1/β ∑_ν̃ Ĩ_r(Ω, ν, ν̃) π_r(Ω, ν̃) f_012(Ω, ν̃)

i.e. `Ĩ_r π_r f` with the right outer frequency at infinity (the last term of
the mirrored Eq. (5)). `Ĩ_r` is the target vertex with its reference part and
its own channel switched off, so the sum of the other two fd channels. Only
the reference bubble π enters. The left part has no K1 class, because `Ĩ_r`
vanishes when ν → ∞.

`sign`, `Ch`, `Sp` and `SG` are as in [`BSE_K1!`](@ref).
"""
function BSE_L_K2!(
    K2   :: MF_K2{Q},
    F0   :: Union{RefVertex{Q}, AbstractVertex{Q}},
    F    :: AbstractVertex{Q},
    Π0   :: MF_Π{Q},
    Π    :: MF_Π{Q},
    SG   :: SymmetryGroup{2, Q},
    sign :: Int,
         :: Type{Ch},
         :: Type{Sp},
    is_mfRG :: Union{Val{true}, Val{false}} = Val(false),
    ;
    mode :: Symbol = :serial
    ) where {Q, Ch <: ChannelTag, Sp <: SpinTag}

    # FL_012(Ω, ν) for one (Ω, ν): sum over the internal frequency ν̃ (called ω)
    @inline function diagram(wtpl)

        Ω, ν    = wtpl
        val     = zero(Q)
        Π0slice = view(Π0, Ω, :)

        for i in eachindex(Π0slice)
            ω = value(meshes(Π0, Val(2))[i])

            # Γp ("p" for prime, as F0 is already taken) = Ĩ_r(Ω, ν, ν̃): the
            # target vertex without F0 (the reference) and without its own
            # channel r, i.e. the sum of the other two fd channels
            Γp  = F(Ω, ν, ω, Ch, Sp; F0 = false, γa = (Ch !== aCh), γp = (Ch !== pCh), γt = (Ch !== tCh))
            # f_012(Ω, ν̃): reference vertex, right outer leg at infinity
            F0p = F0(Ω, _crossing(Ω, ω, Ch), νInf, Ch, Sp)

            val += Γp * Π0slice[i] * F0p
        end

        # 1/β ∑_ν̃, and the channel sign
        return temperature(F) * val * sign
    end

    # evaluate one point per symmetry class; the symmetry group fills in the rest
    SG(K2, InitFunction{2, Q}(diagram); mode)

    return nothing
end

"""
    BSE_K2!(K2, F0, F, FL, Π0, Π, SG, sign, ::Type{Ch}, ::Type{Sp}, is_mfRG = Val(false); mode) -> nothing

Compute the fd K2 class `K̃2_r(Ω, ν)` of channel `Ch` and write it into `K2`
(Lihm 2025, Eq. (S29b), mirrored; see the file header):

    K̃2_r = (F - F_01̄2) [Π̃_r f_012 + Π_r FL_012] + FL_012

The arguments are as in [`BSE_K1!`](@ref); `FL` must hold the left part from
`BSE_L_K2!`. In the t channel, which is solved in the density component, the
left part is added in its density form `2·FL_t - FL_a`.

With `is_mfRG = Val(true)` it computes the K2 part of the linearized fd-BSE
used as preconditioner (Eq. (S74c)): `(f - f_01̄2) π FL_012 + FL_012`, built
from reference quantities only.
"""
function BSE_K2!(
    K2   :: MF_K2{Q},
    F0   :: Union{RefVertex{Q}, AbstractVertex{Q}},
    F    :: AbstractVertex{Q},
    FL   :: AbstractVertex{Q},
    Π0   :: MF_Π{Q},
    Π    :: MF_Π{Q},
    SG   :: SymmetryGroup{2, Q},
    sign :: Int,
         :: Type{Ch},
         :: Type{Sp},
    is_mfRG :: Union{Val{true}, Val{false}} = Val(false),
    ;
    mode :: Symbol = :serial
    ) where {Q, Ch <: ChannelTag, Sp <: SpinTag}

    # K̃2(Ω, ν) for one (Ω, ν): sum over the internal frequency ν̃ (called ω)
    @inline function diagram(wtpl)

        Ω, ν    = wtpl
        val     = zero(Q)
        Π0slice = view(Π0, Ω, :)
        Πslice  = view(Π, Ω, :)

        for i in eachindex(Π0slice)
            ω = value(meshes(Π0, Val(2))[i])

            if is_mfRG === Val(true)
                # linearized BSE: (f - f_01̄2)(Ω, ν, ν̃) and the left part FL_012(Ω, ν̃)
                F0l = F0(Ω, ν, ω, Ch, Sp) - F0(Ω, νInf, ω, Ch, Sp)
                FLr = FL(Ω, _crossing(Ω, ω, Ch), νInf, Ch, Sp)

                # central part
                val += F0l * Π0slice[i] * FLr

            else
                # Fl = (F - F_01̄2)(Ω, ν, ν̃): the ν-dependent part of the target vertex
                Fl  = F( Ω, ν,    ω, Ch, Sp) - F(Ω, νInf, ω, Ch, Sp)
                # F0r = f_012(Ω, ν̃), reference vertex with the right leg at infinity
                F0r = F0(Ω, _crossing(Ω, ω, Ch), νInf, Ch, Sp)
                # FLr = FL_012(Ω, ν̃) = [Ĩ_r π_r f](Ω, ν̃, ∞), the left part
                FLr = FL(Ω, _crossing(Ω, ω, Ch), νInf, Ch, Sp)

                # 1ℓ term (F - F_01̄2) Π̃ f + central part (F - F_01̄2) Π (Ĩ π f)
                val += Fl * ((Πslice[i] - Π0slice[i]) * F0r + Πslice[i] * FLr)
            end
        end

        # 1/β ∑_ν̃, and the channel sign
        return temperature(F) * val * sign
    end

    # evaluate one point per symmetry class; the symmetry group fills in the rest
    SG(K2, InitFunction{2, Q}(diagram); mode)

    # add the left part FL_012(Ω, ν) itself
    if Ch === tCh
        # density form 2·FL_t - FL_a, since the t channel is solved in dSp
        mult_add!(K2, FL.γt.K2, 2)
        mult_add!(K2, FL.γa.K2, -1)
    else
        add!(K2, get_reducible_vertex(FL, Ch).K2)
    end

    return nothing
end




"""
    BSE_K2_new!(K2, F0, F, Π0, Π, SG, sign, ::Type{Ch}, ::Type{Sp}, is_mfRG = Val(false); mode) -> nothing

Compute the fd K2 class `K̃2_r(Ω, ν)` from the "U-closed" form of the BSE and
write it into `K2`. Used by the `:fdPA_new` and `:scPA_new` strategies.

When the right outer frequency goes to infinity, the irreducible vertex next to
it reduces to the bare U. With `K1 = F_01̄2 Π U` (see `BSE_K1_new!`), this
leaves `K2 = (F - F_01̄2) Π U` (Lihm 2025, Eq. (S23c), mirrored). Doing this for
target and reference and subtracting gives

    K̃2_r(Ω, ν) = 1/β ∑_ν̃ [ (F - F_01̄2) Π_r - (f - f_01̄2) π_r ](Ω, ν, ν̃) U

This needs neither the left part `FL` nor Ĩ. The arguments are as in
[`BSE_K1!`](@ref); `is_mfRG` is accepted but not used.
"""
function BSE_K2_new!(
    K2   :: MF_K2{Q},
    F0   :: Union{RefVertex{Q}, AbstractVertex{Q}},
    F    :: AbstractVertex{Q},
    Π0   :: MF_Π{Q},
    Π    :: MF_Π{Q},
    SG   :: SymmetryGroup{2, Q},
    sign :: Int,
         :: Type{Ch},
         :: Type{Sp},
    is_mfRG :: Union{Val{true}, Val{false}} = Val(false),
    ;
    mode :: Symbol = :serial
    ) where {Q, Ch <: ChannelTag, Sp <: SpinTag}

    # bare vertex closing the diagram on the right (+U for pSp and dSp)
    U = bare_vertex(F, Sp)

    # K̃2(Ω, ν) for one (Ω, ν): sum over the internal frequency ν̃ (called ω)
    @inline function diagram(wtpl)

        Ω, ν    = wtpl
        val     = zero(Q)
        Π0slice = view(Π0, Ω, :)
        Πslice  = view(Π,  Ω, :)

        for i in eachindex(Π0slice)
            ω = value(meshes(Π0, Val(2))[i])

            # ν-dependent parts of the target and reference vertex
            Fl  = F( Ω, ν, ω, Ch, Sp) - F( Ω, νInf, ω, Ch, Sp)
            F0l = F0(Ω, ν, ω, Ch, Sp) - F0(Ω, νInf, ω, Ch, Sp)

            # [(F - F_01̄2) Π - (f - f_01̄2) π] U
            val += (Fl * Πslice[i] - F0l * Π0slice[i]) * U
        end

        # 1/β ∑_ν̃, and the channel sign
        return temperature(F) * val * sign
    end

    # evaluate one point per symmetry class; the symmetry group fills in the rest
    SG(K2, InitFunction{2, Q}(diagram); mode)

    return nothing
end
