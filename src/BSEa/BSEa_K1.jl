# =============================================================================
# Bethe-Salpeter equation for the K1 class (local solver).
#
# K1_r(Ω) is the part of the reducible vertex Φ_r that depends only on the
# bosonic frequency: the limit of Φ_r(Ω, ν, ν′) for ν, ν′ → ∞. It is
# proportional to the susceptibility of channel r (Lihm 2025, Eq. (S19a)).
#
# The code solves the finite-difference (fd) BSE in the left-right mirrored
# form of Lihm 2025, Eq. (5) (see docs/physics_and_notation.md, Sec. 1.2):
#
#   Φ̃_r = F Π̃_r f + F Π_r Ĩ_r + F Π_r Ĩ_r π_r f + Ĩ_r π_r f
#
# Sending both outer fermionic frequencies to infinity gives the K1 equation
# (Eq. (S29a), mirrored):
#
#   K̃1_r(Ω) = 1/β ∑_ν̃ F_01̄2(Ω, ν̃) [ Π̃_r(Ω, ν̃) f_012(Ω, ν̃)
#                                   + Π_r(Ω, ν̃) FL_012(Ω, ν̃) ]
#
#   F_01̄2(Ω, ν̃) = F(Ω, ∞, ν̃)    target vertex, left outer leg at infinity
#   f_012(Ω, ν̃) = f(Ω, ν̃, ∞)        reference vertex, right outer leg at infinity
#   FL = Ĩ_r π_r f_012          the "left part", computed before by BSE_L_K2!
#   Π̃_r = Π_r - π_r             target minus reference bubble
#
# The first term is the "1ℓ" (one-loop) term F Π̃ f, the second the "central
# part" F Π Ĩ π f. The terms F Π Ĩ and Ĩ π f do not reach K1: Ĩ_r is made of
# the other two channels, and those vanish once an outer frequency of
# channel r goes to infinity.
#
# Paper ↔ code: ω (bosonic) ↔ Ω; ν̃ (summed) ↔ ω; f, π ↔ F0, Π0;
# F_01̄2 ↔ F(Ω, νInf, ω); 1/β ∑_ν̃ ↔ temperature(F) * sum. The argument `sign`
# absorbs the channel sign and the minus sign of the stored bubble
# Π = (iG)(iG) (BSE_templates.jl, docs Sec. 3).
# =============================================================================

"""
    BSE_K1!(K1, F0, F, FL, Π0, Π, SG, sign, ::Type{Ch}, ::Type{Sp}, is_mfRG = Val(false); mode) -> nothing

Compute the fd K1 class `K̃1_r(Ω)` of channel `Ch` and write it into `K1`
(Lihm 2025, Eq. (S29a) in the mirrored form; see the file header).

- `F0`, `Π0`: reference vertex f and bubble π (e.g. from DMFT).
- `F`, `Π`: target vertex `F = f + ∑_r Φ̃_r` and bubble Π.
- `FL`: the left part `Ĩ_r π_r f`, computed before by `BSE_L_K2!`.
- `Sp`: spin component; `pSp` in the a and p channels, `dSp` (density) in the
  t channel.
- `sign`: the channel's sign (+1 for a and p, -1 for t). It also absorbs the
  minus sign of the stored bubble `Π = (iG)(iG) = -GG`.
- `SG`: symmetry group of K1. Only one Ω per symmetry class is computed.

With `is_mfRG = Val(true)`, it computes instead the K1 part of the linearized
fd-BSE used as preconditioner (Eq. (S74c)): only the central part, built from
reference quantities, `f_01̄2 π FL_012`.
"""
function BSE_K1!(
    K1   :: MF_K1{Q},
    F0   :: Union{RefVertex{Q}, AbstractVertex{Q}},
    F    :: AbstractVertex{Q},
    FL   :: AbstractVertex{Q},
    Π0   :: MF_Π{Q},
    Π    :: MF_Π{Q},
    SG   :: SymmetryGroup{1, Q},
    sign :: Int,
         :: Type{Ch},
         :: Type{Sp},
    is_mfRG :: Union{Val{true}, Val{false}} = Val(false),
    ;
    mode :: Symbol = :serial
    ) where {Q, Ch <: ChannelTag, Sp <: SpinTag}

    # K̃1(Ω) for one bosonic frequency: a sum over the internal frequency ν̃,
    # which runs over the fermionic mesh of the bubble and is called ω here
    @inline function diagram(wtpl)

        Ω       = wtpl[1]
        val     = zero(Q)
        Π0slice = view(Π0, Ω, :)    # π(Ω, ν̃) for all ν̃
        Πslice  = view(Π, Ω, :)     # Π(Ω, ν̃) for all ν̃

        for i in eachindex(Π0slice)
            ω = value(meshes(Π0, Val(2))[i])

            if is_mfRG == Val(true)
                # Linearized BSE (preconditioner): reference quantities only.
                # NOTE: the local solver never gets here: it has no
                # preconditioned iteration (no local build_K3_cache_mfRG!).
                F0l = F0(Ω, νInf, ω, Ch, Sp)
                FLr = FL(Ω, _crossing(Ω, ω, Ch), νInf, Ch, Sp)

                # central part
                val += F0l * Π0slice[i] * FLr
            else
                # Fl = F_01̄2(Ω, ν̃) = F(Ω, ν → ∞, ν̃), target vertex
                Fl  = F( Ω, νInf, ω, Ch, Sp)
                # F0r = f_012(Ω, ν̃) = f(Ω, ν̃, ν′ → ∞), reference vertex. In the
                # p channel the right vertex sits at Ω - ν̃ (Eq. (S15)), which
                # _crossing takes care of.
                F0r = F0(Ω, _crossing(Ω, ω, Ch), νInf, Ch, Sp)
                # FLr = FL_012(Ω, ν̃) = [Ĩ_r π_r f](Ω, ν̃, ν′ → ∞), the left part
                FLr = FL(Ω, _crossing(Ω, ω, Ch), νInf, Ch, Sp)

                # 1ℓ term F Π̃ f + central part F Π (Ĩ π f)
                val += Fl * ((Πslice[i] - Π0slice[i]) * F0r + Πslice[i] * FLr)
            end
        end

        # 1/β ∑_ν̃, and the channel sign
        return temperature(F) * val * sign
    end

    # evaluate one Ω per symmetry class; the symmetry group fills in the rest
    SG(K1, InitFunction{1, Q}(diagram); mode)

    return nothing
end


"""
    BSE_K1_new!(K1, F0, F, Π0, Π, SG, sign, ::Type{Ch}, ::Type{Sp}, is_mfRG = Val(false); mode) -> nothing

Compute the fd K1 class `K̃1_r(Ω)` from the "U-closed" form of the BSE and
write it into `K1`. Used by the `:fdPA_new` and `:scPA_new` strategies.

When the right outer frequency goes to infinity, the irreducible vertex next to
it reduces to the bare interaction U (Lihm 2025, Eq. (S23a): `K1 = F₀ Π F_012`,
here mirrored). Doing this for target and reference and subtracting gives

    K̃1_r(Ω) = 1/β ∑_ν̃ [ F_01̄2(Ω, ν̃) Π_r(Ω, ν̃) - f_01̄2(Ω, ν̃) π_r(Ω, ν̃) ] U

This needs neither the left part `FL` nor Ĩ. `U` is the bare vertex of spin
component `Sp`. The arguments are as in [`BSE_K1!`](@ref); `is_mfRG` is
accepted but not used.
"""
function BSE_K1_new!(
    K1   :: MF_K1{Q},
    F0   :: Union{RefVertex{Q}, AbstractVertex{Q}},
    F    :: AbstractVertex{Q},
    Π0   :: MF_Π{Q},
    Π    :: MF_Π{Q},
    SG   :: SymmetryGroup{1, Q},
    sign :: Int,
         :: Type{Ch},
         :: Type{Sp},
    is_mfRG :: Union{Val{true}, Val{false}} = Val(false),
    ;
    mode :: Symbol = :serial
    ) where {Q, Ch <: ChannelTag, Sp <: SpinTag}

    # bare vertex closing the diagram on the right (+U for pSp and dSp)
    U = bare_vertex(F, Sp)

    # K̃1(Ω) for one bosonic frequency, summed over the internal ν̃ (called ω)
    @inline function diagram(wtpl)

        Ω       = wtpl[1]
        val     = zero(Q)
        Π0slice = view(Π0, Ω, :)
        Πslice  = view(Π,  Ω, :)

        for i in eachindex(Π0slice)
            ω = value(meshes(Π0, Val(2))[i])

            # F_01̄2 and f_01̄2: target and reference vertex, left leg at infinity
            Fl  = F( Ω, νInf, ω, Ch, Sp)
            F0l = F0(Ω, νInf, ω, Ch, Sp)

            # (F Π - f π) U
            val += (Fl * Πslice[i] - F0l * Π0slice[i]) * U
        end

        # 1/β ∑_ν̃, and the channel sign
        return temperature(F) * val * sign
    end

    # evaluate one Ω per symmetry class; the symmetry group fills in the rest
    SG(K1, InitFunction{1, Q}(diagram); mode)

    return nothing
end
