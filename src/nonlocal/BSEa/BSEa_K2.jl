# =============================================================================
# Bethe-Salpeter equation for the K2 class, s-wave lattice solver NL.
#
# Lattice version of BSEa/BSEa_K2.jl (see there for the derivation). At each
# bosonic momentum q (code: P), the mirrored Eq. (S29b) reads
#
#   K̃2_r(Ω, ν; q) = 1/β ∑_ν̃ (F - F_01̄2)(Ω, ν, ν̃; q) [ Π̃_r f_012 + Π_r FL_012 ](Ω, ν̃; q)
#                   + FL_012(Ω, ν; q)
#
# with all vertices averaged over the fermionic momenta (evaluated at
# (kSW, kSW)) and the k-averaged NL bubbles (s-wave approximation, Lihm 2025,
# Eq. (S17)). FL_012 = Ĩ_r π_r f_012 is the left part (BSE_L_K2!). The term
# F Π Ĩ does not reach K2, because Ĩ_r vanishes when the right outer frequency
# goes to infinity.
#
# Paper ↔ code: q ↔ P; ν̃ (summed) ↔ ω; f, π ↔ F0, Π0.
# =============================================================================

"""
    BSE_L_K2!(K2::NL_MF_K2, F0, F, Π0, Π, SG, sign, ::Type{Ch}, ::Type{Sp}, is_mfRG = Val(false); mode = :serial) -> nothing

Lattice (NL) version of [`BSE_L_K2!`](@ref): the K2 class of the left part,

    FL_012(Ω, ν; q) = 1/β ∑_ν̃ Ĩ_r(Ω, ν, ν̃; q) π_r(Ω, ν̃; q) f_012(Ω, ν̃; q),

with s-wave averaged vertices. Here the sum over ν̃ runs over the fermionic box
of K2, and the p-channel crossing (`_crossing`) is applied to the left vertex
`Ĩ_r` instead of the right one. This is equivalent because the pp bubble is
symmetric under ν̃ → Ω - ν̃, apart from the edges of the finite ν̃ box.
"""
function BSE_L_K2!(
    K2   :: NL_MF_K2{Q},
    F0   :: Union{RefVertex{Q}, AbstractVertex{Q}},
    F    :: AbstractVertex{Q},
    Π0   :: NL_MF_Π{Q},
    Π    :: NL_MF_Π{Q},
    SG   :: SymmetryGroup{3, Q},
    sign :: Int,
         :: Type{Ch},
         :: Type{Sp},
    is_mfRG :: Union{Val{true}, Val{false}} = Val(false),
    ;
    mode :: Symbol = :serial
    ) where {Q, Ch <: ChannelTag, Sp <: SpinTag}

    # FL_012(Ω, ν; q) for one point: sum over ν̃ (called ω) on the K2 box
    @inline function diagram(wtpl)

        Ω, ν, P = wtpl
        val     = zero(Q)
        Π0slice = MeshFunction((meshes(Π0, Val(2)),), view(Π0, Ω, :, P))

        for iω in eachindex(meshes(K2, Val(2)))
            ω = value(meshes(K2, Val(2))[iω])

            # Ĩ_r: target vertex without F0 and without its own channel r
            Γl  = F( Ω, ν, _crossing(Ω, ω, Ch), P, kSW, kSW, Ch, Sp; F0 = false, γa = (Ch !== aCh), γp = (Ch !== pCh), γt = (Ch !== tCh))
            # f_012: reference vertex, right outer leg at infinity
            F0r = F0(Ω, ω, νInf, P, kSW, kSW, Ch, Sp)

            val += Γl * Π0slice[ω] * F0r
        end

        # 1/β ∑_ν̃, and the channel sign
        return temperature(F) * val * sign
    end

    # evaluate one point per symmetry class; the symmetry group fills in the rest
    SG(K2, InitFunction{3, Q}(diagram); mode)

    return nothing
end


"""
    BSE_K2!(K2::NL_MF_K2, F0, F, FL, Π0, Π, SG, sign, ::Type{Ch}, ::Type{Sp}, is_mfRG = Val(false); mode = :serial) -> nothing

Lattice (NL) version of [`BSE_K2!`](@ref): `K̃2_r(Ω, ν; q)` from the mirrored
Eq. (S29b) with s-wave averaged vertices (see the file header), written into
`K2`. The left part `FL_012` is added at the end (in the t channel in its density
form `2·FL_t - FL_a`). With `is_mfRG = Val(true)` it computes the linearized
form used by the preconditioner, `(f - f_01̄2) π FL_012 + FL_012`.
"""
function BSE_K2!(
    K2   :: NL_MF_K2{Q},
    F0   :: Union{RefVertex{Q}, AbstractVertex{Q}},
    F    :: AbstractVertex{Q},
    FL   :: AbstractVertex{Q},
    Π0   :: NL_MF_Π{Q},
    Π    :: NL_MF_Π{Q},
    SG   :: SymmetryGroup{3, Q},
    sign :: Int,
         :: Type{Ch},
         :: Type{Sp},
    is_mfRG :: Union{Val{true}, Val{false}} = Val(false),
    ;
    mode :: Symbol = :serial
    ) where {Q, Ch <: ChannelTag, Sp <: SpinTag}

    # K̃2(Ω, ν; q) for one point: sum over the internal frequency ν̃ (called ω)
    @inline function diagram(wtpl)

        Ω, ν, P = wtpl
        val     = zero(Q)
        Π0slice = view(Π0, Ω, :, P)
        Πslice  = view(Π , Ω, :, P)

        for iω in eachindex(Π0slice)
            ω = value(meshes(Π0, Val(2))[iω])

            if is_mfRG === Val(true)

                # linearized BSE: (f - f_01̄2) and the left part (crossing on the left)
                Fl  = F0(Ω, ν, _crossing(Ω, ω, Ch), P, kSW, kSW, Ch, Sp) - F0(Ω, νInf, _crossing(Ω, ω, Ch), P, kSW, kSW, Ch, Sp)
                FLr = FL(Ω, ω, νInf, P, kSW, kSW, Ch, Sp)

                # 1ℓ and central part
                val += Fl * Π0slice[iω] * FLr

            else

                # (F - F_01̄2), f_012 and FL_012, averaged over the fermionic momenta
                Fl  = F( Ω, ν, ω, P, kSW, kSW, Ch, Sp) - F( Ω, νInf, ω, P, kSW, kSW, Ch, Sp)
                F0r = F0(Ω, _crossing(Ω, ω, Ch), νInf, P, kSW, kSW, Ch, Sp)
                FLr = FL(Ω, _crossing(Ω, ω, Ch), νInf, P, kSW, kSW, Ch, Sp)

                # 1ℓ and central part
                val += Fl * ((Πslice[iω] - Π0slice[iω]) * F0r + Πslice[iω] * FLr)

            end

        end

        # 1/β ∑_ν̃, and the channel sign
        return temperature(F) * val * sign
    end

    # evaluate one point per symmetry class; the symmetry group fills in the rest
    SG(K2, InitFunction{3, Q}(diagram); mode)

    # add the left part FL_012(Ω, ν; q); density form in the t channel
    if Ch === tCh
        mult_add!(K2, FL.γt.K2, 2)
        mult_add!(K2, FL.γa.K2, -1)
    else
        add!(K2, get_reducible_vertex(FL, Ch).K2)
    end

    return nothing
end


"""
    BSE_K2_new!(K2::NL_MF_K2, F0, F, Π0, Π, SG, sign, ::Type{Ch}, ::Type{Sp}, is_mfRG = Val(false); mode = :serial) -> nothing

Lattice (NL) version of [`BSE_K2_new!`](@ref), the "U-closed" form
`K̃2(Ω, ν; q) = 1/β ∑_ν̃ [(F - F_01̄2) Π - (f - f_01̄2) π](Ω, ν, ν̃; q) U`, with
s-wave averaged vertices. With `is_mfRG = Val(true)` it computes
`[(F - F_01̄2) - (f - f_01̄2)] Π U` instead.
"""
function BSE_K2_new!(
    K2   :: NL_MF_K2{Q},
    F0   :: AbstractVertex{Q},
    F    :: AbstractVertex{Q},
    Π0   :: NL_MF_Π{Q},
    Π    :: NL_MF_Π{Q},
    SG   :: SymmetryGroup{3, Q},
    sign :: Int,
         :: Type{Ch},
         :: Type{Sp},
    is_mfRG :: Union{Val{true}, Val{false}} = Val(false),
    ;
    mode :: Symbol = :serial
    ) where {Q, Ch <: ChannelTag, Sp <: SpinTag}

    # bare vertex closing the diagram on the right
    U = bare_vertex(F, Sp)

    # K̃2(Ω, ν; q) for one point: sum over the internal frequency ν̃ (called ω)
    @inline function diagram(wtpl)

        Ω, ν, P = wtpl
        val     = zero(Q)
        Π0slice = view(Π0, Ω, :, P)
        Πslice  = view(Π , Ω, :, P)

        for i in eachindex(Π0slice)
            ω = value(meshes(Π0, Val(2))[i])

            # ν-dependent parts of the target and reference vertex
            Fl  = F( Ω, ν, ω, P, kSW, kSW, Ch, Sp) - F( Ω, νInf, ω, P, kSW, kSW, Ch, Sp)
            F0l = F0(Ω, ν, ω, P, kSW, kSW, Ch, Sp) - F0(Ω, νInf, ω, P, kSW, kSW, Ch, Sp)

            # 1ℓ and central part
            if is_mfRG === Val(true)
                val += (Fl - F0l) * Πslice[i] * U
            else
                val += (Fl * Πslice[i] - F0l * Π0slice[i]) * U
            end
        end

        # 1/β ∑_ν̃, and the channel sign
        return temperature(F) * val * sign
    end

    # evaluate one point per symmetry class; the symmetry group fills in the rest
    SG(K2, InitFunction{3, Q}(diagram); mode)

    return nothing
end
