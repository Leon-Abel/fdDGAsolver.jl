# =============================================================================
# Bethe-Salpeter equation for the K2 class, solver NL2.
#
# In NL2 the K2 class keeps the fermionic momentum k of its leg. At each
# bosonic momentum q (code: P) and fermionic momentum k, the mirrored
# Eq. (S29b) reads
#
#   K̃2_r(Ω, ν; q, k) = 1/β 1/N_k ∑_{ν̃,k̃} (F - F_01̄2)(Ω, ν, ν̃; q, k, k̃)
#                          [ Π̃_r f_012 + Π_r FL_012 ](Ω, ν̃; q, k̃)
#                      + FL_012(Ω, ν; q, k)
#
# The left vertex is evaluated at all three momenta, without any averaging, so
# every channel inside it is taken at its exactly converted momenta. The
# bubbles and the vertices on the right (outer leg ν′ = ∞) depend on (q, k̃);
# in the p channel the right vertex sits at the crossed point (Ω - ν̃, q - k̃)
# (_crossing, BSE_templates.jl). The momentum of a leg at infinity does not
# matter, so the code passes the dummy k0 there. FL_012 = Ĩ_r π_r f_012 is the
# left part (BSE_L_K2!). The term F Π Ĩ does not reach K2, because Ĩ_r vanishes
# when the right outer frequency goes to infinity.
#
# The commented-out lines are an older evaluation of the same vertices through
# fixed_momentum_view (fixed_momentum_view.jl).
#
# Paper ↔ code: q ↔ P; ν̃, k̃ (summed) ↔ ω, q; f, π ↔ F0, Π0.
# =============================================================================

"""
    BSE_L_K2!(K2::NL2_MF_K2, F0, F, Π0::NL2_MF_Π, Π::NL2_MF_Π, SG, sign, ::Type{Ch}, ::Type{Sp}, is_mfRG = Val(false); mode = :serial) -> nothing

NL2 version of [`BSE_L_K2!`](@ref): the K2 class of the left part,

    FL_012(Ω, ν; q, k) = 1/β 1/N_k ∑_{ν̃,k̃} Ĩ_r(Ω, ν, ν̃; q, k, k̃) π_r(Ω, ν̃; q, k̃) f_012(Ω, ν̃; q, k̃)

The sums run over the frequency box and the momentum mesh of K2. As in the NL
version, the p-channel crossing is applied to the left vertex `Ĩ_r` instead
of the right one. This is equivalent because the pp bubble is symmetric under
(ν̃, k̃) → (Ω - ν̃, q - k̃), apart from the edges of the finite ν̃ box.
"""
function BSE_L_K2!(
    K2   :: NL2_MF_K2{Q},
    F0   :: Union{RefVertex{Q}, AbstractVertex{Q}},
    F    :: AbstractVertex{Q},
    Π0   :: NL2_MF_Π{Q},
    Π    :: NL2_MF_Π{Q},
    SG   :: SymmetryGroup{4, Q},
    sign :: Int,
         :: Type{Ch},
         :: Type{Sp},
    is_mfRG :: Union{Val{true}, Val{false}} = Val(false),
    ;
    mode :: Symbol = :serial
    ) where {Q, Ch <: ChannelTag, Sp <: SpinTag}

    # FL_012(Ω, ν; q, k) for one point: sum over ν̃ (called ω) and k̃ (called q)
    @inline function diagram(wtpl)

        Ω, ν, P, k = wtpl
        val     = zero(Q)
        Π0slice = MeshFunction((meshes(Π0, Val(2)), meshes(Π0, Val(4))), view(Π0, Ω, :, P, :))

        for iq in eachindex(meshes(K2, Val(4)))
            q = value(meshes(K2, Val(4))[iq])
            # Fview  = fixed_momentum_view(F,  P, k, _crossing(P, q, Ch), Ch)
            # F0view = fixed_momentum_view(F0, P, q, k0, Ch)

            for iω in eachindex(meshes(K2, Val(2)))
                ω = value(meshes(K2, Val(2))[iω])

                # vertices
                # Γl  = Fview( Ω, ν, _crossing(Ω, ω, Ch), Ch, Sp; F0 = false, γa = (Ch !== aCh), γp = (Ch !== pCh), γt = (Ch !== tCh))
                # F0r = F0view(Ω, ω, νInf, Ch, Sp)

                # Ĩ_r: target vertex without F0 and without its own channel r,
                # at the crossed point; f_012: right outer leg at infinity
                Γl  = F( Ω, ν, _crossing(Ω, ω, Ch), P, k, _crossing(P, q, Ch), Ch, Sp; F0 = false, γa = (Ch !== aCh), γp = (Ch !== pCh), γt = (Ch !== tCh))
                F0r = F0(Ω, ω, νInf, P, q, k0, Ch, Sp)

                val += Γl * Π0slice[ω, q] * F0r
            end
        end

        # 1/β 1/N_k ∑_{ν̃,k̃}, and the channel sign
        return temperature(F) * val / numP(F) * sign
    end

    # evaluate one point per symmetry class; the symmetry group fills in the rest
    SG(K2, InitFunction{4, Q}(diagram); mode)

    return nothing
end




"""
    BSE_K2!(K2::NL2_MF_K2, F0, F, FL, Π0::NL2_MF_Π, Π::NL2_MF_Π, SG, sign, ::Type{Ch}, ::Type{Sp}, is_mfRG = Val(false); mode = :serial) -> nothing

NL2 version of [`BSE_K2!`](@ref): `K̃2_r(Ω, ν; q, k)` from the mirrored
Eq. (S29b) with the internal sum over ν̃ and k̃ (see the file header), written
into `K2`. The left part `FL_012` is added at the end (in the t channel in its
density form `2·FL_t - FL_a`). With `is_mfRG = Val(true)` it computes the
linearized form used by the preconditioner, `(f - f_01̄2) π FL_012 + FL_012`
(with the crossing on the left vertex).
"""
function BSE_K2!(
    K2   :: NL2_MF_K2{Q},
    F0   :: Union{RefVertex{Q}, AbstractVertex{Q}},
    F    :: AbstractVertex{Q},
    FL   :: AbstractVertex{Q},
    Π0   :: NL2_MF_Π{Q},
    Π    :: NL2_MF_Π{Q},
    SG   :: SymmetryGroup{4, Q},
    sign :: Int,
         :: Type{Ch},
         :: Type{Sp},
    is_mfRG :: Union{Val{true}, Val{false}} = Val(false),
    ;
    mode :: Symbol = :serial
    ) where {Q, Ch <: ChannelTag, Sp <: SpinTag}

    # K̃2(Ω, ν; q, k) for one point: sum over ν̃ (called ω) and k̃ (called q)
    @inline function diagram(wtpl)

        Ω, ν, P, k = wtpl
        val     = zero(Q)
        Π0slice = view(Π0, Ω, :, P, :)
        Πslice  = view(Π , Ω, :, P, :)

        for iq in axes(Π0slice, 2)
            q = value(meshes(Π0, Val(4))[iq])

            # if is_mfRG === Val(true)
            #     F0view = fixed_momentum_view(F0, P, k,  q, Ch)
            #     FLview = fixed_momentum_view(FL, P, _crossing(P, q, Ch), k0, Ch)
            # else
            #     Fview  = fixed_momentum_view(F,  P, k,  q, Ch)
            #     F0view = fixed_momentum_view(F0, P, _crossing(P, q, Ch), k0, Ch)
            #     FLview = fixed_momentum_view(FL, P, _crossing(P, q, Ch), k0, Ch)
            # end

            for iω in axes(Π0slice, 1)
                ω = value(meshes(Π0, Val(2))[iω])

                if is_mfRG === Val(true)

                    # vertices
                    # Fl  = F0view(Ω, ν, _crossing(Ω, ω, Ch), Ch, Sp) - F0view(Ω, νInf, _crossing(Ω, ω, Ch), Ch, Sp)
                    # FLr = FLview(Ω, ω, νInf, Ch, Sp)

                    # linearized BSE: (f - f_01̄2) and the left part (crossing on the left)
                    Fl  = F0(Ω, ν, _crossing(Ω, ω, Ch), P, k, _crossing(P, q, Ch), Ch, Sp) - F0(Ω, νInf, _crossing(Ω, ω, Ch), P, k, _crossing(P, q, Ch), Ch, Sp)
                    FLr = FL(Ω, ω, νInf, P, q, k0, Ch, Sp)

                    # 1ℓ and central part
                    val += Fl * Π0slice[iω, iq] * FLr

                else

                    # vertices
                    # Fl  = Fview( Ω, ν,    ω, Ch, Sp) - Fview( Ω, νInf, ω, Ch, Sp)
                    # F0r = F0view(Ω, _crossing(Ω, ω, Ch), νInf, Ch, Sp)
                    # FLr = FLview(Ω, _crossing(Ω, ω, Ch), νInf, Ch, Sp)

                    # (F - F_01̄2) at all three momenta; f_012 and FL_012 at
                    # the crossed point (Ω - ν̃, q - k̃) in the p channel
                    Fl  = F( Ω, ν, ω, P, k, q, Ch, Sp) - F( Ω, νInf, ω, P, k, q, Ch, Sp)
                    F0r = F0(Ω, _crossing(Ω, ω, Ch), νInf, P, _crossing(P, q, Ch), k0, Ch, Sp)
                    FLr = FL(Ω, _crossing(Ω, ω, Ch), νInf, P, _crossing(P, q, Ch), k0, Ch, Sp)

                    # 1ℓ and central part
                    val += Fl * ((Πslice[iω, iq] - Π0slice[iω, iq]) * F0r + Πslice[iω, iq] * FLr)

                end
            end
        end

        # 1/β 1/N_k ∑_{ν̃,k̃}, and the channel sign
        return temperature(F) * val / numP(F) * sign
    end

    # evaluate one point per symmetry class; the symmetry group fills in the rest
    SG(K2, InitFunction{4, Q}(diagram); mode)

    # add the left part FL_012(Ω, ν; q, k); density form in the t channel
    if Ch === tCh
        mult_add!(K2, FL.γt.K2, 2)
        mult_add!(K2, FL.γa.K2, -1)
    else
        add!(K2, get_reducible_vertex(FL, Ch).K2)
    end

    return nothing
end



"""
    BSE_K2_new!(K2::NL2_MF_K2, F0, F, Π0::NL2_MF_Π, Π::NL2_MF_Π, SG, sign, ::Type{Ch}, ::Type{Sp}, is_mfRG = Val(false); mode = :serial) -> nothing

NL2 version of [`BSE_K2_new!`](@ref), the "U-closed" form

    K̃2(Ω, ν; q, k) = 1/β 1/N_k ∑_{ν̃,k̃} [(F - F_01̄2) Π - (f - f_01̄2) π](Ω, ν, ν̃; q, k, k̃) U

With `is_mfRG = Val(true)` it computes `[(F - F_01̄2) - (f - f_01̄2)] Π U`
instead.

NOTE: here the ν̃ sum runs over the K2 box only, while the local and NL
versions sum over the whole bubble mesh (the earlier loops over the bubble
mesh are commented out). For large ν̃, `F - F_01̄2` tends to the K2 class at
ν, which does not decay, so the cut drops a tail of the ν̃ sum.
"""
function BSE_K2_new!(
    K2   :: NL2_MF_K2{Q},
    F0   :: AbstractVertex{Q},
    F    :: AbstractVertex{Q},
    Π0   :: NL2_MF_Π{Q},
    Π    :: NL2_MF_Π{Q},
    SG   :: SymmetryGroup{4, Q},
    sign :: Int,
         :: Type{Ch},
         :: Type{Sp},
    is_mfRG :: Union{Val{true}, Val{false}} = Val(false),
    ;
    mode :: Symbol = :serial
    ) where {Q, Ch <: ChannelTag, Sp <: SpinTag}

    # bare vertex closing the diagram on the right
    U = bare_vertex(F, Sp)

    # K̃2(Ω, ν; q, k) for one point: sum over ν̃ (called ω) and k̃ (called q)
    @inline function diagram(wtpl)

        Ω, ν, P, k = wtpl
        val     = zero(Q)
        Π0slice = MeshFunction((meshes(Π0, Val(2)), meshes(Π0, Val(4))), view(Π0, Ω, :, P, :))
        Πslice  = MeshFunction((meshes(Π,  Val(2)), meshes(Π,  Val(4))), view(Π,  Ω, :, P, :))

        # Old: the same through fixed_momentum_view, summed over the bubble mesh
        # for iq in axes(Π0slice, 2)
        #     q = value(meshes(Π0, Val(4))[iq])
        #     Fview  = fixed_momentum_view(F,  P, k, q, Ch)
        #     F0view = fixed_momentum_view(F0, P, k, q, Ch)

        #     for iω in axes(Π0slice, 1)
        #         ω = value(meshes(Π0, Val(2))[iω])

        #         # vertices
        #         Fl  = Fview( Ω, ν, ω, Ch, Sp) - Fview( Ω, νInf, ω, Ch, Sp)
        #         F0l = F0view(Ω, ν, ω, Ch, Sp) - F0view(Ω, νInf, ω, Ch, Sp)

        #         # 1ℓ and central part
        #         if is_mfRG === Val(true)
        #             val += (Fl - F0l) * Πslice[iω, iq] * U
        #         else
        #             val += (Fl * Πslice[iω, iq] - F0l * Π0slice[iω, iq]) * U
        #         end
        #     end
        # end

        # Old: loop over the whole bubble mesh
        # for iq in eachindex(meshes(Πslice, Val(2))), iω in eachindex(meshes(Πslice, Val(1)))
        #     ω = value(meshes(Πslice, Val(1))[iω])
        #     q = value(meshes(Πslice, Val(2))[iq])

        # ν̃ on the K2 box and k̃ on the K2 momentum mesh (see the NOTE above)
        for iq in eachindex(meshes(K2, Val(4))), iω in eachindex(meshes(K2, Val(2)))
            ω = value(meshes(K2, Val(2))[iω])
            q = value(meshes(K2, Val(4))[iq])
            Π_  = Πslice[ω, q]
            Π0_ = Π0slice[ω, q]

            # ν-dependent parts of the target and reference vertex
            Fl  = F( Ω, ν, ω, P, k, q, Ch, Sp) - F( Ω, νInf, ω, P, k, q, Ch, Sp)
            F0l = F0(Ω, ν, ω, P, k, q, Ch, Sp) - F0(Ω, νInf, ω, P, k, q, Ch, Sp)

            # 1ℓ and central part
            if is_mfRG === Val(true)
                val += (Fl - F0l) * Π_ * U
            else
                val += (Fl * Π_ - F0l * Π0_) * U
            end
        end

        # 1/β 1/N_k ∑_{ν̃,k̃}, and the channel sign
        return temperature(F) * val / numP(F) * sign
    end

    # evaluate one point per symmetry class; the symmetry group fills in the rest
    SG(K2, InitFunction{4, Q}(diagram); mode)

    return nothing
end
