# =============================================================================
# Bethe-Salpeter equation for the K1 class, solver NL2.
#
# As in the NL solver (nonlocal/BSEa/BSEa_K1.jl), but the bubbles and the K2
# parts of the vertices keep the internal fermionic momentum k̃, which is summed
# together with ν̃. At each bosonic momentum q (code: P), the mirrored
# Eq. (S29a) reads
#
#   K̃1_r(Ω; q) = 1/β 1/N_k ∑_{ν̃,k̃} F_01̄2 [ Π̃_r f_012 + Π_r FL_012 ]
#
# with everything at (Ω, ν̃; q, k̃). F_01̄2 (left outer leg at ν = ∞) depends on
# k̃ only through its K̄2, and f_012 and FL_012 (right outer leg at ν′ = ∞) only
# through their K2. The momentum of a leg at infinity does not matter, so the
# code passes the dummy k0 there. In the p channel the right vertex sits at the
# crossed point (Ω - ν̃, q - k̃) (_crossing, BSE_templates.jl). The terms F Π Ĩ
# and Ĩ π f do not reach K1, because Ĩ_r vanishes when an outer frequency goes
# to infinity.
#
# Paper ↔ code: q ↔ P; ν̃, k̃ (summed) ↔ ω, q; f, π ↔ F0, Π0.
# =============================================================================

"""
    BSE_K1!(K1::NL_MF_K1, F0, F, FL, Π0::NL2_MF_Π, Π::NL2_MF_Π, SG, sign, ::Type{Ch}, ::Type{Sp}, is_mfRG = Val(false); mode = :serial) -> nothing

NL2 version of [`BSE_K1!`](@ref): compute `K̃1_r(Ω; q)` with the internal sum
over ν̃ and k̃ (see the file header) and write it into `K1`. With
`is_mfRG = Val(true)` it computes the linearized form used by the
preconditioner, `f_01̄2 π FL_012`.
"""
function BSE_K1!(
    K1   :: NL_MF_K1{Q},
    F0   :: Union{RefVertex{Q}, AbstractVertex{Q}},
    F    :: AbstractVertex{Q},
    FL   :: AbstractVertex{Q},
    Π0   :: NL2_MF_Π{Q},
    Π    :: NL2_MF_Π{Q},
    SG   :: SymmetryGroup{2, Q},
    sign :: Int,
         :: Type{Ch},
         :: Type{Sp},
    is_mfRG :: Union{Val{true}, Val{false}} = Val(false),
    ;
    mode :: Symbol = :serial
    ) where {Q, Ch <: ChannelTag, Sp <: SpinTag}

    # K̃1(Ω; q) for one point: sum over ν̃ (called ω) and k̃ (called q)
    @inline function diagram(wtpl)

        Ω, P    = wtpl
        val     = zero(Q)
        Π0slice = view(Π0, Ω, :, P, :)
        Πslice  = view(Π , Ω, :, P, :)

        for i in eachindex(Π0slice)
            ω = value(meshes(Π0, Val(2))[i.I[1]])
            q = value(meshes(Π0, Val(4))[i.I[2]])

            if is_mfRG === Val(true)
                # mfRG for the K1 class: linearized BSE, reference quantities only
                # vertices
                Fl  = F0(Ω, νInf, ω, P, k0, q, Ch, Sp)
                FLr = FL(Ω, _crossing(Ω, ω, Ch), νInf, P, _crossing(P, q, Ch), k0, Ch, Sp)

                # central part
                val += Fl * Π0slice[i] * FLr

            else
                # fdDΓA for the K1 class
                # vertices: F_01̄2 (target), f_012 (reference), FL_012 (left part)
                Fl  = F( Ω, νInf, ω, P, k0,  q, Ch, Sp)
                F0r = F0(Ω, _crossing(Ω, ω, Ch), νInf, P, _crossing(P, q, Ch), k0, Ch, Sp)
                FLr = FL(Ω, _crossing(Ω, ω, Ch), νInf, P, _crossing(P, q, Ch), k0, Ch, Sp)

                # 1ℓ and central part
                val += Fl * ((Πslice[i] - Π0slice[i]) * F0r + Πslice[i] * FLr)
            end
        end

        # 1/β 1/N_k ∑_{ν̃,k̃}, and the channel sign
        return temperature(F) * val / numP(F) * sign
    end

    # evaluate one point per symmetry class; the symmetry group fills in the rest
    SG(K1, InitFunction{2, Q}(diagram); mode)

    return nothing
end



"""
    BSE_K1_new!(K1::NL_MF_K1, F0, F, Π0::NL2_MF_Π, Π::NL2_MF_Π, SG, sign, ::Type{Ch}, ::Type{Sp}, is_mfRG = Val(false); mode = :serial) -> nothing

NL2 version of [`BSE_K1_new!`](@ref), the "U-closed" form
`K̃1(Ω; q) = 1/β 1/N_k ∑_{ν̃,k̃} [F_01̄2 Π - f_01̄2 π](Ω, ν̃; q, k̃) U`. With
`is_mfRG = Val(true)` it computes `(F - f)_01̄2 Π U` instead.
"""
function BSE_K1_new!(
    K1   :: NL_MF_K1{Q},
    F0   :: AbstractVertex{Q},
    F    :: AbstractVertex{Q},
    Π0   :: NL2_MF_Π{Q},
    Π    :: NL2_MF_Π{Q},
    SG   :: SymmetryGroup{2, Q},
    sign :: Int,
         :: Type{Ch},
         :: Type{Sp},
    is_mfRG :: Union{Val{true}, Val{false}} = Val(false),
    ;
    mode :: Symbol = :serial
    ) where {Q, Ch <: ChannelTag, Sp <: SpinTag}


    # bare vertex closing the diagram on the right
    U = bare_vertex(F, Sp)

    # K̃1(Ω; q) for one point: sum over ν̃ (called ω) and k̃ (called q)
    @inline function diagram(wtpl)

        Ω, P    = wtpl
        val     = zero(Q)
        Π0slice = MeshFunction((meshes(Π0, Val(2)), meshes(Π0, Val(4))), view(Π0, Ω, :, P, :))
        Πslice  = MeshFunction((meshes(Π,  Val(2)), meshes(Π,  Val(4))), view(Π,  Ω, :, P, :))

        for iq in eachindex(meshes(Πslice, Val(2))), iω in eachindex(meshes(Πslice, Val(1)))
            ω = value(meshes(Πslice, Val(1))[iω])
            q = value(meshes(Πslice, Val(2))[iq])
            Π_  = Πslice[iω, iq]
            Π0_ = Π0slice[iω, iq]

            # vertices: F_01̄2 and f_01̄2
            Fl  = F( Ω, νInf, ω, P, k0, q, Ch, Sp)
            F0l = F0(Ω, νInf, ω, P, k0, q, Ch, Sp)

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
    SG(K1, InitFunction{2, Q}(diagram); mode)

    return nothing
end
