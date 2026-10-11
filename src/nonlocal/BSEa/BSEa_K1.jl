# =============================================================================
# Bethe-Salpeter equation for the K1 class, s-wave lattice solver NL.
#
# Lattice version of BSEa/BSEa_K1.jl (see there for the derivation). At each
# bosonic momentum q (code: P), the mirrored Eq. (S29a) reads
#
#   K̃1_r(Ω; q) = 1/β ∑_ν̃ F_01̄2(Ω, ν̃; q) [ Π̃_r(Ω, ν̃; q) f_012(Ω, ν̃; q)
#                                         + Π_r(Ω, ν̃; q) FL_012(Ω, ν̃; q) ]
#
# where all vertices are evaluated at the fermionic momenta (kSW, kSW), i.e.
# averaged over k and k′, and the bubbles are the k-averaged NL bubbles
# (s-wave approximation, Lihm 2025, Eq. (S17)). The terms F Π Ĩ and Ĩ π f do not
# reach K1, because Ĩ_r vanishes when an outer frequency goes to infinity.
#
# Paper ↔ code: q ↔ P; ν̃ (summed) ↔ ω; f, π ↔ F0, Π0.
# =============================================================================

"""
    BSE_K1!(K1::NL_MF_K1, F0, F, FL, Π0, Π, SG, sign, ::Type{Ch}, ::Type{Sp}, is_mfRG = Val(false); mode = :serial) -> nothing

Lattice (NL) version of [`BSE_K1!`](@ref): compute `K̃1_r(Ω; q)` at every
bosonic momentum q and write it into `K1` (see the file header). The vertices
are s-wave averaged and the bubbles k-averaged. With `is_mfRG = Val(true)` it
computes the linearized form used by the preconditioner, `f_01̄2 π FL_012`.
"""
function BSE_K1!(
    K1   :: NL_MF_K1{Q},
    F0   :: Union{RefVertex{Q}, AbstractVertex{Q}},
    F    :: AbstractVertex{Q},
    FL   :: AbstractVertex{Q},
    Π0   :: NL_MF_Π{Q},
    Π    :: NL_MF_Π{Q},
    SG   :: SymmetryGroup{2, Q},
    sign :: Int,
         :: Type{Ch},
         :: Type{Sp},
    is_mfRG :: Union{Val{true}, Val{false}} = Val(false),
    ;
    mode :: Symbol = :serial
    ) where {Q, Ch <: ChannelTag, Sp <: SpinTag}

    # K̃1(Ω; q) for one point: sum over the internal frequency ν̃ (called ω)
    @inline function diagram(wtpl)

        Ω, P    = wtpl
        val     = zero(Q)
        Π0slice = view(Π0, Ω, :, P)
        Πslice  = view(Π , Ω, :, P)

        for i in eachindex(Π0slice)
            ω = value(meshes(Π0, Val(2))[i])

            if is_mfRG === Val(true)
                # mfRG for the K1 class: linearized BSE, reference quantities only
                # vertices
                Fl  = F0(Ω, νInf, ω, P, kSW, kSW, Ch, Sp)
                FLr = FL(Ω, _crossing(Ω, ω, Ch), νInf, P, kSW, kSW, Ch, Sp)

                # central part
                val += Fl * Π0slice[i] * FLr

            else
                # fdDΓA for the K1 class
                # vertices: F_01̄2 (target), f_012 (reference), FL_012 (left part),
                # all averaged over the fermionic momenta
                Fl  = F( Ω, νInf, ω, P, kSW, kSW, Ch, Sp)
                F0r = F0(Ω, _crossing(Ω, ω, Ch), νInf, P, kSW, kSW, Ch, Sp)
                FLr = FL(Ω, _crossing(Ω, ω, Ch), νInf, P, kSW, kSW, Ch, Sp)

                # 1ℓ and central part
                val += Fl * ((Πslice[i] - Π0slice[i]) * F0r + Πslice[i] * FLr)
            end
        end

        # 1/β ∑_ν̃, and the channel sign
        return temperature(F) * val * sign
    end

    # evaluate one point per symmetry class; the symmetry group fills in the rest
    SG(K1, InitFunction{2, Q}(diagram); mode)

    return nothing
end



"""
    BSE_K1_new!(K1::NL_MF_K1, F0, F, Π0, Π, SG, sign, ::Type{Ch}, ::Type{Sp}, is_mfRG = Val(false); mode = :serial) -> nothing

Lattice (NL) version of [`BSE_K1_new!`](@ref), the "U-closed" form
`K̃1(Ω; q) = 1/β ∑_ν̃ [F_01̄2 Π - f_01̄2 π](Ω, ν̃; q) U`, with s-wave averaged
vertices. With `is_mfRG = Val(true)` it computes `(F - f)_01̄2 Π U` instead.
"""
function BSE_K1_new!(
    K1   :: NL_MF_K1{Q},
    F0   :: AbstractVertex{Q},
    F    :: AbstractVertex{Q},
    Π0   :: NL_MF_Π{Q},
    Π    :: NL_MF_Π{Q},
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

    # K̃1(Ω; q) for one point: sum over the internal frequency ν̃ (called ω)
    @inline function diagram(wtpl)

        Ω, P    = wtpl
        val     = zero(Q)
        Π0slice = view(Π0, Ω, :, P)
        Πslice  = view(Π , Ω, :, P)

        for i in eachindex(Π0slice)
            ω = value(meshes(Π0, Val(2))[i])

            # F_01̄2 and f_01̄2, averaged over the fermionic momenta
            Fl  = F( Ω, νInf, ω, P, kSW, kSW, Ch, Sp)
            F0l = F0(Ω, νInf, ω, P, kSW, kSW, Ch, Sp)

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
    SG(K1, InitFunction{2, Q}(diagram); mode)

    return nothing
end
