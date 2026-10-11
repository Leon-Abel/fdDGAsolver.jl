# =============================================================================
# One-loop Bethe-Salpeter kernels, s-wave lattice solver NL (strategy
# :fdPA_1loop).
#
# The one-loop approximation keeps only the first term of the mirrored fd-BSE
# (Lihm 2025, Eq. (5)): Φ̃_r ≈ F Π̃_r f. Per class this is
#
#   K̃1 = F_01̄2 Π̃ f_012,   K̃2 = (F - F_01̄2) Π̃ f_012,   K̃3 = (F - F_01̄2) Π̃ (f - f_012),
#
# at each bosonic momentum q, with s-wave averaged vertices and k-averaged
# bubbles. With is_mfRG = Val(true) the kernels compute the same linearized
# terms as the full kernels, because the preconditioner (mfRG.jl) needs them.
#
# NOTE: BSE_K2_1loop! also adds the left part FL.K2, but the :fdPA_1loop sweep
# does not recompute FL, so FL holds whatever was last stored there (e.g. by
# the preconditioner). BSE_K3_1loop! leaves the left part out (commented out).
# =============================================================================

"""
    BSE_K1_1loop!(K1::NL_MF_K1, F0, F, FL, Π0, Π, SG, sign, ::Type{Ch}, ::Type{Sp}, is_mfRG = Val(false); mode = :serial) -> nothing

K1 class from the one-loop term only, `K̃1(Ω; q) = 1/β ∑_ν̃ F_01̄2 Π̃ f_012`
(see the file header). With `is_mfRG = Val(true)`: `f_01̄2 π FL_012`, as in
[`BSE_K1!`](@ref).
"""
function BSE_K1_1loop!(
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
                # mfRG for the K1 class
                # vertices
                Fl  = F0(Ω, νInf, ω, P, kSW, kSW, Ch, Sp)
                FLr = FL(Ω, _crossing(Ω, ω, Ch), νInf, P, kSW, kSW, Ch, Sp)

                # central part
                val += Fl * Π0slice[i] * FLr

            else
                # fdDΓA for the K1 class, one-loop term only
                # vertices
                Fl  = F( Ω, νInf, ω, P, kSW, kSW, Ch, Sp)
                F0r = F0(Ω, _crossing(Ω, ω, Ch), νInf, P, kSW, kSW, Ch, Sp)

                # 1ℓ part
                val += Fl * (Πslice[i] - Π0slice[i]) * F0r
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
    BSE_K2_1loop!(K2::NL_MF_K2, F0, F, FL, Π0, Π, SG, sign, ::Type{Ch}, ::Type{Sp}, is_mfRG = Val(false); mode = :serial) -> nothing

K2 class from the one-loop term, `K̃2(Ω, ν; q) = 1/β ∑_ν̃ (F - F_01̄2) Π̃ f_012`,
plus the left part `FL.K2` (see the NOTE in the file header). With
`is_mfRG = Val(true)` it computes the linearized form as in [`BSE_K2!`](@ref).
"""
function BSE_K2_1loop!(
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

                Fl  = F0(Ω, ν, _crossing(Ω, ω, Ch), P, kSW, kSW, Ch, Sp) - F0(Ω, νInf, _crossing(Ω, ω, Ch), P, kSW, kSW, Ch, Sp)
                FLr = FL(Ω, ω, νInf, P, kSW, kSW, Ch, Sp)

                # 1ℓ and central part
                val += Fl * Π0slice[iω] * FLr

            else

                Fl  = F( Ω, ν, ω, P, kSW, kSW, Ch, Sp) - F( Ω, νInf, ω, P, kSW, kSW, Ch, Sp)
                F0r = F0(Ω, _crossing(Ω, ω, Ch), νInf, P, kSW, kSW, Ch, Sp)

                # 1ℓ part
                val += Fl * (Πslice[iω] - Π0slice[iω]) * F0r

            end

        end

        # 1/β ∑_ν̃, and the channel sign
        return temperature(F) * val * sign
    end

    # evaluate one point per symmetry class; the symmetry group fills in the rest
    SG(K2, InitFunction{3, Q}(diagram); mode)

    # add the left part FL.K2 (see the NOTE in the file header)
    if Ch === tCh
        mult_add!(K2, FL.γt.K2, 2)
        mult_add!(K2, FL.γa.K2, -1)
    else
        add!(K2, get_reducible_vertex(FL, Ch).K2)
    end

    return nothing
end


"""
    BSE_K3_1loop!(K3::NL_MF_K3, FL, cache_Γ, cache_F, cache_F0, Π0, Π, SG, sign1, sign2, ::Type{Ch}, ::Type{Sp}, is_mfRG = Val(false); mode = :serial) -> nothing

K3 class from the one-loop term only, `K̃3 = 1/β ∑_ν̃ (F - F_01̄2) Π̃ (f - f_012)`,
from the caches (the left part is not added). With `is_mfRG = Val(true)` it
computes the right and central parts of the linearized form, as in
[`BSE_K3!`](@ref).
"""
function BSE_K3_1loop!(
    K3       :: NL_MF_K3{Q},
    FL       :: AbstractVertex{Q},
    cache_Γ  :: NL_MF_K3{Q},
    cache_F  :: NL_MF_K3{Q},
    cache_F0 :: NL_MF_K3{Q},
    Π0       :: NL_MF_Π{Q},
    Π        :: NL_MF_Π{Q},
    SG       :: SymmetryGroup{4, Q},
    sign1    :: Int,
    sign2    :: Int,
             :: Type{Ch},
             :: Type{Sp},
    is_mfRG  :: Union{Val{true}, Val{false}} = Val(false),
    ;
    mode     :: Symbol = :serial
    ) where {Q, Ch <: ChannelTag, Sp <: SpinTag}

    # K̃3(Ω, ν, ν′; q) for one point: sum over ν̃ (called ω) on the K3 box
    @inline function diagram(wtpl)

        Ω, ν, νp, P = wtpl
        val = zero(Q)

        if Ch === aCh || Ch === tCh
            Γslice  = view(cache_Γ,  Ω, νp,  :, P)  # using symmetry Γ[Ω, ω, νp] = Γ[Ω, νp, ω]
        elseif Ch === pCh
            Γslice  = view(cache_Γ,  Ω, :, νp, P)
        end

        γL = get_reducible_vertex(FL, Ch)

        Fslice  = view(cache_F,  Ω,  ν,  :, P)
        F0slice = view(cache_F0, Ω,  :, νp, P)

        for i in eachindex(Fslice)
            ω = value(meshes(cache_F, Val(3))[i])
            Π0_val = Π0[Ω, ω, P]
            Π_val  = Π[ Ω, ω, P]

            if is_mfRG === Val(true)
                # mfRG for the K3 class

                # right part
                val += Fslice[i] * Π0_val * Γslice[i] * sign1

                # central part
                ω_ = _crossing(Ω, ω, Ch)
                if is_inbounds(ω_, meshes(γL.K3, Val(2)))
                    if Ch === aCh || Ch === pCh
                        val += Fslice[i] * Π0_val * γL.K3[Ω, ω_, νp, P] * sign2
                    elseif Ch === tCh
                        val += Fslice[i] * Π0_val * (2 * FL.γt.K3[Ω, ω_, νp, P] - FL.γa.K3[Ω, ω_, νp, P]) * sign2
                    end
                end

            else
                # fdDΓA for the K3 class
                # 1ℓ term
                val += Fslice[i] * (Π_val - Π0_val) * F0slice[i] * sign1
            end
        end

        # (left part not added)
        if Ch === aCh || Ch === pCh
            return temperature(FL) * val# + γL.K3[Ω, ν, νp, P]
        else
            return temperature(FL) * val# + 2 * FL.γt.K3[Ω, ν, νp, P] - FL.γa.K3[Ω, ν, νp, P]
        end

    end

    # evaluate one point per symmetry class; the symmetry group fills in the rest
    SG(K3, InitFunction{4, Q}(diagram); mode)

    return nothing
end
