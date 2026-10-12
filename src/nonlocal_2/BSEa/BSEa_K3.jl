# =============================================================================
# Bethe-Salpeter equation for the K3 class, solver NL2.
#
# K3 is s-wave in NL2, so its BSE is the one of the NL solver
# (nonlocal/BSEa/BSEa_K3.jl; see there and BSEa/BSEa_K3.jl for the names "1ℓ",
# "right", "central" and "left" part). At each bosonic momentum q (code: P),
# the mirrored Eq. (S29d) reads
#
#   K̃3_r = (F - F_01̄2) [ Π̃_r (f - f_012) + Π_r Ĩ_r + Π_r FL ] + FL
#
# summed over the internal frequency ν̃ on the K3 box. The vertex slices come
# from the s-wave averaged caches (nonlocal_2/build_K3_cache.jl). The NL2
# bubbles are evaluated at k = kSW, i.e. averaged over the loop momentum, which
# corresponds to the k-averaged NL bubble (Lihm 2025, Eq. (S17)).
#
# Paper ↔ code: q ↔ P; ν̃ (summed) ↔ ω; ν′ ↔ νp; f, π ↔ F0, Π0; cache names:
# Γ = Ĩ_r, F0 = f - f_012, F = F - F_01̄2.
# =============================================================================

"""
    BSE_L_K3!(K3::NL_MF_K3, cache_Γ, cache_F0, Π0::NL2_MF_Π, Π::NL2_MF_Π, SG, sign, ::Type{Ch}, ::Type{Sp}, is_mfRG = Val(false); mode = :serial) -> nothing

NL2 version of [`BSE_L_K3!`](@ref): the K3 class of the left part,
`FL(Ω, ν, ν′; q) = 1/β ∑_ν̃ Ĩ_r(Ω, ν, ν̃; q) π_r(Ω, ν̃; q) (f - f_012)(Ω, ν̃, ν′; q)`,
from the caches and the k-averaged bubble; the sum runs over the K3 box.
"""
function BSE_L_K3!(
    K3       :: NL_MF_K3{Q},
    cache_Γ  :: NL_MF_K3{Q},
    cache_F0 :: NL_MF_K3{Q},
    Π0       :: NL2_MF_Π{Q},
    Π        :: NL2_MF_Π{Q},
    SG       :: SymmetryGroup{4, Q},
    sign     :: Int,
             :: Type{Ch},
             :: Type{Sp},
    is_mfRG  :: Union{Val{true}, Val{false}} = Val(false),
    ;
    mode     :: Symbol = :serial
    ) where {Q, Ch <: ChannelTag, Sp <: SpinTag}

    T = temperature(meshes(K3, Val(1)))

    # FL(Ω, ν, ν′; q) for one point: sum over ν̃ (called ω) on the K3 box
    @inline function diagram(wtpl)

        Ω, ν, νp, P = wtpl
        val     = zero(Q)
        Γslice  = view(cache_Γ,  Ω, ν,  :, P)     # Ĩ_r(Ω, ν, ν̃; q)
        F0slice = view(cache_F0, Ω, :, νp, P)     # (f - f_012)(Ω, ν̃, ν′; q)

        for i in eachindex(Γslice)
            ω = value(meshes(cache_Γ, Val(3))[i])
            # reference bubble averaged over the loop momentum
            Π0_val = Π0[Ω, ω, P, kSW]

            val += Γslice[i] * Π0_val * F0slice[i]
        end

        # 1/β ∑_ν̃, and the channel sign
        return T * val * sign
    end

    # evaluate one point per symmetry class; the symmetry group fills in the rest
    SG(K3, InitFunction{4, Q}(diagram); mode)

    return nothing
end


"""
    BSE_K3!(K3::NL_MF_K3, FL, cache_Γ, cache_F, cache_F0, Π0::NL2_MF_Π, Π::NL2_MF_Π, SG, sign1, sign2, ::Type{Ch}, ::Type{Sp}, is_mfRG = Val(false); mode = :serial) -> nothing

NL2 version of [`BSE_K3!`](@ref): `K̃3_r(Ω, ν, ν′; q)` from the mirrored
Eq. (S29d), as in the NL solver, with the bubbles averaged over the loop
momentum (see the file header). Arguments and signs as in the local method.
With `is_mfRG = Val(true)` it computes the linearized form used by the
preconditioner.
"""
function BSE_K3!(
    K3       :: NL_MF_K3{Q},
    FL       :: AbstractVertex{Q},
    cache_Γ  :: NL_MF_K3{Q},
    cache_F  :: NL_MF_K3{Q},
    cache_F0 :: NL_MF_K3{Q},
    Π0       :: NL2_MF_Π{Q},
    Π        :: NL2_MF_Π{Q},
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

        # Ĩ_r(Ω, ν̃, ν′; q) for all ν̃
        if Ch === aCh || Ch === tCh
            Γslice  = view(cache_Γ,  Ω, νp,  :, P)  # using symmetry Γ[Ω, ω, νp] = Γ[Ω, νp, ω]
        elseif Ch === pCh
            Γslice  = view(cache_Γ,  Ω, :, νp, P)
        end

        # left part of this channel (FL = Ĩ π (f - f_012))
        γL = get_reducible_vertex(FL, Ch)

        Fslice  = view(cache_F,  Ω,  ν,  :, P)     # (F - F_01̄2)(Ω, ν, ν̃; q)
        F0slice = view(cache_F0, Ω,  :, νp, P)     # (f - f_012)(Ω, ν̃, ν′; q)

        for i in eachindex(Fslice)
            ω = value(meshes(cache_F, Val(3))[i])
            # bubbles averaged over the loop momentum
            Π0_val = Π0[Ω, ω, P, kSW]
            Π_val  = Π[ Ω, ω, P, kSW]

            if is_mfRG === Val(true)
                # mfRG for the K3 class: reference bubble π only

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
                # 1ℓ and right part
                val += Fslice[i] * ((Π_val - Π0_val) * F0slice[i] + Π_val * Γslice[i]) * sign1

                # central part, with FL at the crossed frequency in the p channel
                ω_ = _crossing(Ω, ω, Ch)
                if is_inbounds(ω_, meshes(γL.K3, Val(2)))
                    if Ch === aCh || Ch === pCh
                        val += Fslice[i] * Π_val * γL.K3[Ω, ω_, νp, P] * sign2
                    elseif Ch === tCh
                        val += Fslice[i] * Π_val * (2 * FL.γt.K3[Ω, ω_, νp, P] - FL.γa.K3[Ω, ω_, νp, P]) * sign2
                    end
                end
            end
        end

        # add the left part FL(Ω, ν, ν′; q); density form in the t channel
        if Ch === aCh || Ch === pCh
            return temperature(FL) * val + γL.K3[Ω, ν, νp, P]
        else
            return temperature(FL) * val + 2 * FL.γt.K3[Ω, ν, νp, P] - FL.γa.K3[Ω, ν, νp, P]
        end

    end

    # evaluate one point per symmetry class; the symmetry group fills in the rest
    SG(K3, InitFunction{4, Q}(diagram); mode)

    return nothing
end
