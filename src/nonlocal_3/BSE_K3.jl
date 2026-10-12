# =============================================================================
# Bethe-Salpeter equation for the K3 class, solver NL3.
#
# In NL3 the K3 class keeps all momenta, so the internal sum runs over the
# fermionic momentum k̃ as well. At each (q, k, k′) the mirrored Eq. (S29d)
# reads
#
#   K̃3_r(Ω, ν, ν′; q, k, k′) = 1/β 1/N_k ∑_{ν̃,k̃} (F - F_01̄2)(Ω, ν, ν̃; q, k, k̃)
#                                  [ Π̃_r (f - f_012) + Π_r Ĩ_r + Π_r FL ](Ω, ν̃, ν′; q, k̃, k′)
#                              + FL(Ω, ν, ν′; q, k, k′)
#
# with the bubbles at (Ω, ν̃; q, k̃) (as in NL2) and the vertex slices from the
# NL3 caches (nonlocal_3/build_K3_cache.jl). As in the local code, the central
# part takes FL at the crossed point (Ω - ν̃, q - k̃) in the p channel. The sum
# over ν̃ runs over the K3 box. See BSEa/BSEa_K3.jl for the names "1ℓ",
# "right", "central" and "left" part, and for the signs.
#
# The K1 and K2 classes of NL3 use the NL2 kernels (nonlocal_2/BSEa/).
#
# Paper ↔ code: q ↔ P; k′ ↔ kp; ν̃, k̃ (summed) ↔ ω, q; ν′ ↔ νp; f, π ↔ F0,
# Π0; cache names: Γ = Ĩ_r, F0 = f - f_012, F = F - F_01̄2.
# =============================================================================

"""
    BSE_L_K3!(K3::NL3_MF_K3, cache_Γ, cache_F0, Π0::NL2_MF_Π, Π::NL2_MF_Π, SG, sign, ::Type{Ch}, ::Type{Sp}, is_mfRG = Val(false); mode = :serial) -> nothing

NL3 version of [`BSE_L_K3!`](@ref): the K3 class of the left part,

    FL(Ω, ν, ν′; q, k, k′) = 1/β 1/N_k ∑_{ν̃,k̃} Ĩ_r(Ω, ν, ν̃; q, k, k̃) π_r(Ω, ν̃; q, k̃) (f - f_012)(Ω, ν̃, ν′; q, k̃, k′)

from the caches; the ν̃ sum runs over the K3 box.
"""
function BSE_L_K3!(
    K3       :: NL3_MF_K3{Q},
    cache_Γ  :: NL3_MF_K3{Q},
    cache_F0 :: NL3_MF_K3{Q},
    Π0       :: NL2_MF_Π{Q},
    Π        :: NL2_MF_Π{Q},
    SG       :: SymmetryGroup{6, Q},
    sign     :: Int,
             :: Type{Ch},
             :: Type{Sp},
    is_mfRG  :: Union{Val{true}, Val{false}} = Val(false),
    ;
    mode     :: Symbol = :serial
    ) where {Q, Ch <: ChannelTag, Sp <: SpinTag}

    T = temperature(meshes(K3, Val(1)))
    nK = length(meshes(K3, Val(4)))

    # FL(Ω, ν, ν′; q, k, k′) for one point: sum over ν̃ (called ω) and k̃ (called q)
    @inline function diagram(wtpl)

        Ω, ν, νp, P, k, kp = wtpl
        val     = zero(Q)
        Γslice  = view(cache_Γ,  Ω, ν,  :, P, k,  :)     # Ĩ_r(Ω, ν, ν̃; q, k, k̃)
        F0slice = view(cache_F0, Ω, :, νp, P, :, kp)     # (f - f_012)(Ω, ν̃, ν′; q, k̃, k′)

        for i in eachindex(Γslice)
            ω = value(meshes(cache_Γ, Val(3))[i.I[1]])
            q = value(meshes(cache_Γ, Val(6))[i.I[2]])
            Π0_val = Π0[Ω, ω, P, q]

            val += Γslice[i] * Π0_val * F0slice[i]
        end

        # 1/β 1/N_k ∑_{ν̃,k̃}, and the channel sign
        return T * val / nK * sign
    end

    # evaluate one point per symmetry class; the symmetry group fills in the rest
    SG(K3, InitFunction{6, Q}(diagram); mode)

    return nothing
end


"""
    BSE_K3!(K3::NL3_MF_K3, FL, cache_Γ, cache_F, cache_F0, Π0::NL2_MF_Π, Π::NL2_MF_Π, SG, sign1, sign2, ::Type{Ch}, ::Type{Sp}, is_mfRG = Val(false); mode = :serial) -> nothing

NL3 version of [`BSE_K3!`](@ref): `K̃3_r(Ω, ν, ν′; q, k, k′)` from the mirrored
Eq. (S29d) with the internal sum over ν̃ and k̃ (see the file header), written
into `K3`. Arguments and signs as in the local method. With
`is_mfRG = Val(true)` it computes the linearized form used by the
preconditioner.
"""
function BSE_K3!(
    K3       :: NL3_MF_K3{Q},
    FL       :: AbstractVertex{Q},
    cache_Γ  :: NL3_MF_K3{Q},
    cache_F  :: NL3_MF_K3{Q},
    cache_F0 :: NL3_MF_K3{Q},
    Π0       :: NL2_MF_Π{Q},
    Π        :: NL2_MF_Π{Q},
    SG       :: SymmetryGroup{6, Q},
    sign1    :: Int,
    sign2    :: Int,
             :: Type{Ch},
             :: Type{Sp},
    is_mfRG  :: Union{Val{true}, Val{false}} = Val(false),
    ;
    mode     :: Symbol = :serial
    ) where {Q, Ch <: ChannelTag, Sp <: SpinTag}

    mK = meshes(K3, Val(4))
    nK = length(mK)

    # K̃3(Ω, ν, ν′; q, k, k′) for one point: sum over ν̃ (called ω) and k̃ (called q)
    @inline function diagram(wtpl)

        Ω, ν, νp, P, k, kp = wtpl
        val = zero(Q)

        # Ĩ_r(Ω, ν̃, ν′; q, k̃, k′) for all ν̃, k̃
        if Ch === aCh || Ch === tCh
            Γslice  = view(cache_Γ, Ω, νp, :, P, kp, :)  # using symmetry Γ[Ω, ω, νp] = Γ[Ω, νp, ω]
        elseif Ch === pCh
            Γslice  = view(cache_Γ, Ω, :, νp, P, :, kp)
        end

        # left part of this channel (FL = Ĩ π (f - f_012))
        γL = get_reducible_vertex(FL, Ch)

        Fslice  = view(cache_F,  Ω, ν,  :, P, k,  :)     # (F - F_01̄2)(Ω, ν, ν̃; q, k, k̃)
        F0slice = view(cache_F0, Ω, :, νp, P, :, kp)     # (f - f_012)(Ω, ν̃, ν′; q, k̃, k′)

        for i in eachindex(Fslice)
            ω = value(meshes(cache_F, Val(3))[i.I[1]])
            q = value(meshes(cache_F, Val(6))[i.I[2]])
            Π0_val = Π0[Ω, ω, P, q]
            Π_val  = Π[ Ω, ω, P, q]

            if is_mfRG === Val(true)
                # mfRG for the K3 class: reference bubble π only

                # right part
                val += Fslice[i] * Π0_val * Γslice[i] * sign1

                # central part, with FL at the crossed point in the p channel
                ω_ = _crossing(Ω, ω, Ch)
                q_ = fold_back(_crossing(P, q, Ch), mK)
                if is_inbounds(ω_, meshes(FL.γt.K3, Val(2)))
                    if Ch === aCh || Ch === pCh
                        val += Fslice[i] * Π0_val * γL.K3[Ω, ω_, νp, P, q_, kp] * sign2
                    elseif Ch === tCh
                        val += Fslice[i] * Π0_val * (2 * FL.γt.K3[Ω, ω_, νp, P, q_, kp] - FL.γa.K3[Ω, ω_, νp, P, q_, kp]) * sign2
                    end
                end

            else
                # fdDΓA for the K3 class
                # 1ℓ and right part
                val += Fslice[i] * ((Π_val - Π0_val) * F0slice[i] + Π_val * Γslice[i]) * sign1

                # central part, with FL at the crossed point in the p channel
                ω_ = _crossing(Ω, ω, Ch)
                q_ = fold_back(_crossing(P, q, Ch), mK)
                if is_inbounds(ω_, meshes(γL.K3, Val(2)))
                    if Ch === aCh || Ch === pCh
                        val += Fslice[i] * Π_val * γL.K3[Ω, ω_, νp, P, q_, kp] * sign2
                    elseif Ch === tCh
                        val += Fslice[i] * Π_val * (2 * FL.γt.K3[Ω, ω_, νp, P, q_, kp] - FL.γa.K3[Ω, ω_, νp, P, q_, kp]) * sign2
                    end
                end
            end
        end

        # 1/β 1/N_k ∑, and the left part FL(Ω, ν, ν′; q, k, k′); density form in the t channel
        if Ch === aCh || Ch === pCh
            return temperature(FL) * val / nK + γL.K3[Ω, ν, νp, P, k, kp]
        else
            return temperature(FL) * val / nK + 2 * FL.γt.K3[Ω, ν, νp, P, k, kp] - FL.γa.K3[Ω, ν, νp, P, k, kp]
        end

    end

    # evaluate one point per symmetry class; the symmetry group fills in the rest
    SG(K3, InitFunction{6, Q}(diagram); mode)

    return nothing
end
