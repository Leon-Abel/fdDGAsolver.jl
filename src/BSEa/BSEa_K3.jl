# =============================================================================
# Bethe-Salpeter equation for the K3 class (local solver).
#
# K3_r(Ω, ν, ν′) is what remains of Φ_r after K1, K2 and K̄2 are subtracted. It
# decays in every frequency direction, so it lives on the smallest box.
#
# From the mirrored fd-BSE (Lihm 2025, Eq. (5); docs/physics_and_notation.md,
# Sec. 1.2)
#
#   Φ̃_r = F Π̃_r f + F Π_r Ĩ_r + F Π_r Ĩ_r π_r f + Ĩ_r π_r f
#
# one gets the K3 equation (Eq. (S29d), mirrored):
#
#   K̃3_r = (F - F_01̄2) [ Π̃_r (f - f_012) + Π_r Ĩ_r + Π_r FL ] + FL
#
# summed over the internal frequency ν̃, with
#   (F - F_01̄2)(Ω, ν, ν̃)     target vertex minus its ν → ∞ limit
#   (f - f_012)(Ω, ν̃, ν′)    reference vertex minus its ν′ → ∞ limit
#   Ĩ_r(Ω, ν̃, ν′)            sum of the other two fd channels
#   FL = Ĩ_r π_r (f - f_012)  K3 class of the left part (BSE_L_K3!)
# The four pieces are called "1ℓ" (F Π̃ f), "right part" (F Π Ĩ), "central
# part" (F Π FL) and "left part" (FL).
#
# Each piece has the form A(Ω, ν, ν̃) Π(Ω, ν̃) B(Ω, ν̃, ν′). The factors A and B
# are taken from caches precomputed on the K3 box (build_K3_cache.jl), so the
# sum over ν̃ only runs over the K3 box. All factors decay for large ν̃, which
# makes this truncation acceptable.
#
# With MBE vertices the caches hold f - s_r and F - S_r, and the same kernel
# gives the multiboson part M̃_r instead (Eq. (S31), boson_exchange.jl).
#
# Paper ↔ code: ω (bosonic) ↔ Ω; ν̃ (summed) ↔ ω; ν′ ↔ νp; f, π ↔ F0, Π0.
# The cache names: Γ = Ĩ_r, F0 = f - f_012, F = F - F_01̄2.
# =============================================================================

"""
    BSE_L_K3!(K3, cache_Γ, cache_F0, Π0, Π, SG, sign, ::Type{Ch}, ::Type{Sp}, is_mfRG = Val(false); mode) -> nothing

Compute the K3 class of the left part of the fd-BSE and write it into `K3`:

    FL(Ω, ν, ν′) = 1/β ∑_ν̃ Ĩ_r(Ω, ν, ν̃) π_r(Ω, ν̃) (f - f_012)(Ω, ν̃, ν′)

`cache_Γ` holds `Ĩ_r(Ω, ν, ν̃)` and `cache_F0` holds `(f - f_012)(Ω, ν̃, ν′)` on
the K3 box (`build_K3_cache.jl`); the sum over ν̃ runs over that box. Only the
reference bubble π enters. In the p channel `cache_F0` holds the crossed spin
component and the sign is -1 (the form of Lihm 2025, Eq. (S9a)), see
`BSE_templates.jl`.
"""
function BSE_L_K3!(
    K3       :: MF_K3{Q},
    cache_Γ  :: MF_K3{Q},
    cache_F0 :: MF_K3{Q},
    Π0       :: MF_Π{Q},
    Π        :: MF_Π{Q},
    SG       :: SymmetryGroup{3, Q},
    sign     :: Int,
             :: Type{Ch},
             :: Type{Sp},
    is_mfRG  :: Union{Val{true}, Val{false}} = Val(false),
    ;
    mode     :: Symbol = :serial
    ) where {Q, Ch <: ChannelTag, Sp <: SpinTag}

    T = temperature(meshes(K3, Val(1)))

    # FL(Ω, ν, ν′) for one point: sum over the internal ν̃ (called ω) on the K3 box
    @inline function diagram(wtpl)

        Ω, ν, νp = wtpl
        val     = zero(Q)
        Γslice  = view(cache_Γ,  Ω, ν,  :)     # Ĩ_r(Ω, ν, ν̃) for all ν̃
        F0slice = view(cache_F0, Ω, :, νp)     # (f - f_012)(Ω, ν̃, ν′) for all ν̃

        for i in eachindex(Γslice)
            ω = value(meshes(cache_Γ, Val(3))[i])
            Π0_val = Π0[Ω, ω]

            val += Γslice[i] * Π0_val * F0slice[i]
        end

        # 1/β ∑_ν̃, and the channel sign
        return T * val * sign
    end

    # evaluate one point per symmetry class; the symmetry group fills in the rest
    SG(K3, InitFunction{3, Q}(diagram); mode)

    return nothing
end


"""
    BSE_K3!(K3, FL, cache_Γ, cache_F, cache_F0, Π0, Π, SG, sign1, sign2, ::Type{Ch}, ::Type{Sp}, is_mfRG = Val(false); mode) -> nothing

Compute the fd K3 class `K̃3_r(Ω, ν, ν′)` of channel `Ch` and write it into
`K3` (Lihm 2025, Eq. (S29d), mirrored; see the file header):

    K̃3_r = 1/β ∑_ν̃ (F - F_01̄2) { [Π̃ (f - f_012) + Π Ĩ_r] sign1 + Π FL sign2 } + FL

- `cache_F`: `(F - F_01̄2)(Ω, ν, ν̃)`; `cache_F0`: `(f - f_012)(Ω, ν̃, ν′)`;
  `cache_Γ`: `Ĩ_r`. In the a and t channels `cache_Γ` is read as `Ĩ_r(Ω, ν′, ν̃)`,
  which equals `Ĩ_r(Ω, ν̃, ν′)` by the ν ↔ ν′ symmetry; in the p channel it is
  the crossed-spin cache `cache_Γpx(Ω, ν̃, ν′)`.
- `FL`: the left part from `BSE_L_K3!`. In the central part it is evaluated at
  the crossed frequency of the p channel (`_crossing`, Eq. (S15)).
- `sign1`, `sign2`: signs of the 1ℓ/right parts and of the central part
  (`BSE_templates.jl`).
- In the t channel, solved in the density component, FL enters in its density
  form `2·FL_t - FL_a`.

With `is_mfRG = Val(true)` it computes the K3 part of the linearized fd-BSE
used as preconditioner (Eq. (S74c)): the right and central parts with the
reference bubble π, plus FL. In that case the caches hold reference quantities
(`build_K3_cache_mfRG!`).
"""
function BSE_K3!(
    K3       :: MF_K3{Q},
    FL       :: AbstractVertex{Q},
    cache_Γ  :: MF_K3{Q},
    cache_F  :: MF_K3{Q},
    cache_F0 :: MF_K3{Q},
    Π0       :: MF_Π{Q},
    Π        :: MF_Π{Q},
    SG       :: SymmetryGroup{3, Q},
    sign1    :: Int,
    sign2    :: Int,
             :: Type{Ch},
             :: Type{Sp},
    is_mfRG  :: Union{Val{true}, Val{false}} = Val(false),
    ;
    mode     :: Symbol = :serial
    ) where {Q, Ch <: ChannelTag, Sp <: SpinTag}

    # K̃3(Ω, ν, ν′) for one point: sum over the internal ν̃ (called ω) on the K3 box
    @inline function diagram(wtpl)

        Ω, ν, νp = wtpl
        val = zero(Q)

        # Ĩ_r(Ω, ν̃, ν′) for all ν̃
        if Ch === aCh || Ch === tCh
            Γslice  = view(cache_Γ,  Ω, νp,  :)  # using symmetry Γ[Ω, ω, νp] = Γ[Ω, νp, ω]
        elseif Ch === pCh
            Γslice  = view(cache_Γ,  Ω, :, νp)
        end

        # left part of this channel (FL = Ĩ π (f - f_012))
        γL = get_reducible_vertex(FL, Ch)

        Fslice  = view(cache_F,  Ω,  ν,  :)     # (F - F_01̄2)(Ω, ν, ν̃)
        F0slice = view(cache_F0, Ω,  :, νp)     # (f - f_012)(Ω, ν̃, ν′)

        for i in eachindex(Fslice)
            ω = value(meshes(cache_F, Val(3))[i])
            Π0_val = Π0[Ω, ω]
            Π_val  = Π[ Ω, ω]

            if is_mfRG === Val(true)
                # linearized BSE (preconditioner): reference bubble π only

                # right part
                val += Fslice[i] * Π0_val * Γslice[i] * sign1

                # central part
                ω_ = _crossing(Ω, ω, Ch)
                if is_inbounds(ω_, meshes(γL.K3, Val(2)))
                    if Ch === aCh || Ch === pCh
                        val += Fslice[i] * Π0_val * γL.K3[Ω, ω_, νp] * sign2
                    elseif Ch === tCh
                        val += Fslice[i] * Π0_val * (2 * FL.γt.K3[Ω, ω_, νp] - FL.γa.K3[Ω, ω_, νp]) * sign2
                    end
                end

            else
                # fd-BSE for the K3 class
                # 1ℓ part (F - F_01̄2) Π̃ (f - f_012) and right part (F - F_01̄2) Π Ĩ
                val += Fslice[i] * ((Π_val - Π0_val) * F0slice[i] + Π_val * Γslice[i]) * sign1

                # central part (F - F_01̄2) Π FL, with FL at the crossed
                # frequency in the p channel; zero outside the K3 box of FL
                ω_ = _crossing(Ω, ω, Ch)
                if is_inbounds(ω_, meshes(γL.K3, Val(2)))
                    if Ch === aCh || Ch === pCh
                        val += Fslice[i] * Π_val * γL.K3[Ω, ω_, νp] * sign2
                    elseif Ch === tCh
                        val += Fslice[i] * Π_val * (2 * FL.γt.K3[Ω, ω_, νp] - FL.γa.K3[Ω, ω_, νp]) * sign2
                    end
                end
            end
        end

        # add the left part FL(Ω, ν, ν′) itself; in the t channel in its
        # density form 2·FL_t - FL_a
        if Ch === aCh || Ch === pCh
            return temperature(FL) * val + γL.K3[Ω, ν, νp]
        else
            return temperature(FL) * val + 2 * FL.γt.K3[Ω, ν, νp] - FL.γa.K3[Ω, ν, νp]
        end

    end

    # evaluate one point per symmetry class; the symmetry group fills in the rest
    SG(K3, InitFunction{3, Q}(diagram); mode)

    return nothing
end
