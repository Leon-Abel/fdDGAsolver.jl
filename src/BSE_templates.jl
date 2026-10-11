# =============================================================================
# Channel wrappers for the Bethe-Salpeter equation (BSE) kernels.
#
# The kernels (BSEa/*.jl, nonlocal*/BSEa/*.jl) are written once for any
# channel. The wrappers in this file call them for a given solver S and
# channel r, choosing:
#   - the bubble: particle-hole (Πph, Π0ph) for a and t, particle-particle
#     (Πpp, Π0pp) for p (Lihm 2025, Eq. (S8));
#   - the symmetry group of the class (S.SGph, S.SGpp, or the *L versions for
#     the left part FL);
#   - the spin component and the overall sign:
#       a:  pSp, +1   (Lihm 2025, Eq. (S9c): Φ_a = -1/β ∑ I Π F)
#       p:  pSp, +1   (Eq. (S15), right vertex at Ω - ν̃, see _crossing)
#       t:  dSp, -1   (Eq. (S14) for the density channel D)
#     Each sign is the paper's sign times -1, because the stored bubble is
#     (iG)(iG) = -GG (bubble.jl). The K3 kernels take two signs, see below.
#
# The t channel is solved in the density component dSp, where its BSE is
# diagonal (Eq. (S14)). The kernel returns
#   Φ_t^D = 2 Φ_t^= + Φ_t^× = 2 Φ_t^= - Φ_a^=   (crossing: Φ_t^× = -Φ_a^=, Eq. (S11))
# and the wrapper stores Φ_t^= = (Φ_t^D + Φ_a^=) / 2. This needs the new a
# channel first, so the channels are always updated in the order p, a, t.
#
# Results go to S.Fbuff (new Φ̃_r) or, for the left part Ĩ_r π_r f, to S.FL.
# =============================================================================

"""
    _crossing(Ω, ω, ::Type{Ch}) -> frequency

Frequency at which the vertex on the right of the bubble is evaluated, for the
internal (summed) frequency `ω`. In the p channel it is `Ω - ω`: the code uses
the p-channel BSE in the form of Lihm 2025, Eq. (S15), whose right vertex sits
at `Ω - ν̃`. In the a and t channels it is `ω` itself.

(`boson_exchange.jl` defines an unrelated `_crossing` that maps the classes
K2 ↔ K̄2.)
"""
_crossing(Ω, ω, :: Type{pCh}) = Ω - ω
_crossing(Ω, ω, :: Type{aCh}) = ω
_crossing(Ω, ω, :: Type{tCh}) = ω


# -----------------------------------------------------------------------------
# K1 class, full BSE
# -----------------------------------------------------------------------------

"""
    BSE_K1!(S::AbstractSolver, ::Type{Ch}, is_mfRG = Val(false)) -> nothing

Compute the fd K1 class of channel `Ch` (Lihm 2025, Eq. (S29a), mirrored) and
write it to `S.Fbuff`, using the kernel `BSE_K1!(K1, F0, F, FL, Π0, Π, …)`.
The bubble, symmetry group, spin component and sign are chosen as in the file
header. For the t channel the density result is converted back to `Φ_t^=`.

With `is_mfRG = Val(true)` it computes the linearized BSE used by the
preconditioner (Eq. (S74c)).
"""
function BSE_K1!(
    S       :: AbstractSolver,
            :: Type{aCh},
    is_mfRG :: Union{Val{true}, Val{false}} = Val(false),
    )       :: Nothing
    BSE_K1!(S.Fbuff.γa.K1, S.F0, S.F, S.FL, S.Π0ph, S.Πph, S.SGph[1], +1, aCh, pSp, is_mfRG; S.mode)
end

# p channel, see the docstring above
function BSE_K1!(
    S       :: AbstractSolver,
            :: Type{pCh},
    is_mfRG :: Union{Val{true}, Val{false}} = Val(false),
    )       :: Nothing
    BSE_K1!(S.Fbuff.γp.K1, S.F0, S.F, S.FL, S.Π0pp, S.Πpp, S.SGpp[1], +1, pCh, pSp, is_mfRG; S.mode)
end

# t channel, solved in the density component, see the docstring above
function BSE_K1!(
    S       :: AbstractSolver,
            :: Type{tCh},
    is_mfRG :: Union{Val{true}, Val{false}} = Val(false),
    )       :: Nothing
    BSE_K1!(S.Fbuff.γt.K1, S.F0, S.F, S.FL, S.Π0ph, S.Πph, S.SGph[1], -1, tCh, dSp, is_mfRG; S.mode)

    # Currently S.Fbuff.γt.K1 has Φ_t^D = 2 Φ_t^= + Φ_t^× = 2 Φ_t^= - Φ_a^=
    # We want to store Φ_t^= = (Φ_t^D + Φ_a^=) / 2
    add!(S.Fbuff.γt.K1, S.Fbuff.γa.K1)
    S.Fbuff.γt.K1.data ./= 2

    return nothing
end


# -----------------------------------------------------------------------------
# K2 class, left part of the BSE
# -----------------------------------------------------------------------------

"""
    BSE_L_K2!(S::AbstractSolver, ::Type{Ch}, is_mfRG = Val(false)) -> nothing

Compute the K2 class of the left part, `Ĩ_r π_r f` with the outer frequency
ν′ → ∞, and write it to `S.FL`. This is the last term of the mirrored fd-BSE
(Lihm 2025, Eq. (5)); it is reused by the K1 and K2 BSEs of the same sweep.
Same channel choices as [`BSE_K1!`](@ref), with the symmetry groups of FL.
"""
function BSE_L_K2!(
    S       :: AbstractSolver,
            :: Type{aCh},
    is_mfRG :: Union{Val{true}, Val{false}} = Val(false),
    ) :: Nothing
    BSE_L_K2!(S.FL.γa.K2, S.F0, S.F, S.Π0ph, S.Πph, S.SGphL[2], +1, aCh, pSp, is_mfRG; S.mode)
end

# p channel, see the docstring above
function BSE_L_K2!(
    S       :: AbstractSolver,
            :: Type{pCh},
    is_mfRG :: Union{Val{true}, Val{false}} = Val(false),
    ) :: Nothing
    BSE_L_K2!(S.FL.γp.K2, S.F0, S.F, S.Π0pp, S.Πpp, S.SGppL[2], +1, pCh, pSp, is_mfRG; S.mode)
end

# t channel, solved in the density component, see the docstring above
function BSE_L_K2!(
    S       :: AbstractSolver,
            :: Type{tCh},
    is_mfRG :: Union{Val{true}, Val{false}} = Val(false),
    ) :: Nothing
    BSE_L_K2!(S.FL.γt.K2, S.F0, S.F, S.Π0ph, S.Πph, S.SGphL[2], -1, tCh, dSp, is_mfRG; S.mode)

    # Currently S.FL.γt.K2 has Φ_t^D = 2 Φ_t^= + Φ_t^× = 2 Φ_t^= - Φ_a^=
    # We want to store Φ_t^= = (Φ_t^D + Φ_a^=) / 2
    add!(S.FL.γt.K2, S.FL.γa.K2)
    S.FL.γt.K2.data ./= 2

    return nothing
end


# -----------------------------------------------------------------------------
# K2 class, full BSE
# -----------------------------------------------------------------------------

"""
    BSE_K2!(S::AbstractSolver, ::Type{Ch}, is_mfRG = Val(false)) -> nothing

Compute the fd K2 class of channel `Ch` (Lihm 2025, Eq. (S29b), mirrored) and
write it to `S.Fbuff`. Needs the left part `S.FL` from `BSE_L_K2!`. Same
channel choices as [`BSE_K1!`](@ref).
"""
function BSE_K2!(
    S       :: AbstractSolver,
            :: Type{aCh},
    is_mfRG :: Union{Val{true}, Val{false}} = Val(false),
    ) :: Nothing
    BSE_K2!(S.Fbuff.γa.K2, S.F0, S.F, S.FL, S.Π0ph, S.Πph, S.SGph[2], +1, aCh, pSp, is_mfRG; S.mode)
end

# p channel, see the docstring above
function BSE_K2!(
    S       :: AbstractSolver,
            :: Type{pCh},
    is_mfRG :: Union{Val{true}, Val{false}} = Val(false),
    ) :: Nothing
    BSE_K2!(S.Fbuff.γp.K2, S.F0, S.F, S.FL, S.Π0pp, S.Πpp, S.SGpp[2], +1, pCh, pSp, is_mfRG; S.mode)
end

# t channel, solved in the density component, see the docstring above
function BSE_K2!(
    S       :: AbstractSolver,
            :: Type{tCh},
    is_mfRG :: Union{Val{true}, Val{false}} = Val(false),
    ) :: Nothing
    BSE_K2!(S.Fbuff.γt.K2, S.F0, S.F, S.FL, S.Π0ph, S.Πph, S.SGph[2], -1, tCh, dSp, is_mfRG; S.mode)

    # Currently S.Fbuff.γt.K2 has Φ_t^D = 2 Φ_t^= + Φ_t^× = 2 Φ_t^= - Φ_a^=
    # We want to store Φ_t^= = (Φ_t^D + Φ_a^=) / 2
    add!(S.Fbuff.γt.K2, S.Fbuff.γa.K2)
    S.Fbuff.γt.K2.data ./= 2

    return nothing
end


# -----------------------------------------------------------------------------
# K3 class, left part of the BSE
# -----------------------------------------------------------------------------

"""
    BSE_L_K3!(S::AbstractSolver, ::Type{Ch}, is_mfRG = Val(false)) -> nothing

Compute the K3 class of the left part, `Ĩ_r π_r (f - f_012)`, and write it to
`S.FL`. It uses the vertex caches from `build_K3_cache!`. Signs and spins:
- a: +1, pSp.
- p: -1. The right vertex is the crossed spin component f^× (`cache_F0p`), so
  this is the p-channel BSE in the form of Lihm 2025, Eq. (S9a).
- t: -1, density component dSp, converted back to `Φ_t^=` as in the file header.
"""
function BSE_L_K3!(
    S       :: AbstractSolver,
            :: Type{aCh},
    is_mfRG :: Union{Val{true}, Val{false}} = Val(false),
    ) :: Nothing
    BSE_L_K3!(S.FL.γa.K3, S.cache_Γa, S.cache_F0a, S.Π0ph, S.Πph, S.SGphL[3], +1, aCh, pSp, is_mfRG; S.mode)
end

# p channel (crossed-spin form, Eq. (S9a)), see the docstring above
function BSE_L_K3!(
    S       :: AbstractSolver,
            :: Type{pCh},
    is_mfRG :: Union{Val{true}, Val{false}} = Val(false),
    ) :: Nothing
    BSE_L_K3!(S.FL.γp.K3, S.cache_Γpp, S.cache_F0p, S.Π0pp, S.Πpp, S.SGppL[3], -1, pCh, pSp, is_mfRG; S.mode)
end

# t channel, solved in the density component, see the docstring above
function BSE_L_K3!(
    S       :: AbstractSolver,
            :: Type{tCh},
    is_mfRG :: Union{Val{true}, Val{false}} = Val(false),
    ) :: Nothing
    BSE_L_K3!(S.FL.γt.K3, S.cache_Γt, S.cache_F0t, S.Π0ph, S.Πph, S.SGphL[3], -1, tCh, dSp, is_mfRG; S.mode)

    # Currently S.FL.γt.K3 has Φ_t^D = 2 Φ_t^= + Φ_t^× = 2 Φ_t^= - Φ_a^=
    # We want to store Φ_t^= = (Φ_t^D + Φ_a^=) / 2
    add!(S.FL.γt.K3, S.FL.γa.K3)
    S.FL.γt.K3.data ./= 2

    return nothing
end

# -----------------------------------------------------------------------------
# K3 class, full BSE
# -----------------------------------------------------------------------------

"""
    BSE_K3!(S::AbstractSolver, ::Type{Ch}, is_mfRG = Val(false)) -> nothing

Compute the fd K3 class of channel `Ch` (Lihm 2025, Eq. (S29d), mirrored; with
MBE caches Eq. (S31)) and write it to `S.Fbuff`. Needs the caches from
`build_K3_cache!` and the left part `S.FL`. The kernel takes two signs: one for
the one-loop and right parts, one for the central part.
- a: (+1, +1), pSp.
- p: (-1, +1). The one-loop and right parts use the crossed-spin caches
  (`cache_Γpx`, `cache_F0p`), i.e. the form of Eq. (S9a). The central part uses
  FL^= with the frequency crossing of Eq. (S15).
- t: (-1, -1), density component dSp, converted back to `Φ_t^=`.
"""
function BSE_K3!(
    S       :: AbstractSolver,
            :: Type{aCh},
    is_mfRG :: Union{Val{true}, Val{false}} = Val(false),
    ) :: Nothing
    BSE_K3!(S.Fbuff.γa.K3, S.FL, S.cache_Γa, S.cache_Fa, S.cache_F0a, S.Π0ph, S.Πph, S.SGph[3], +1, +1, aCh, pSp, is_mfRG; S.mode)
end

# p channel, see the docstring above
function BSE_K3!(
    S       :: AbstractSolver,
            :: Type{pCh},
    is_mfRG :: Union{Val{true}, Val{false}} = Val(false),
    ) :: Nothing
    BSE_K3!(S.Fbuff.γp.K3, S.FL, S.cache_Γpx, S.cache_Fp, S.cache_F0p, S.Π0pp, S.Πpp, S.SGpp[3], -1, +1, pCh, pSp, is_mfRG; S.mode)
end

# t channel, solved in the density component, see the docstring above
function BSE_K3!(
    S       :: AbstractSolver,
            :: Type{tCh},
    is_mfRG :: Union{Val{true}, Val{false}} = Val(false),
    ) :: Nothing
    BSE_K3!(S.Fbuff.γt.K3, S.FL, S.cache_Γt, S.cache_Ft, S.cache_F0t, S.Π0ph, S.Πph, S.SGph[3], -1, -1, tCh, dSp, is_mfRG; S.mode)

    # Currently S.Fbuff.γt.K3 has Φ_t^D = 2 Φ_t^= + Φ_t^× = 2 Φ_t^= - Φ_a^=
    # We want to store Φ_t^= = (Φ_t^D + Φ_a^=) / 2
    add!(S.Fbuff.γt.K3, S.Fbuff.γa.K3)
    S.Fbuff.γt.K3.data ./= 2

    return nothing
end




# -----------------------------------------------------------------------------
# K1 class, alternative form K1 = F_01̄2 Π U ("U-closed", used by :*_new)
# -----------------------------------------------------------------------------

"""
    BSE_K1_new!(S::AbstractSolver, ::Type{Ch}, is_mfRG = Val(false)) -> nothing

Compute the fd K1 class of channel `Ch` from the "U-closed" form
`K̃1 = (F_01̄2 Π - f_01̄2 π) U` (Lihm 2025, Eq. (S23a), mirrored), without the
left part, and write it to `S.Fbuff`. Used by the `:fdPA_new`/`:scPA_new`
strategies. Same channel choices as [`BSE_K1!`](@ref).
"""
function BSE_K1_new!(
    S       :: AbstractSolver,
            :: Type{aCh},
    is_mfRG :: Union{Val{true}, Val{false}} = Val(false),
    )       :: Nothing
    BSE_K1_new!(S.Fbuff.γa.K1, S.F0, S.F, S.Π0ph, S.Πph, S.SGph[1], +1, aCh, pSp, is_mfRG; S.mode)
end

# p channel, see the docstring above
function BSE_K1_new!(
    S       :: AbstractSolver,
            :: Type{pCh},
    is_mfRG :: Union{Val{true}, Val{false}} = Val(false),
    )       :: Nothing
    BSE_K1_new!(S.Fbuff.γp.K1, S.F0, S.F, S.Π0pp, S.Πpp, S.SGpp[1], +1, pCh, pSp, is_mfRG; S.mode)
end

# t channel, solved in the density component, see the docstring above
function BSE_K1_new!(
    S       :: AbstractSolver,
            :: Type{tCh},
    is_mfRG :: Union{Val{true}, Val{false}} = Val(false),
    )       :: Nothing
    BSE_K1_new!(S.Fbuff.γt.K1, S.F0, S.F, S.Π0ph, S.Πph, S.SGph[1], -1, tCh, dSp, is_mfRG; S.mode)

    # Currently S.Fbuff.γt.K1 has Φ_t^D = 2 Φ_t^= + Φ_t^× = 2 Φ_t^= - Φ_a^=
    # We want to store Φ_t^= = (Φ_t^D + Φ_a^=) / 2
    add!(S.Fbuff.γt.K1, S.Fbuff.γa.K1)
    S.Fbuff.γt.K1.data ./= 2

    return nothing
end


# -----------------------------------------------------------------------------
# K2 class, alternative form K2 = (F - F_01̄2) Π U ("U-closed", used by :*_new)
# -----------------------------------------------------------------------------

"""
    BSE_K2_new!(S::AbstractSolver, ::Type{Ch}, is_mfRG = Val(false)) -> nothing

Compute the fd K2 class of channel `Ch` from the "U-closed" form
`K̃2 = [(F - F_01̄2) Π - (f - f_01̄2) π] U` (Lihm 2025, Eq. (S23c), mirrored),
without the left part, and write it to `S.Fbuff`. Used by the
`:fdPA_new`/`:scPA_new` strategies. Same channel choices as
[`BSE_K1!`](@ref).
"""
function BSE_K2_new!(
    S       :: AbstractSolver,
            :: Type{aCh},
    is_mfRG :: Union{Val{true}, Val{false}} = Val(false),
    )       :: Nothing
    BSE_K2_new!(S.Fbuff.γa.K2, S.F0, S.F, S.Π0ph, S.Πph, S.SGph[2], +1, aCh, pSp, is_mfRG; S.mode)
end

# p channel, see the docstring above
function BSE_K2_new!(
    S       :: AbstractSolver,
            :: Type{pCh},
    is_mfRG :: Union{Val{true}, Val{false}} = Val(false),
    )       :: Nothing
    BSE_K2_new!(S.Fbuff.γp.K2, S.F0, S.F, S.Π0pp, S.Πpp, S.SGpp[2], +1, pCh, pSp, is_mfRG; S.mode)
end

# t channel, solved in the density component, see the docstring above
function BSE_K2_new!(
    S       :: AbstractSolver,
            :: Type{tCh},
    is_mfRG :: Union{Val{true}, Val{false}} = Val(false),
    )       :: Nothing
    BSE_K2_new!(S.Fbuff.γt.K2, S.F0, S.F, S.Π0ph, S.Πph, S.SGph[2], -1, tCh, dSp, is_mfRG; S.mode)

    # Currently S.Fbuff.γt.K2 has Φ_t^D = 2 Φ_t^= + Φ_t^× = 2 Φ_t^= - Φ_a^=
    # We want to store Φ_t^= = (Φ_t^D + Φ_a^=) / 2
    add!(S.Fbuff.γt.K2, S.Fbuff.γa.K2)
    S.Fbuff.γt.K2.data ./= 2

    return nothing
end




# -----------------------------------------------------------------------------
# K1 class, one-loop BSE (:fdPA_1loop)
# -----------------------------------------------------------------------------
# The *_1loop kernels keep only the one-loop term F Π̃ f of the fd-BSE. They
# exist only for the lattice solvers (nonlocal*/BSEa/BSE_1loop.jl), so these
# wrappers do not work for the local solver.

"""
    BSE_K1_1loop!(S::AbstractSolver, ::Type{Ch}, is_mfRG = Val(false)) -> nothing

K1 class from the one-loop term `F Π̃ f` of the fd-BSE only (strategy
`:fdPA_1loop`), written to `S.Fbuff`. Lattice solvers only. Same channel
choices as [`BSE_K1!`](@ref).
"""
function BSE_K1_1loop!(
    S       :: AbstractSolver,
            :: Type{aCh},
    is_mfRG :: Union{Val{true}, Val{false}} = Val(false),
    )       :: Nothing
    BSE_K1_1loop!(S.Fbuff.γa.K1, S.F0, S.F, S.FL, S.Π0ph, S.Πph, S.SGph[1], +1, aCh, pSp, is_mfRG; S.mode)
end

# p channel, see the docstring above
function BSE_K1_1loop!(
    S       :: AbstractSolver,
            :: Type{pCh},
    is_mfRG :: Union{Val{true}, Val{false}} = Val(false),
    )       :: Nothing
    BSE_K1_1loop!(S.Fbuff.γp.K1, S.F0, S.F, S.FL, S.Π0pp, S.Πpp, S.SGpp[1], +1, pCh, pSp, is_mfRG; S.mode)
end

# t channel, solved in the density component, see the docstring above
function BSE_K1_1loop!(
    S       :: AbstractSolver,
            :: Type{tCh},
    is_mfRG :: Union{Val{true}, Val{false}} = Val(false),
    )       :: Nothing
    BSE_K1_1loop!(S.Fbuff.γt.K1, S.F0, S.F, S.FL, S.Π0ph, S.Πph, S.SGph[1], -1, tCh, dSp, is_mfRG; S.mode)

    # Currently S.Fbuff.γt.K1 has Φ_t^D = 2 Φ_t^= + Φ_t^× = 2 Φ_t^= - Φ_a^=
    # We want to store Φ_t^= = (Φ_t^D + Φ_a^=) / 2
    add!(S.Fbuff.γt.K1, S.Fbuff.γa.K1)
    S.Fbuff.γt.K1.data ./= 2

    return nothing
end


# -----------------------------------------------------------------------------
# K2 class, one-loop BSE (:fdPA_1loop)
# -----------------------------------------------------------------------------

"""
    BSE_K2_1loop!(S::AbstractSolver, ::Type{Ch}, is_mfRG = Val(false)) -> nothing

K2 class from the one-loop term `F Π̃ f` of the fd-BSE only, written to
`S.Fbuff`. Lattice solvers only. Same channel choices as [`BSE_K1!`](@ref).
"""
function BSE_K2_1loop!(
    S       :: AbstractSolver,
            :: Type{aCh},
    is_mfRG :: Union{Val{true}, Val{false}} = Val(false),
    ) :: Nothing
    BSE_K2_1loop!(S.Fbuff.γa.K2, S.F0, S.F, S.FL, S.Π0ph, S.Πph, S.SGph[2], +1, aCh, pSp, is_mfRG; S.mode)
end

# p channel, see the docstring above
function BSE_K2_1loop!(
    S       :: AbstractSolver,
            :: Type{pCh},
    is_mfRG :: Union{Val{true}, Val{false}} = Val(false),
    ) :: Nothing
    BSE_K2_1loop!(S.Fbuff.γp.K2, S.F0, S.F, S.FL, S.Π0pp, S.Πpp, S.SGpp[2], +1, pCh, pSp, is_mfRG; S.mode)
end

# t channel, solved in the density component, see the docstring above
function BSE_K2_1loop!(
    S       :: AbstractSolver,
            :: Type{tCh},
    is_mfRG :: Union{Val{true}, Val{false}} = Val(false),
    ) :: Nothing
    BSE_K2_1loop!(S.Fbuff.γt.K2, S.F0, S.F, S.FL, S.Π0ph, S.Πph, S.SGph[2], -1, tCh, dSp, is_mfRG; S.mode)

    # Currently S.Fbuff.γt.K2 has Φ_t^D = 2 Φ_t^= + Φ_t^× = 2 Φ_t^= - Φ_a^=
    # We want to store Φ_t^= = (Φ_t^D + Φ_a^=) / 2
    add!(S.Fbuff.γt.K2, S.Fbuff.γa.K2)
    S.Fbuff.γt.K2.data ./= 2

    return nothing
end

# -----------------------------------------------------------------------------
# K3 class, one-loop BSE (:fdPA_1loop)
# -----------------------------------------------------------------------------

"""
    BSE_K3_1loop!(S::AbstractSolver, ::Type{Ch}, is_mfRG = Val(false)) -> nothing

K3 class from the one-loop term `F Π̃ f` of the fd-BSE only, written to
`S.Fbuff`. Lattice solvers only. Same channel choices and signs as
[`BSE_K3!`](@ref).
"""
function BSE_K3_1loop!(
    S       :: AbstractSolver,
            :: Type{aCh},
    is_mfRG :: Union{Val{true}, Val{false}} = Val(false),
    ) :: Nothing
    BSE_K3_1loop!(S.Fbuff.γa.K3, S.FL, S.cache_Γa, S.cache_Fa, S.cache_F0a, S.Π0ph, S.Πph, S.SGph[3], +1, +1, aCh, pSp, is_mfRG; S.mode)
end

# p channel, see the docstring above
function BSE_K3_1loop!(
    S       :: AbstractSolver,
            :: Type{pCh},
    is_mfRG :: Union{Val{true}, Val{false}} = Val(false),
    ) :: Nothing
    BSE_K3_1loop!(S.Fbuff.γp.K3, S.FL, S.cache_Γpx, S.cache_Fp, S.cache_F0p, S.Π0pp, S.Πpp, S.SGpp[3], -1, +1, pCh, pSp, is_mfRG; S.mode)
end

# t channel, solved in the density component, see the docstring above
function BSE_K3_1loop!(
    S       :: AbstractSolver,
            :: Type{tCh},
    is_mfRG :: Union{Val{true}, Val{false}} = Val(false),
    ) :: Nothing
    BSE_K3_1loop!(S.Fbuff.γt.K3, S.FL, S.cache_Γt, S.cache_Ft, S.cache_F0t, S.Π0ph, S.Πph, S.SGph[3], -1, -1, tCh, dSp, is_mfRG; S.mode)

    # Currently S.Fbuff.γt.K3 has Φ_t^D = 2 Φ_t^= + Φ_t^× = 2 Φ_t^= - Φ_a^=
    # We want to store Φ_t^= = (Φ_t^D + Φ_a^=) / 2
    add!(S.Fbuff.γt.K3, S.Fbuff.γa.K3)
    S.Fbuff.γt.K3.data ./= 2

    return nothing
end
