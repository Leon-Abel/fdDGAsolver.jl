# =============================================================================
# Precomputed vertex slices ("caches") for the K3 Bethe-Salpeter equation
# (local solver).
#
# Every term of the K3 BSE (BSEa/BSEa_K3.jl; Lihm 2025, Eq. (S29d), mirrored)
# has the form
#
#   C(Ω, ν, ν′) = 1/β ∑_ν̃ A(Ω, ν, ν̃) Π(Ω, ν̃) B(Ω, ν̃, ν′).
#
# A does not depend on ν′ and B not on ν. So both are evaluated once per sweep
# and stored, instead of being recomputed inside the loops over ν and ν′.
# Evaluating a full vertex is expensive (it sums F0 and three channels at
# converted frequencies), so this saves most of the work.
#
# Only the parts that decay at large frequency are stored: from each vertex
# its limit for the outer frequency → ∞ is subtracted. That limit contains K1
# and K2 only, which are handled by the K1 and K2 BSEs (Eqs. (S29a-c)).
#
# Caches (all on the K3 box; ν̃ = internal frequency, called ω in the code):
#   B-type, indexed (Ω, ν̃, ν′):
#     cache_F0r = f - f_012           reference vertex minus its ν′ → ∞ limit
#     cache_Γpx = Ĩ_p                 crossed spin component (for Eq. (S9a))
#   A-type, indexed (Ω, ν, ν̃):
#     cache_Γr  = Ĩ_r                 sum of the other two fd channels
#     cache_Fr  = F - F_01̄2           target vertex minus its ν → ∞ limit
# For the p channel f is stored in the crossed spin component; for the t
# channel all caches hold the density combination dSp = 2·pSp + xSp, with
# xSp_t = -pSp_a by crossing symmetry (types.jl).
#
# Because the caches live on the K3 box, the ν̃ sum of the K3 BSE is cut at the
# K3 box (while the bubble mesh is larger).
# =============================================================================

"""
    build_K3_cache!(S::ParquetSolver) -> nothing

Fill the K3 caches of the solver from the current target vertex `S.F` and
the reference vertex `S.F0` (see the file header for what each cache holds).
Called at the start of every sweep, before the K3 BSEs. Work is split over
threads and MPI ranks; `mpi_allreduce!` collects the parts.
"""
function build_K3_cache!(
    S :: ParquetSolver{Q}
    ) where {Q}

    # Zero-initialise all caches before filling them.
    set!(S.cache_Γpx, 0)
    set!(S.cache_F0p, 0)
    set!(S.cache_F0a, 0)
    set!(S.cache_F0t, 0)
    set!(S.cache_Γpp, 0)
    set!(S.cache_Γa,  0)
    set!(S.cache_Γt,  0)
    set!(S.cache_Fp,  0)
    set!(S.cache_Fa,  0)
    set!(S.cache_Ft,  0)

    # PART 1: caches indexed as (Ω, ν̃, ν′).
    # These are the vertices B(Ω, ν̃, ν′) that stand to the right of the bubble
    # in C(Ω, ν, ν′) = A(Ω, ν, ν̃) Π(Ω, ν̃) B(Ω, ν̃, ν′). They depend on ν′ but not
    # on ν.

    # Γpx: Ĩ_p of the target in the crossed spin component (xSp), for the
    # p-channel BSE in the form of Lihm 2025, Eq. (S9a).
    Threads.@threads for i in mpi_split(1 : length(S.cache_Γpx.data))
        Ω, ω, νp = value.(MatsubaraFunctions.to_meshes(S.cache_Γpx, i))

        S.cache_Γpx[i] = S.F( Ω, ω, νp, pCh, xSp; F0=false, γp=false)
        S.cache_F0p[i] = S.F0(Ω, ω, νp, pCh, xSp) - S.F0(Ω, ω, νInf, pCh, xSp)
        S.cache_F0a[i] = S.F0(Ω, ω, νp, aCh, pSp) - S.F0(Ω, ω, νInf, aCh, pSp)
        S.cache_F0t[i] = S.F0(Ω, ω, νp, tCh, pSp) - S.F0(Ω, ω, νInf, tCh, pSp)

        # Convert from pSp (parallel spin component) to dSp (density component)
        # using the relation dSp = 2 * pSp + xSp = 2 * pSp - pSp(a) (crossing symmetry)
        S.cache_F0t[i] = 2 * S.cache_F0t[i] - S.cache_F0a[i]
    end

    mpi_allreduce!(S.cache_Γpx)
    mpi_allreduce!(S.cache_F0p)
    mpi_allreduce!(S.cache_F0a)
    mpi_allreduce!(S.cache_F0t)


    # PART 2: caches indexed as (Ω, ν, ν̃).
    # These are the vertices A(Ω, ν, ν̃) that stand to the left of the bubble.

    Threads.@threads for i in mpi_split(1 : length(S.cache_Γpp.data))
        Ω, ν, ω = value.(MatsubaraFunctions.to_meshes(S.cache_Γpp, i))

        # Ĩ_r in each channel r = p, a, t: no reference part, no own channel
        S.cache_Γpp[i] = S.F(Ω, ν, ω, pCh, pSp; F0=false, γp=false)
        S.cache_Γa[i]  = S.F(Ω, ν, ω, aCh, pSp; F0=false, γa=false)
        S.cache_Γt[i]  = S.F(Ω, ν, ω, tCh, pSp; F0=false, γt=false)

        # F - F_01̄2 in each channel. Only F0 and the channel's own Φ̃_r survive
        # ν → ∞, so take (F0 + Φ̃_r)(ν) - (F0 + Φ̃_r)(∞) and add Ĩ_r from above.
        S.cache_Fp[i]  = ( S.F(Ω,    ν, ω, pCh, pSp; γa=false, γt=false)
                         - S.F(Ω, νInf, ω, pCh, pSp; γa=false, γt=false)
                         + S.cache_Γpp[i] )

        S.cache_Fa[i]  = ( S.F(Ω,    ν, ω, aCh, pSp; γp=false, γt=false)
                         - S.F(Ω, νInf, ω, aCh, pSp; γp=false, γt=false)
                         + S.cache_Γa[i] )

        S.cache_Ft[i]  = ( S.F(Ω,    ν, ω, tCh, pSp; γp=false, γa=false)
                         - S.F(Ω, νInf, ω, tCh, pSp; γp=false, γa=false)
                         + S.cache_Γt[i] )

        # Convert from pSp (parallel spin component) to dSp (density component)
        # using the relation dSp = 2 * pSp + xSp = 2 * pSp - pSp(a) (crossing symmetry)
        S.cache_Γt[i] = S.cache_Γt[i] * 2 - S.cache_Γa[i]
        S.cache_Ft[i] = S.cache_Ft[i] * 2 - S.cache_Fa[i]
    end

    mpi_allreduce!(S.cache_Γpp)
    mpi_allreduce!(S.cache_Γa)
    mpi_allreduce!(S.cache_Γt)
    mpi_allreduce!(S.cache_Fp)
    mpi_allreduce!(S.cache_Fa)
    mpi_allreduce!(S.cache_Ft)

    return nothing
end
