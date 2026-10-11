# =============================================================================
# Vertex caches for the K3 Bethe-Salpeter equation, s-wave lattice solver NL.
#
# Lattice version of build_K3_cache.jl (see there for the idea). Each term of
# the K3 BSE has the form
#
#   C(Ω, ν, ν′; q) = 1/β ∑_ν̃ A(Ω, ν, ν̃; q) Π(Ω, ν̃; q) B(Ω, ν̃, ν′; q),
#
# and the factors A and B are evaluated once per sweep and stored on the K3
# box, at every bosonic momentum q. All vertices are evaluated at the fermionic
# momenta (kSW, kSW), i.e. averaged over k and k′ (s-wave approximation, Lihm
# 2025, Eq. (S17)). Only the parts that decay at large frequency are stored.
#
# Caches (ν̃ = internal frequency, called ω in the code):
#   B-type, indexed (Ω, ν̃, ν′, q):  cache_F0r = f - f_012,  cache_Γpx = Ĩ_p (xSp)
#   A-type, indexed (Ω, ν, ν̃, q):   cache_Γr = Ĩ_r,  cache_Fr = F - F_01̄2
# For the p channel f is stored in the crossed spin component; the t-channel
# caches hold the density combination dSp = 2·pSp - pSp(a).
# =============================================================================

"""
    build_K3_cache!(S::NL_ParquetSolver) -> nothing

Fill the K3 caches of the NL solver from the current target vertex `S.F` and
the reference vertex `S.F0`, at every bosonic momentum, with s-wave averaged
vertices (see the file header).
"""
function build_K3_cache!(
    S :: NL_ParquetSolver{Q}
    ) where {Q}

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

    # B-type: vertices that stand to the right of the bubble, (Ω, ν̃, ν′, q)

    Threads.@threads for i in mpi_split(1 : length(S.cache_Γpx.data))
        Ω, ω, νp, P = value.(MatsubaraFunctions.to_meshes(S.cache_Γpx, i))

        # Ĩ_p in the crossed spin component (p-channel BSE in the form of Eq. (S9a))
        S.cache_Γpx[i] = S.F( Ω, ω, νp, P, kSW, kSW, pCh, xSp; F0=false, γp=false)

        # f - f_012 in each channel
        S.cache_F0p[i] = ( S.F0(Ω, ω,   νp, P, kSW, kSW, pCh, xSp)
                         - S.F0(Ω, ω, νInf, P, kSW, kSW, pCh, xSp) )
        S.cache_F0a[i] = ( S.F0(Ω, ω,   νp, P, kSW, kSW, aCh, pSp)
                         - S.F0(Ω, ω, νInf, P, kSW, kSW, aCh, pSp) )
        S.cache_F0t[i] = ( S.F0(Ω, ω,   νp, P, kSW, kSW, tCh, pSp)
                         - S.F0(Ω, ω, νInf, P, kSW, kSW, tCh, pSp) )

        # Convert from pSp (parallel spin component) to dSp (density component)
        # using the relation dSp = 2 * pSp + xSp = 2 * pSp - pSp(a) (crossing symmetry)
        S.cache_F0t[i] = 2 * S.cache_F0t[i] - S.cache_F0a[i]
    end

    mpi_allreduce!(S.cache_Γpx)
    mpi_allreduce!(S.cache_F0p)
    mpi_allreduce!(S.cache_F0a)
    mpi_allreduce!(S.cache_F0t)


    # A-type: vertices that stand to the left of the bubble, (Ω, ν, ν̃, q)

    Threads.@threads for i in mpi_split(1 : length(S.cache_Γpp.data))
        Ω, ν, ω, P = value.(MatsubaraFunctions.to_meshes(S.cache_Γpp, i))

        # Ĩ_r in each channel r = p, a, t: no reference part, no own channel
        S.cache_Γpp[i] = S.F(Ω, ν, ω, P, kSW, kSW, pCh, pSp; F0=false, γp=false)
        S.cache_Γa[i]  = S.F(Ω, ν, ω, P, kSW, kSW, aCh, pSp; F0=false, γa=false)
        S.cache_Γt[i]  = S.F(Ω, ν, ω, P, kSW, kSW, tCh, pSp; F0=false, γt=false)

        # F - F_01̄2 in each channel: (F0 + Φ̃_r)(ν) - (F0 + Φ̃_r)(∞), plus Ĩ_r from above
        S.cache_Fp[i]  = ( S.F(Ω,    ν, ω, P, kSW, kSW, pCh, pSp; γa=false, γt=false)
                         - S.F(Ω, νInf, ω, P, kSW, kSW, pCh, pSp; γa=false, γt=false)
                         + S.cache_Γpp[i] )
        S.cache_Fa[i]  = ( S.F(Ω,    ν, ω, P, kSW, kSW, aCh, pSp; γp=false, γt=false)
                         - S.F(Ω, νInf, ω, P, kSW, kSW, aCh, pSp; γp=false, γt=false)
                         + S.cache_Γa[i] )
        S.cache_Ft[i]  = ( S.F(Ω,    ν, ω, P, kSW, kSW, tCh, pSp; γp=false, γa=false)
                         - S.F(Ω, νInf, ω, P, kSW, kSW, tCh, pSp; γp=false, γa=false)
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



"""
    build_K3_cache_mfRG!(S::NL_ParquetSolver) -> nothing

Caches for the linearized BSE of the preconditioner (Lihm 2025, Eq. (S74c)):
the same as [`build_K3_cache!`](@ref), except that `cache_Fr` holds the
reference vertex, `f - f_01̄2`, instead of the target vertex. Not used: the
preconditioner calls the two-argument method `build_K3_cache_mfRG!(S,
`is_first_iteration`)` (nonlocal_2/build_K3_cache.jl).
"""
function build_K3_cache_mfRG!(
    S :: NL_ParquetSolver{Q}
    ) where {Q}

    # The case (3) `cache_F*` is now computed with the reference vertex, not the target vertex.

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

    # B-type: vertices that stand to the right of the bubble

    Threads.@threads for i in mpi_split(1 : length(S.cache_Γpx.data))
        Ω, ω, νp, P = value.(MatsubaraFunctions.to_meshes(S.cache_Γpx, i))

        S.cache_Γpx[i] = S.F( Ω, ω, νp, P, kSW, kSW, pCh, xSp; F0=false, γp=false)

        S.cache_F0p[i] = ( S.F0(Ω, ω,   νp, P, kSW, kSW, pCh, xSp)
                         - S.F0(Ω, ω, νInf, P, kSW, kSW, pCh, xSp) )
        S.cache_F0a[i] = ( S.F0(Ω, ω,   νp, P, kSW, kSW, aCh, pSp)
                         - S.F0(Ω, ω, νInf, P, kSW, kSW, aCh, pSp) )
        S.cache_F0t[i] = ( S.F0(Ω, ω,   νp, P, kSW, kSW, tCh, pSp)
                         - S.F0(Ω, ω, νInf, P, kSW, kSW, tCh, pSp) )

        # Convert from pSp (parallel spin component) to dSp (density component)
        # using the relation dSp = 2 * pSp + xSp = 2 * pSp - pSp(a) (crossing symmetry)
        S.cache_F0t[i] = 2 * S.cache_F0t[i] - S.cache_F0a[i]
    end

    mpi_allreduce!(S.cache_Γpx)
    mpi_allreduce!(S.cache_F0p)
    mpi_allreduce!(S.cache_F0a)
    mpi_allreduce!(S.cache_F0t)


    # A-type: vertices that stand to the left of the bubble

    Threads.@threads for i in mpi_split(1 : length(S.cache_Γpp.data))
        Ω, ν, ω, P = value.(MatsubaraFunctions.to_meshes(S.cache_Γpp, i))

        # r-irreducible vertex in each channel r = p, a, t
        S.cache_Γpp[i] = S.F(Ω, ν, ω, P, kSW, kSW, pCh, pSp; F0=false, γp=false)
        S.cache_Γa[i]  = S.F(Ω, ν, ω, P, kSW, kSW, aCh, pSp; F0=false, γa=false)
        S.cache_Γt[i]  = S.F(Ω, ν, ω, P, kSW, kSW, tCh, pSp; F0=false, γt=false)

        # f - f_01̄2 in each channel: the reference vertex instead of the target
        S.cache_Fp[i]  = ( S.F0(Ω,    ν, ω, P, kSW, kSW, pCh, pSp)
                         - S.F0(Ω, νInf, ω, P, kSW, kSW, pCh, pSp) )
        S.cache_Fa[i]  = ( S.F0(Ω,    ν, ω, P, kSW, kSW, aCh, pSp)
                         - S.F0(Ω, νInf, ω, P, kSW, kSW, aCh, pSp) )
        S.cache_Ft[i]  = ( S.F0(Ω,    ν, ω, P, kSW, kSW, tCh, pSp)
                         - S.F0(Ω, νInf, ω, P, kSW, kSW, tCh, pSp) )

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
