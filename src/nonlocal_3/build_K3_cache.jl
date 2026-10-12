# =============================================================================
# Vertex caches for the K3 Bethe-Salpeter equation, solver NL3.
#
# As in build_K3_cache.jl (see there for the idea), but K3 keeps all momenta
# in NL3, so nothing is averaged: each term of the K3 BSE has the form
#
#   C(Ω, ν, ν′; q, k, k′) = 1/β 1/N_k ∑_{ν̃,k̃} A(Ω, ν, ν̃; q, k, k̃) Π(Ω, ν̃; q, k̃) B(Ω, ν̃, ν′; q, k̃, k′),
#
# and A and B are stored on the K3 box at all three momenta. Only the parts
# that decay at large frequency are stored; the asymptotic parts are handled
# by the K1 and K2 BSEs. The ν̃ sum is therefore cut at the K3 box.
#
# Caches (ν̃, k̃ = internal frequency and momentum, called ω, q in the code):
#   B-type, (Ω, ν̃, ν′, q, k̃, k′):  cache_F0r = f - f_012,  cache_Γpx = Ĩ_p (xSp)
#   A-type, (Ω, ν, ν̃, q, k, k̃):    cache_Γr = Ĩ_r,  cache_Fr = F - F_01̄2
#                                   (linearized BSE: cache_Fr = f - f_01̄2)
# For the p channel f is stored in the crossed spin component; the t-channel
# caches hold the density combination dSp = 2·pSp - pSp(a).
# =============================================================================

"""
    build_K3_cache!(S::NL3_ParquetSolver) -> nothing

Fill the K3 caches of the NL3 solver from the current target vertex `S.F` and
the reference vertex `S.F0`, at all momenta (see the file header).
"""
function build_K3_cache!(
    S :: NL3_ParquetSolver{Q}
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

    # B-type: vertices that stand to the right of the bubble, (Ω, ν̃, ν′, q, k̃, k′)

    Threads.@threads for i in mpi_split(1 : length(S.cache_Γpx.data))
        Ω, ω, νp, P, k, q = value.(MatsubaraFunctions.to_meshes(S.cache_Γpx, i))

        # Ĩ_p in the crossed spin component (p-channel BSE in the form of Eq. (S9a))
        S.cache_Γpx[i] = S.F( Ω, ω, νp, P, k, q, pCh, xSp; F0=false, γp=false)

        # f - f_012 in each channel

        S.cache_F0p[i] = ( S.F0(Ω, ω,   νp, P, k, q, pCh, xSp)
                         - S.F0(Ω, ω, νInf, P, k, q, pCh, xSp) )
        S.cache_F0a[i] = ( S.F0(Ω, ω,   νp, P, k, q, aCh, pSp)
                         - S.F0(Ω, ω, νInf, P, k, q, aCh, pSp) )
        S.cache_F0t[i] = ( S.F0(Ω, ω,   νp, P, k, q, tCh, pSp)
                         - S.F0(Ω, ω, νInf, P, k, q, tCh, pSp) )

        # Convert from pSp (parallel spin component) to dSp (density component)
        # using the relation dSp = 2 * pSp + xSp = 2 * pSp - pSp(a) (crossing symmetry)
        S.cache_F0t[i] = 2 * S.cache_F0t[i] - S.cache_F0a[i]
    end

    mpi_allreduce!(S.cache_Γpx)
    mpi_allreduce!(S.cache_F0p)
    mpi_allreduce!(S.cache_F0a)
    mpi_allreduce!(S.cache_F0t)


    # A-type: vertices that stand to the left of the bubble, (Ω, ν, ν̃, q, k, k̃)

    Threads.@threads for i in mpi_split(1 : length(S.cache_Γpp.data))
        Ω, ν, ω, P, k, q = value.(MatsubaraFunctions.to_meshes(S.cache_Γpp, i))

        # Ĩ_r in each channel r = p, a, t: no reference part, no own channel
        S.cache_Γpp[i] = S.F(Ω, ν, ω, P, k, q, pCh, pSp; F0=false, γp=false)
        S.cache_Γa[i]  = S.F(Ω, ν, ω, P, k, q, aCh, pSp; F0=false, γa=false)
        S.cache_Γt[i]  = S.F(Ω, ν, ω, P, k, q, tCh, pSp; F0=false, γt=false)

        # F - F_01̄2 in each channel: (F0 + Φ̃_r)(ν) - (F0 + Φ̃_r)(∞), plus Ĩ_r from above
        S.cache_Fp[i]  = ( S.F(Ω,    ν, ω, P, k, q, pCh, pSp; γa=false, γt=false)
                         - S.F(Ω, νInf, ω, P, k, q, pCh, pSp; γa=false, γt=false)
                         + S.cache_Γpp[i] )
        S.cache_Fa[i]  = ( S.F(Ω,    ν, ω, P, k, q, aCh, pSp; γp=false, γt=false)
                         - S.F(Ω, νInf, ω, P, k, q, aCh, pSp; γp=false, γt=false)
                         + S.cache_Γa[i] )
        S.cache_Ft[i]  = ( S.F(Ω,    ν, ω, P, k, q, tCh, pSp; γp=false, γa=false)
                         - S.F(Ω, νInf, ω, P, k, q, tCh, pSp; γp=false, γa=false)
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
    build_K3_cache_mfRG!(S::NL3_ParquetSolver, is_first_iteration) -> nothing

Caches for the linearized BSE of the preconditioner (Lihm 2025, Eq. (S74c)),
as for NL and NL2 (`nonlocal_2/build_K3_cache.jl`), at all momenta:
`cache_Γpx`, `cache_Γr` = `Ĩ_r` from the current target vertex on every call,
and `cache_Fr` = `f - f_01̄2` only when `is_first_iteration = true`. Unlike the
NL/NL2 method, `Ĩ_r` is computed directly with `F0 = false`.

NOTE: the ν = ∞ term of `cache_Fr` is evaluated at averaged fermionic momenta
(kSW), while [`build_K3_cache!`](@ref) uses the actual (k, k̃). The two agree
when the reference f does not depend on the fermionic momenta (e.g. the local
DMFT vertex).
"""
function build_K3_cache_mfRG!(
    S :: NL3_ParquetSolver{Q},
    is_first_iteration :: Bool,
    ) where {Q}

    # If `is_first_iteration = true`, compute F0 and Γ.
    # If `is_first_iteration = false`, compute Γ.

    # The `cache_F*` caches hold the reference vertex here, not the target vertex.

    # B-type: Ĩ_p in the crossed spin component
    @inline function diagram_Γpx(wtpl)
        Ω, ν, ω, P, k, q = wtpl
        return S.F(Ω, ν, ω, P, k, q, pCh, xSp; F0=false, γp=false)
    end
    S.SGpp[3](S.cache_Γpx, InitFunction{6, Q}(diagram_Γpx); mode = S.mode)


    # A-type: Ĩ_r in each channel
    @inline function diagram_Γpp(wtpl)
        Ω, ν, ω, P, k, q = wtpl
        return S.F(Ω, ν, ω, P, k, q, pCh, pSp; F0=false, γp=false)
    end
    @inline function diagram_Γa(wtpl)
        Ω, ν, ω, P, k, q = wtpl
        return S.F(Ω, ν, ω, P, k, q, aCh, pSp; F0=false, γa=false)
    end
    @inline function diagram_Γt(wtpl)
        Ω, ν, ω, P, k, q = wtpl
        return S.F(Ω, ν, ω, P, k, q, tCh, pSp; F0=false, γt=false)
    end

    S.SGpp[3](S.cache_Γpp, InitFunction{6, Q}(diagram_Γpp); mode = S.mode)
    S.SGph[3](S.cache_Γa,  InitFunction{6, Q}(diagram_Γa); mode = S.mode)
    S.SGph[3](S.cache_Γt,  InitFunction{6, Q}(diagram_Γt); mode = S.mode)

    # Convert from pSp (parallel spin component) to dSp (density component)
    # using the relation dSp = 2 * pSp + xSp = 2 * pSp - pSp(a) (crossing symmetry)
    @. S.cache_Γt.data = S.cache_Γt.data * 2 - S.cache_Γa.data


    if is_first_iteration
        # Reference vertex f - f_01̄2. Needs to be cached only in the first mfRG
        # iteration, because f does not change. (The ν = ∞ term at kSW, see the
        # NOTE in the docstring.)

        @inline function diagram_F0p(wtpl)
            Ω, ν, ω, P, k, q = wtpl
            return S.F0(Ω, ν, ω, P, k, q, pCh, pSp) - S.F0(Ω, νInf, ω, P, kSW, kSW, pCh, pSp)
        end
        @inline function diagram_F0a(wtpl)
            Ω, ν, ω, P, k, q = wtpl
            return S.F0(Ω, ν, ω, P, k, q, aCh, pSp) - S.F0(Ω, νInf, ω, P, kSW, kSW, aCh, pSp)
        end
        @inline function diagram_F0t(wtpl)
            Ω, ν, ω, P, k, q = wtpl
            return S.F0(Ω, ν, ω, P, k, q, tCh, pSp) - S.F0(Ω, νInf, ω, P, kSW, kSW, tCh, pSp)
        end

        S.SGpp[3](S.cache_Fp, InitFunction{6, Q}(diagram_F0p); mode = S.mode)
        S.SGph[3](S.cache_Fa, InitFunction{6, Q}(diagram_F0a); mode = S.mode)
        S.SGph[3](S.cache_Ft, InitFunction{6, Q}(diagram_F0t); mode = S.mode)

        # Convert from pSp (parallel spin component) to dSp (density component)
        # using the relation dSp = 2 * pSp + xSp = 2 * pSp - pSp(a) (crossing symmetry)
        @. S.cache_Ft.data = S.cache_Ft.data * 2 - S.cache_Fa.data
    end

    return nothing
end
