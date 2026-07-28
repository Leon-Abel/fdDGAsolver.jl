# Test suite for the finite-difference Parquet Approximation (fdPA) applied to the
# Single Impurity Anderson Model (SIAM).
#
# The fdPA solves the parquet equations for the vertex ΔF = F - F0, where F0
# is the vertex of a solved reference system and F is the vertex of the target system.
# Convergence is checked by verifying that S_fd.F + S_fd.F0 ≈ S.F, i.e. the fdPA
# increment reconstructs the full scPA vertex of the target system.

using fdDGAsolver
using MatsubaraFunctions
using HDF5
using Test

@testset "SIAM fdPA" begin
    using MPI
    MPI.Init()

    # ---------------------------------------------------------------------------------------------------
    # Physical parameters
    # ---------------------------------------------------------------------------------------------------
    T = 0.1     # temperature
    U = 1.0     # on-site Coulomb interaction
    D = 10.0    # half-bandwidth
    e = -0.3    # Impurity on-site energy ε_d plus U/2, Σ_H computed as (n_σ - 1/2) * U
    Δ = π / 3   # hybridization strength

    # ---------------------------------------------------------------------------------------------------
    # Frequency grid sizes
    # ---------------------------------------------------------------------------------------------------
    nmax = 12
    nG   = 48nmax         # number of positive fermionic frequencies for the Green's function
    nK1  = 8nmax          # number of positive bosonic frequencies for K1
    nK2  = (nmax, nmax)  # (bosonic, fermionic) grid sizes for K2
    nK3  = (nmax, nmax)  # (bosonic, fermionic) grid sizes for K3
    # NOTE: nG must be ≥ nK1 * mΠν_factor (default 6). The bubble mesh has nK1*mΠν_factor
    # fermionic points; for frequencies outside the G mesh, bubbles! falls back to the 1/iν
    # tail, which is non-zero even when G0 = 0. This causes FL to be non-zero which in 
    # turn means, that the fd-equations do not simplify to the sc equations.
    # With nG ≥ nK1*mΠν_factor all bubble frequencies fall inside the G mesh, 
    # so Π0 = 0 exactly and the trivial fdPA reproduces the scPA result.

    # ---------------------------------------------------------------------------------------------------
    # Reference point: scPA solution at the reference parameters
    # ---------------------------------------------------------------------------------------------------
    S0 = parquet_solver_siam_parquet_approximation(nG, nK1, nK2, nK3; e, T, U, Δ, D)
    init_sym_grp!(S0)
    res = fdDGAsolver.solve!(S0; strategy = :scPA, parallel_mode = :threads, verbose = false);

    # ---------------------------------------------------------------------------------------------------
    # Trivial fdPA: set reference to zero
    # ---------------------------------------------------------------------------------------------------
    S0_fd = parquet_solver_siam_parquet_approximation(nG, nK1, nK2, nK3; e, T, U, Δ, D)
    init_sym_grp!(S0_fd)
    res = fdDGAsolver.solve!(S0_fd; strategy = :fdPA, parallel_mode = :threads, verbose = false);

    # fdPA increment is zero, so scPA and fdPA results must agree
    @test absmax(S0.Σ - S0_fd.Σ) ≈ 0
    @test maximum(abs.(flatten(S0.F) .- flatten(S0_fd.F))) ≈ 0

    # ---------------------------------------------------------------------------------------------------
    # Target point: scPA solution at different physical parameters
    # ---------------------------------------------------------------------------------------------------
    # These serve as the ground-truth reference to validate the fdPA result against.
    Δ_fd = π / 5   # new hybridization strength
    e_fd = 0.5     # new impurity level position
    D_fd = 20.0    # new half-bandwidth

    S = parquet_solver_siam_parquet_approximation(nG, nK1, nK2, nK3; e = e_fd, T, U, Δ = Δ_fd, D = D_fd)
    init_sym_grp!(S)
    res = fdDGAsolver.solve!(S; strategy = :scPA, parallel_mode = :threads, verbose = false);

    # ---------------------------------------------------------------------------------------------------
    # Non-trivial fdPA: reference → target, matched grid sizes
    # ---------------------------------------------------------------------------------------------------
    # Build the fdPA solver from the reference solution S0, using the bare Green's function
    # of the target system. The fdPA iterates to find the incremental correction ΔF = F - F0.
    Gbare = fdDGAsolver.siam_bare_Green(meshes(S0.G, Val(1)); e = e_fd, Δ = Δ_fd, D = D_fd)

    S_fd = ParquetSolver(nK1, nK2, nK3, Gbare, S0.G, S0.Σ, S0.F)
    init_sym_grp!(S_fd)
    res = fdDGAsolver.solve!(S_fd; strategy = :fdPA, parallel_mode = :threads, verbose = false);

    # self-energy: S_fd.Σ should recover S.Σ
    @test absmax(S_fd.Σ - S.Σ) < 3e-5

    # vertex: S_fd.F + S_fd.F0 should recover S.F for each channel and asymptotic class
    @test absmax(S_fd.F.γp.K1 + S_fd.F0.γp.K1 - S.F.γp.K1) < 5e-4
    @test absmax(S_fd.F.γa.K1 + S_fd.F0.γa.K1 - S.F.γa.K1) < 5e-4
    @test absmax(S_fd.F.γt.K1 + S_fd.F0.γt.K1 - S.F.γt.K1) < 5e-4
    @test absmax(S_fd.F.γp.K2 + S_fd.F0.γp.K2 - S.F.γp.K2) < 1e-3
    @test absmax(S_fd.F.γa.K2 + S_fd.F0.γa.K2 - S.F.γa.K2) < 1e-3
    @test absmax(S_fd.F.γt.K2 + S_fd.F0.γt.K2 - S.F.γt.K2) < 1e-3
    @test absmax(S_fd.F.γp.K3 + S_fd.F0.γp.K3 - S.F.γp.K3) < 1e-3
    @test absmax(S_fd.F.γa.K3 + S_fd.F0.γa.K3 - S.F.γa.K3) < 1e-3
    @test absmax(S_fd.F.γt.K3 + S_fd.F0.γt.K3 - S.F.γt.K3) < 1e-3

    # same check iterated over all (channel, class) combinations
    for ch in [:γa, :γp, :γt], class in [:K1, :K2, :K3]
        @test absmax(getproperty(getproperty(S_fd.F, ch), class)
                   + getproperty(getproperty(S_fd.F0, ch), class)
                   - getproperty(getproperty(S.F, ch), class)) < 2e-3
    end

    # ---------------------------------------------------------------------------------------------------
    # Non-trivial fdPA: reference → target, mismatched (coarser) grid sizes
    # ---------------------------------------------------------------------------------------------------
    # The fdPA is re-run with a smaller vertex grid to verify that the solver handles
    # mismatched grid sizes between the reference (S0) and the target (S_fd2) correctly.
    # Coarser grids introduce truncation errors, so tolerances are relaxed relative to above.
    nmax = 8
    nK1 = 8nmax
    nK2 = (nmax, nmax)
    nK3 = (nmax, nmax)

    Gbare = fdDGAsolver.siam_bare_Green(meshes(S0.G, Val(1)); e = e_fd, Δ = Δ_fd, D = D_fd)

    S_fd2 = ParquetSolver(nK1, nK2, nK3, Gbare, S0.G, S0.Σ, S0.F)
    init_sym_grp!(S_fd2)
    res = fdDGAsolver.solve!(S_fd2; strategy = :fdPA, parallel_mode = :threads, verbose = false);

    # coarse-grid self-energy is still close to the target
    @test absmax(S_fd2.Σ - S.Σ) < 3e-3
    
    # vertex evaluator check at specific in-bounds frequencies
    Ω = MatsubaraFrequency(T, 0, Boson)
    ν = MatsubaraFrequency(T, 1, Fermion)
    νp = MatsubaraFrequency(T, -2, Fermion)

    # channel-level: S_fd2.F + S_fd2.F0 should reproduce S.F for each reducible channel
    for ch in [:γa, :γp, :γt]
        z1 = getproperty(S_fd2.F, ch)(Ω, ν, νp) + getproperty(S_fd2.F0, ch)(Ω, ν, νp)
        z2 = getproperty(S.F, ch)(Ω, ν, νp)
        @test z1 ≈ z2 atol = 5e-4
    end

    # full vertex: S_fd2.F should agree with S.F for all channels and spin projections
    for ch in [pCh, tCh, aCh], sp in [pSp, xSp]
        @test S_fd2.F(Ω, ν, νp, ch, sp) ≈ S.F(Ω, ν, νp, ch, sp) atol = 5e-4
    end

    # ---------------------------------------------------------------------------------------------------
    # Vertex asymptotic limits: νInf selects only the native channel
    # ---------------------------------------------------------------------------------------------------
    # When one fermionic argument is νInf, only the channel whose K2 leg aligns with the
    # finite argument can contribute a non-zero K2 or K3 term. The other two channels
    # collapse to their K1 part, which is already absorbed into the native channel's K2.
    # Therefore, evaluating with only the native channel flag set must give the same result
    # as evaluating with all channel flags set.
    F = S_fd.F

    Ω = MatsubaraFrequency(T, 1, Boson)
    ω = MatsubaraFrequency(T, 3, Fermion)

    for F0 in (true, false), ch in (pCh, tCh, aCh), sp in (pSp, xSp, dSp)
        # ν' = νInf: only the channel whose K2 depends on ν contributes
        x = F(Ω, ω, νInf, ch, sp; F0, γp = (ch === pCh), γt = (ch === tCh), γa = (ch === aCh))
        y = F(Ω, ω, νInf, ch, sp; F0)
        @test x ≈ y

        # ν = νInf: only the channel whose K2p depends on ν' contributes
        x = F(Ω, νInf, ω, ch, sp; F0, γp = (ch === pCh), γt = (ch === tCh), γa = (ch === aCh))
        y = F(Ω, νInf, ω, ch, sp; F0)
        @test x ≈ y
    end

    # explicit check: F(Ω, ω, νInf) = γ_ch(Ω, ω, νInf) + F0.γ_ch(Ω, ω, νInf) + F0.F0(Ω, ω, νInf)
    # The fdPA vertex F is stored as the increment ΔF, so the full vertex is F + F0
    @test F(Ω, ω, νInf, pCh, pSp) ≈ F.γp(Ω, ω, νInf) + F.F0.γp(Ω, ω, νInf) + F.F0.F0(Ω, ω, νInf, pCh, pSp)
    @test F(Ω, ω, νInf, tCh, pSp) ≈ F.γt(Ω, ω, νInf) + F.F0.γt(Ω, ω, νInf) + F.F0.F0(Ω, ω, νInf, tCh, pSp)
    @test F(Ω, ω, νInf, aCh, pSp) ≈ F.γa(Ω, ω, νInf) + F.F0.γa(Ω, ω, νInf) + F.F0.F0(Ω, ω, νInf, aCh, pSp)
    @test F(Ω, νInf, ω, pCh, pSp) ≈ F.γp(Ω, νInf, ω) + F.F0.γp(Ω, νInf, ω) + F.F0.F0(Ω, νInf, ω, pCh, pSp)
    @test F(Ω, νInf, ω, tCh, pSp) ≈ F.γt(Ω, νInf, ω) + F.F0.γt(Ω, νInf, ω) + F.F0.F0(Ω, νInf, ω, tCh, pSp)
    @test F(Ω, νInf, ω, aCh, pSp) ≈ F.γa(Ω, νInf, ω) + F.F0.γa(Ω, νInf, ω) + F.F0.F0(Ω, νInf, ω, aCh, pSp)
end
