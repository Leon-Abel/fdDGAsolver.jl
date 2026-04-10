# test_siam_scPA.jl
#
# Integration test for the self-consistent Parquet Approximation (scPA) applied to
# the Single Impurity Anderson Model (SIAM) at
#   1. half-filling (particle-hole symmetry) and
#   2. finite doping (broken particle-hole symmetry).
#
# The SIAM describes a single correlated impurity level coupled to a non-interacting
# bath. The Hamiltonian reads:
#
#   H = Σ_σ ε_d n_{d,σ} + U n_{d,↑} n_{d,↓} + Σ_{k,σ} ε_k c†_{k,σ} c_{k,σ}
#       + Σ_{k,σ} (V_k d†_σ c_{k,σ} + h.c.)


using fdDGAsolver
using MatsubaraFunctions
using HDF5
using Test

# ---------------------------------------------------------------------------------------------------
# Test 1: SIAM at half-filling (ε_d + Σ_Hartree = 0, particle-hole symmetry)
# ---------------------------------------------------------------------------------------------------

@testset "SIAM parquet half-filling" begin
    using MPI
    MPI.Init()

    # ---------------------------------------------------------------------------------------------------
    # Physical and numerical parameters
    # ---------------------------------------------------------------------------------------------------

    T = 0.1     # Temperature 
    U = 1.0     # Local Coulomb repulsion on the impurity site
    e = 0.0     # Impurity on-site energy ε_d plus Hartree contribution of the self-energy
    Δ = π / 5   # Hybridization strength to the bath
    D = 10.0    # Half-bandwidth

    # Frequency grid sizes
    # The asymptotic decomposition allows different box sizes per vertex class
    nmax = 6            # Base resolution parameter
    nG   = 6nmax        # number of positive fermionic Matsubara frequencies for G and Σ
    nK1  = 4nmax        # number of positive bosonic frequencies for K1 and the bubble
    nK2  = (nmax, nmax) # (bosonic, fermionic) grid sizes for K2 
    nK3  = (nmax, nmax) # (bosonic, fermionic) grid sizes for K3

    # -----------------------------------------------------------------------------------
    # Loop over number types and parallelization modes
    # -----------------------------------------------------------------------------------

    # The solver supports both real (Float64) and complex (ComplexF64) arithmetic.
    # At half-filling Float64 and ComplexF64 have to return the same result.
    # This is because at half-filling the self-energy and propagator are purely imaginary.
    # Due to the code convention of storing (i*G) and (i*Σ), the stored values are purely real.
    # The vertex components (K1, K2, K3) are also purely real which follows from complex
    # conjugation and time-reversal symmetry:
    #   [F(ν1​,ν2​,ν3​,ν4​)]* = F(−ν1​,−ν2​,−ν3​,−ν4​) = F(ν1​,ν2​,ν3​,ν4​)
    # Therefore Float64 is sufficient and yields the same result as ComplexF64.
    # The four parallelization modes (:serial, :threads, :polyester, :hybrid) all perform
    # the same computation; this loop verifies that all modes give identical results.
    # :polyester verifies that the code defaults back to :serial for an unknown scheme. 
    for Q in [Float64, ComplexF64], mode in [:serial, :threads, :hybrid,] #:polyester]

        # Build solver
        S = parquet_solver_siam_parquet_approximation(nG, nK1, nK2, nK3, Q; e, T, D, Δ, U)

        # Initialize the symmetry groups for the self-energy and each vertex channel.
        # Symmetry groups encode relations such as time-reversal, particle-hole conjugation,
        # and crossing symmetry, which are used to reduce the number of independent
        # components that need to be stored and computed.
        init_sym_grp!(S)

        # Run the self-consistent parquet solver using Anderson mixing to find the
        # fixed-point solution F*, Σ* satisfying the full parquet + Dyson + SDE system:
        #   F  = U + Σ_r BSE_r[F, G]        (parquet / Bethe-Salpeter equations per channel)
        #   G  = G_bare / (1 - Σ G_bare)    (Dyson equation)
        #   Σ  = SDE[F, G]                  (Schwinger-Dyson equation for the self-energy)
        res = fdDGAsolver.solve!(S; strategy = :scPA, verbose = false, parallel_mode = mode);

        # -----------------------------------------------------------------------------------
        # Test: self-energy Σ(iν)
        # -----------------------------------------------------------------------------------
        # At half-filling PH symmetry enforces Σ(iν_n) = -Σ(-iν_{n+1})* = -Σ(-iν_n).
        # Verify that the results are identical for both Number types and all modes.

        @test S.Σ.(π * T .* [-3, -1, 1, 3]) ≈ [-0.052254330862359824, -0.03843903766760937, 0.03843903766760937, 0.052254330862359824]
        @test S.Σ.(π * T .* [1, 3]) ≈ -S.Σ.(π * T .* [-1, -3])

        # -----------------------------------------------------------------------------------
        # Tests: K1 vertex class in each channel
        # -----------------------------------------------------------------------------------
        # Verify that the results are identical for both Number types and all modes.

        @test S.F.γa.K1.(2π * T .* -2:2) ≈ [0.14137982758187426, 0.5546047324922341, 0.2438526467553354, 0.09950853979301386]
        @test S.F.γp.K1.(2π * T .* -2:2) ≈ [-0.10998983785678543, -0.24406268938560566, -0.16083938676651757, -0.08256881993800179]
        @test S.F.γt.K1.(2π * T .* -2:2) ≈ [0.015676298896395317, 0.15524108943442552, 0.041480347657872174, 0.0084597383808721]
        
        # old values
        # @test S.Σ.(π * T .* [-3, -1, 1, 3]) ≈ [-0.052138235296134906, -0.03838544776344314, 0.03838544776344314, 0.052138235296134906]
        # @test S.F.γa.K1.(2π * T .* -2:2) ≈ [0.13203850929270397, 0.5403615530152339, 0.2333246221064017, 0.09056300899983459]
        # @test S.F.γp.K1.(2π * T .* -2:2) ≈ [-0.10420799999591804, -0.2403951910434166, -0.15592452265704748, -0.07622568434721624]
        # @test S.F.γt.K1.(2π * T .* -2:2) ≈ [0.013898648018808482, 0.1499562726081748, 0.03867632419082161, 0.007160400841240708]

        # -----------------------------------------------------------------------------------
        # Type consistency checks
        # -----------------------------------------------------------------------------------
        # Verify that the internal number type of the vertex and self-energy data matches
        # the requested type Q (either Float64 or ComplexF64).
        @test eltype(S.F) == Q
        @test eltype(S.Σ.data) == Q
    end
end

# ---------------------------------------------------------------------------------------------------
# Test 2: SIAM away from half-filling (finite doping, broken particle-hole symmetry)
# ---------------------------------------------------------------------------------------------------
# Setting e = 0.5 ≠ 0 breaks PH symmetry. The self-energy acquires a non-trivial
# imaginary part at all Matsubara frequencies (the Hartree shift no longer cancels ε_d),
# and the vertex becomes complex even in real-frequency-equivalent representations.
# This requires ComplexF64 arithmetic (default type in this test).

@testset "SIAM parquet doped" begin
    using MPI
    MPI.Init()

    # ---------------------------------------------------------------------------------------------------
    # Physical and numerical parameters
    # ---------------------------------------------------------------------------------------------------

    T = 0.1     # Temperature 
    U = 1.0     # Hubbard-U: local Coulomb repulsion on the impurity site
    e = 0.5     # Impurity on-site energy plus Hartree contribution of the self-energy
    Δ = π / 5   # Hybridization strength to the bath
    D = 10.0    # Half-bandwidth

    # Frequency grid sizes
    # nK2/nK3 bosonic dimension increased by 1 compared to half-filling
    # to better resolve asymmetries in the bosonic frequency dependence when PH symmetry
    # is broken.
    nmax = 6            # Base resolution parameter
    nG  = 6nmax         # number of positive fermionic Matsubara frequencies for G and Σ
    nK1 = 4nmax         # number of positive bosonic frequencies for K1 and the bubble
    nK2 = (nmax + 1, nmax)  # (bosonic, fermionic) grid sizes for K2 
    nK3 = (nmax + 1, nmax)  # (bosonic, fermionic) grid sizes for K3

    # Build solver (no explicit Q type → defaults to ComplexF64, required for broken PH)
    S = parquet_solver_siam_parquet_approximation(nG, nK1, nK2, nK3; e, T, D, Δ, U)

    # Initialize the symmetry groups
    init_sym_grp!(S)

    # Solve with threads parallelization (single mode tested here)
    res = fdDGAsolver.solve!(S; strategy = :scPA, verbose = false, parallel_mode = :threads);

    # -----------------------------------------------------------------------------------
    # Test: self-energy Σ(iν)
    # -----------------------------------------------------------------------------------
    # Away from half-filling Σ(iν) is no longer purely imaginary and the antisymmetry
    # Σ(iν_n) = -Σ(-iν_n) is replaced by Σ(iν_n) = -Σ(-iν_n)* (complex conjugate),
    # reflecting time-reversal symmetry.
    @test S.Σ.(π * T .* [-3, -1, 1, 3]) ≈ [-0.03900137428170572 - 0.1688475093744904im, -0.025303816148085505 - 0.17459212979569272im, 0.025303816148085505 - 0.17459212979569272im, 0.03900137428170572 - 0.1688475093744904im]
    @test S.Σ.(π * T .* [1, 3]) ≈ -conj(S.Σ.(π * T .* [-1, -3]))

    # -----------------------------------------------------------------------------------
    # Tests: K1 vertex class in each channel
    # -----------------------------------------------------------------------------------
    # K1(iΩ) now has a small but non-zero imaginary part due to the broken PH symmetry.
    @test S.F.γa.K1.(2π * T .* -2:2) ≈ [0.12842622646499313 + 3.203753416887923e-5im, 0.4291342969150894 + 1.9955643558711365e-5im, 0.2136912314738301 - 4.5777323159823946e-5im, 0.09142186742802784 - 7.495576366353933e-6im]
    @test S.F.γp.K1.(2π * T .* -2:2) ≈ [-0.13112696113252206 + 0.06485950342241105im, -0.24906689900393225 + 0.02071250168245983im, -0.1803803417803459 - 0.05330070462093495im, -0.10146322462920851 - 0.06585224545148298im]
    @test S.F.γt.K1.(2π * T .* -2:2) ≈ [0.012662832567717497 + 6.330790775097969e-5im, 0.09985805465055425 + 2.0618763904736757e-5im, 0.03139767129069324 - 5.282913366400817e-5im, 0.007037408353052702 - 6.171180401643899e-5im]

    # old values
    # @test S.Σ.(π * T .* [-3, -1, 1, 3]) ≈ [-0.0389123277075552 - 0.16855090184215607im, -0.025252640312580586 - 0.17429637478745583im, 0.025252640312580586 - 0.17429637478745583im, 0.0389123277075552 - 0.16855090184215607im]
    # @test S.F.γa.K1.(2π * T .* -2:2) ≈ [0.11925962005661812 + 8.57514999021054e-5im, 0.416232811242488 + 3.319936929625957e-5im, 0.20353141073439696 - 8.209974062547027e-5im, 0.08259294412067451 - 7.660306021755952e-5im]
    # @test S.F.γp.K1.(2π * T .* -2:2) ≈ [-0.12570450372739117 + 0.06583917638195431im, -0.24548654160724023 + 0.021014183409764874im, -0.17578586892780296 - 0.05408344507941078im, -0.09544981624337806 - 0.06686768343644132im]
    # @test S.F.γt.K1.(2π * T .* -2:2) ≈ [0.011016969129875598 + 7.455601769977568e-5im, 0.09533547032272821 + 2.4477973720973318e-5im, 0.028843799251846686 - 6.260706942592786e-5im, 0.005828299701446311 - 7.32415771550901e-5im]
end;
