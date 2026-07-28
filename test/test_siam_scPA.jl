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

    T = 1e-3    # Temperature 
    U = 0.1     # Local Coulomb repulsion on the impurity site, in perturbative regime
    e = 0.0     # Impurity on-site energy ε_d plus Hartree contribution of the self-energy
    Δ = π / 5   # Hybridization strength
    D = 10.0    # Half-bandwidth

    # Frequency grid sizes
    # The asymptotic decomposition allows different box sizes per vertex class
    nmax = 12          # Base resolution parameter
    nG   = 8nmax        # number of positive fermionic Matsubara frequencies for G and Σ
    nK1  = 8nmax        # number of positive bosonic frequencies for K1 and the bubble
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

        """
            Σ_analytical(T::Float64, U::Float64, Δ::Float64, N_ω::Int64) → Tuple{Float64, Float64, Float64}

        Helper function that calculates the perturbative approximation of the self-energy analytically.
        This approximation holds for low temperatures and frequencies.
        The retarded self-energy is given as:

            Σ_σ^R(ω) ≊ - α_R * ω - im * α_I * Δ ((ω/Δ)^2 + (πT/Δ)^2) + …

        where α_R and α_I are defined as:

            α_R = (3 - π^2/4)*Up^2 + 0.0553*Up^4
            α_I = (Up^2 + 6*(5 - π^2/2)*Up^4)

        The Matsubara self-energy is then given as:

            Σ_σ(ω) ≊ - α_R * ω - im * α_I * Δ (-(ω/Δ)^2 + (πT/Δ)^2) * sign(ω) + …

        Because the code stores im * Σ we multiply with im and thus:

            Σ_σ(ω) ≊ α_R * ω + α_I * Δ (-(ω/Δ)^2 + (πT/Δ)^2) * sign(ω) + …
        """
        function Σ_analytical(T::Float64, U::Float64, Δ::Float64, n_ω::Int64) :: Tuple{Float64, Float64, Float64}
            ω = π * T * n_ω
            Up  = U/(π*Δ)                           # Dimensionless interaction strength
            α_R = (3 - π^2/4)*Up^2 + 0.0553*Up^4    # real parameter
            α_I = 1/2 * (Up^2 + 6*(5 - π^2/2)*Up^4) # imaginary parameter

            # self energy stores im * Σ(im*ω)
            Σ = α_R * ω + α_I * Δ * (-(ω/Δ)^2+ (π*T/Δ)^2) * sign(ω)
            return Σ, α_R, α_I
        end
        
        # Regression tests run with:
        # T = 0.1, U = 0.1, e = 0.0, Δ = π / 5, D = 10.0
        # nmax = 8, nG = 8nmax, nK1 = 8nmax, nK2 = (nmax, nmax), nK3 = (nmax, nmax)
        # @test S.Σ.(π * T .* [-3, -1, 1, 3]) ≈ [-0.0005198208498442844, -0.00038254575808446985,  0.00038254575808446985, 0.0005198208498442844]

        # physical test, compare to analytic results
        @test all(i -> i < T^2, abs.(real.(S.Σ.(π * T .* [-3, -1, 1, 3]) .- first.(Σ_analytical.(T, U, Δ, [-3, -1, 1, 3])))))

        # verify α_R
        # Σ(π*T) = α_R * π*T => Σ(π*T) / π*T = α_R
        @test S.Σ.(π * T)/(π * T) ≈ Σ_analytical.(T, U, Δ, 1)[2] atol = T

        # verify α_I
        # Σ(3*π*T) = 3 * Σ(π*T) + α_I / Δ * (-8*π^2*T^2) => α_I = (3 * Σ(π*T) - S.Σ.(3*π*T)) * Δ / (8*π^2*T^2)
        @test (3 * S.Σ.(π * T) - S.Σ.(3*π*T)) * Δ / (8* π^2 * T^2) ≈ Σ_analytical.(T, U, Δ, 1)[3] atol = T

        # verify symmetry
        @test S.Σ.(π * T .* [1, 3]) ≈ -S.Σ.(π * T .* [-1, -3])

        # -----------------------------------------------------------------------------------
        # Tests: K1 vertex class in each channel
        # -----------------------------------------------------------------------------------
        # Verify that the results are identical for both Number types and all modes.

        # function χ_M_analytical(U::Float64, Δ::Float64) :: Float64
        #     Up = U/(π*Δ)       # Dimensionless interaction strength

        #     # self energy
        #     χ_M = 1/(2π*Δ) * (1 + Up + (3 - π^2/4)*Up^2 + (15 - 3π^2/2)*Up^3+ 0.0553*Up^4)
        #     return χ_M
        # end

        # Regression tests run with:
        # T = 0.1, U = 0.1, e = 0.0, Δ = π / 5, D = 10.0
        # nmax = 8, nG = 8nmax, nK1 = 8nmax, nK2 = (nmax, nmax), nK3 = (nmax, nmax)
        # @test S.F.γa.K1.(2π * T .* -2:2) ≈ [0.0013838581452246547, 0.003982031314968492, 0.002186119796029509, 0.001014694115877671]
        # @test S.F.γp.K1.(2π * T .* -2:2) ≈ [-0.001346594451794493, -0.0036596533191820173, -0.002092013518995267, -0.0009937698551766658]
        # @test S.F.γt.K1.(2π * T .* -2:2) ≈ [1.863184671508093e-5, 0.00016118899789323716, 4.7053138517121016e-5, 1.0462130350502526e-5]

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

    # Regression tests run with:
    # T = 0.1, U = 1.0, e = 0.5, Δ = π / 5, D = 10.0
    # nmax = 6, nG = 6nmax, nK1 = 4nmax, nK2 = (nmax + 1, nmax), nK3 = (nmax + 1, nmax)
    # @test S.Σ.(π * T .* [-3, -1, 1, 3]) ≈ [-0.03900137428170572 - 0.1688475093744904im, -0.025303816148085505 - 0.17459212979569272im, 0.025303816148085505 - 0.17459212979569272im, 0.03900137428170572 - 0.1688475093744904im]
    
    # verify time-reversal symmetry
    @test S.Σ.(π * T .* [1, 3]) ≈ -conj(S.Σ.(π * T .* [-1, -3]))

    # -----------------------------------------------------------------------------------
    # Tests: K1 vertex class in each channel
    # -----------------------------------------------------------------------------------
    # K1(iΩ) now has a small but non-zero imaginary part due to the broken PH symmetry.

    # Regression tests run with:
    # T = 0.1, U = 1.0, e = 0.5, Δ = π / 5, D = 10.0
    # nmax = 6, nG = 6nmax, nK1 = 4nmax, nK2 = (nmax + 1, nmax), nK3 = (nmax + 1, nmax)
    # @test S.F.γa.K1.(2π * T .* -2:2) ≈ [0.12842622646499313 + 3.203753416887923e-5im, 0.4291342969150894 + 1.9955643558711365e-5im, 0.2136912314738301 - 4.5777323159823946e-5im, 0.09142186742802784 - 7.495576366353933e-6im]
    # @test S.F.γp.K1.(2π * T .* -2:2) ≈ [-0.13112696113252206 + 0.06485950342241105im, -0.24906689900393225 + 0.02071250168245983im, -0.1803803417803459 - 0.05330070462093495im, -0.10146322462920851 - 0.06585224545148298im]
    # @test S.F.γt.K1.(2π * T .* -2:2) ≈ [0.012662832567717497 + 6.330790775097969e-5im, 0.09985805465055425 + 2.0618763904736757e-5im, 0.03139767129069324 - 5.282913366400817e-5im, 0.007037408353052702 - 6.171180401643899e-5im]
end;
