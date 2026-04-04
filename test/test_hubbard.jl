# Test suite for the nonlocal (lattice) Hubbard model infrastructure.
#
# The tested objects are:
#   1. Bare (non-interacting) lattice Green function G₀(k, iν)
#   2. Particle-particle (pp) and particle-hole (ph) bubbles Π_{pp/ph}(q, iΩ, iν)
#   3. Dyson equation  G⁻¹ = G₀⁻¹ - Σ
#   4. Lattice occupation  n = T Σ_{iν,k} G(k, iν) e^{iν 0⁺} + 1/2

using fdDGAsolver # MPI.Init() is called internally here
using MatsubaraFunctions
using StaticArrays
using Test

@testset "Hubbard model" begin
    # ---------------------------------------------------------------------------------------------------
    # Physical and numerical parameters
    # ---------------------------------------------------------------------------------------------------

    T = 0.5     # temperature
    t1 = 1.3    # nearest-neighbour hopping amplitude 
    μ = 0.2     # chemical potential

    # Number of positive fermionic Matsubara frequencies ν_n = 2π(n + 1/2) * T kept in G, Σ.
    nG = 5
    # fermionic mesh; n ∈ [-nG,nG-1]
    mG = MatsubaraMesh(T, nG, Fermion) # 2*nG = 10 ν-points

    # Reciprocal lattice vectors (lattice constant a = 1)
    k1 = 2pi * SVector(1., 0.)
    k2 = 2pi * SVector(0., 1.)
    # BrillouinZone(L, k1, k1) discretises the BZ spanned by k1, k2 with an L×L k-point grid.
    # MatsubaraFunctions.euclidean(mK.points[n], mK) to recover euclidean coordinates
    mK = BrillouinZoneMesh(BrillouinZone(8, k1, k2)) # L*L = 64 k-points

    # ---------------------------------------------------------------------------------------------------
    # TEST 1: Bare Green function
    # ---------------------------------------------------------------------------------------------------
    # 
    # The non-interacting (bare) lattice Green function is
    #
    #   G₀(k, iν) = 1 / (iν + μ - ε_k)
    #
    # where ε_k = -2 * t1 * (cos(k1) + cos(k2)) for the nearest-neighbour square lattice
    # (see hubbard.jl → hubbard_band)
    #
    # Convention: Gbare stores im * G_physical, so we test -im * Gbare = G_physical
    # Keeps G real-valued at half-filling and simplifies the causality check Im Σ ≤ 0.

    Gbare = hubbard_bare_Green(mG, mK; μ, t1) # 2*nG x L*L array

    # Check G₀ at high-symmetry points of the square-lattice BZ.
    # Matsubara frequencies are evaluated at ν_n = 2π(n + 1/2) * T for n = -2,-1,0,1,2.
    # k-values used are:
    #   k = (0,  0):       ε_k = -4 * t1   ⇒ G₀⁻¹ = iν + μ + 4t1
    #   k = (0,  π/2):     ε_k = -2 * t1   ⇒ G₀⁻¹ = iν + μ + 2t1
    #   k = (0,  π):       ε_k =  0        ⇒ G₀⁻¹ = iν + μ
    #   k = (0,  3π/2):    ε_k = -2 * t1   ⇒ G₀⁻¹ = iν + μ + 2t1
    #   k = (π,  π):       ε_k = +4 * t1   ⇒ G₀⁻¹ = iν + μ - 4t1
    for ν in 2π * T * ((-2:2) .+ 1/2)
        @test -im * Gbare(ν, SVector(0., 0.)  ) ≈ 1 / (im * ν + μ + 4t1)
        @test -im * Gbare(ν, SVector(0., π/2) ) ≈ 1 / (im * ν + μ + 2t1)
        @test -im * Gbare(ν, SVector(0., π)   ) ≈ 1 / (im * ν + μ + 0t1)
        @test -im * Gbare(ν, SVector(0., 3π/2)) ≈ 1 / (im * ν + μ + 2t1)
        @test -im * Gbare(ν, SVector{2, Float64}(π, π)) ≈ 1 / (im * ν + μ - 4t1)

        # Verify periodicity: G₀(k + G_nm, iν) = G₀(k, iν) for G_nm = (2π*m, 2π*n)
        @test Gbare(ν, SVector(0.2 + 2π, 0.4 - 2π)) ≈ Gbare(ν, SVector(0.2, 0.4))
    end

    # ---------------------------------------------------------------------------------------------------
    # Bubble creation
    # ---------------------------------------------------------------------------------------------------
    #
    # The particle-particle (pp) and particle-hole (ph) bubbles are defined as
    #
    #   Π^{pp}(iΩ, iν; q, k) = G(k; iν) * G(iΩ-iν; q-k)
    #   Π^{ph}(iΩ, iν; q, k) = G(k; iν) * G(iΩ+iν; k+q)
    #
    # In the nonlocal version the bubbles retain the full (q, k) momentum
    # dependence, giving MeshFunctions with meshes (iΩ, iν, q, k).
    #
    # Two equivalent algorithms are provided:
    #   bubbles_momentum_space! : direct product in k-space (see nonlocal/bubble.jl)
    #   bubbles_real_space!     : FFT-based convolution     (see nonlocal/bubble.jl)
    # Both should yield the same result.

    mΠΩ = MatsubaraMesh(T, 4, Boson)    # bosonic mesh for transfer frequency Ω, 2*4 - 1 = 7 Ω-points
    mΠν = MatsubaraMesh(T, 8, Fermion)  # fermionic mesh for loop frequency ν, 2*8 = 16 ν-points

    # Allocate bubble arrays: (iΩ, iν, q, k)
    Πpp = MeshFunction(mΠΩ, mΠν, mK, mK) # 7 x 16 x 64 x 64 array
    Πph = MeshFunction(mΠΩ, mΠν, mK, mK) # 7 x 16 x 64 x 64 array

    # momentum space; faster, higher memory cost
    fdDGAsolver.bubbles_momentum_space!(Πpp, Πph, Gbare)
    Πpp_mom_space  = copy(Πpp)
    Πph_mom_space  = copy(Πph)

    # real space; slower, lower memory cost
    fdDGAsolver.bubbles_real_space!(Πpp, Πph, Gbare)
    Πpp_real_space = copy(Πpp)
    Πph_real_space = copy(Πph)

    # verify equivalence
    @test absmax(Πpp_mom_space - Πpp_real_space) < 1e-10
    @test absmax(Πph_mom_space - Πph_real_space) < 1e-10

    # ---------------------------------------------------------------------------------------------------
    # TEST 2: Dyson equation
    # ---------------------------------------------------------------------------------------------------
    #
    # The full lattice Green function satisfies the Dyson equation
    #
    #   G(k, iν)⁻¹ = G₀(k, iν)⁻¹ - Σ(k, iν)
    #
    # which in the code convention (storing im*G and im*Σ) reads 
    #
    #   (i G)⁻¹ = (i G₀)⁻¹ + (i Σ)
    #
    # (see dyson.jl → Dyson!)

    G = MeshFunction(mG, mK) # target Green function (will be overwritten)
    Σ = MeshFunction(mG, mK) # self-energy

    # --- Trivial case: Σ = 0 → G = G₀ ---
    set!(Σ, 0)
    fdDGAsolver.Dyson!(G, Σ, Gbare) # updates G in place
    @test absmax(G - Gbare) < 1e-10

    # --- Non-trivial case: constant complex self-energy ---
    set!(Σ, -0.5 + 0.2im)
    fdDGAsolver.Dyson!(G, Σ, Gbare) # updates G

    # Pick one (ν, k) point for the explicit check.
    ν, k = π * T, SVector(π/2, 0.)
    @test G(ν, k) ≈ 1 / (1 / Gbare(ν, k) + Σ(ν, k))

    # Cross-check in physical (un-prefactored) variables.
    # Multiplying both sides of the code Dyson equation by -i recovers the
    # standard form:  (-i G)⁻¹ = (-i G₀)⁻¹ - (-i Σ).
    @test 1 / (-im * G(ν, k)) ≈ 1 / (-im * Gbare(ν, k)) - (-im * Σ(ν, k))


    # ---------------------------------------------------------------------------------------------------
    # TEST 3: Bubble equations (momentum-space algorithm)
    # ---------------------------------------------------------------------------------------------------

    # Re-compute Gbare and bubbles with the same parameters so the tests below
    # are independent of any mutation that occurred in section 3.
    Gbare = hubbard_bare_Green(mG, mK; μ, t1)
    Πpp = MeshFunction(mΠΩ, mΠν, mK, mK)
    Πph = MeshFunction(mΠΩ, mΠν, mK, mK)
    fdDGAsolver.bubbles_momentum_space!(Πpp, Πph, Gbare)

    # Pick specific Matsubara and momentum indices for the explicit check.
    Ω = MatsubaraFrequency(T, 3, Boson)     # iΩ_3  = 2π * 3 * T 
    ν = MatsubaraFrequency(T, -2, Fermion)  # iν_-2 = 2π * (3/2) * T
    P = value(mK[58])                       # arbitrary transfer momentum q
    k = value(mK[23])                       # arbitrary loop momentum k

    # Particle-particle bubble:  Π_{pp}(iΩ, iν; q, k) = G₀(iν, k) * G₀(iΩ-iν, q-k)
    @test Πpp(Ω, ν, P, k) ≈ Gbare(ν, k) * Gbare(Ω - ν, P - k) # check exact form of ≈
    # Particle-hole bubble:  Π_{ph}(iΩ, iν; q, k) = G₀(iν, k) * G₀(iΩ+iν, q+k)
    @test Πph(Ω, ν, P, k) ≈ Gbare(ν, k) * Gbare(Ω + ν, P + k)

    # ---------------------------------------------------------------------------------------------------
    # TEST 4: Local occupation per spin
    # ---------------------------------------------------------------------------------------------------
    #
    # The local occupation per spin is
    #
    #   n = 1/2 + T Σ_{iν} (1/N_k) Σ_k  (G(k, iν) - 1/(im ν)) e^{iν 0⁺} # der
    #
    # which in code variables (storing im * G) becomes
    #
    #   n = 1/2 + T/N_k · Im[ Σ_{ν,k} (im * G(iν, k)) ]
    #
    # (see dyson.jl → compute_occupation).
    # 
    # Compare to values calculated from (1/N_k) Σ_k f(β, ϵ_k - μ)
    #
    # The particle-hole symmetry n(μ) + n(-μ) = 1 (exact for t2 = t3 = 0) is explicitly verified
    # by the last two pairs of @test statements.

    # Use larger frequency and momentum mesh for better numerical accuracy.
    mG = MatsubaraMesh(T, 10000, Fermion)
    mK = BrillouinZoneMesh(BrillouinZone(16, k1, k2))

    @test abs(compute_occupation(hubbard_bare_Green(mG, mK; t1=1., μ=-4.)) - 0.030153827416513745) < 1e-4
    @test abs(compute_occupation(hubbard_bare_Green(mG, mK; t1=1., μ=-2.)) - 0.19640688635922182) < 1e-4
    # Half-filling: n = 1/2 exactly by particle-hole symmetry (μ = 0, t2 = t3 = 0)
    @test compute_occupation(hubbard_bare_Green(mG, mK; t1=1., μ=0.)) ≈ 0.5
    # Particle-hole symmetry: n(μ) = 1 - n(-μ)
    @test compute_occupation(hubbard_bare_Green(mG, mK; t1=1., μ=2.)) ≈ 1 - compute_occupation(hubbard_bare_Green(mG, mK; t1=1., μ=-2.))
    @test compute_occupation(hubbard_bare_Green(mG, mK; t1=1., μ=4.)) ≈ 1 - compute_occupation(hubbard_bare_Green(mG, mK; t1=1., μ=-4.))
end
