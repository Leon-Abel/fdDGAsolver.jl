# Test suite for fixed_momentum_view, which returns a callable view of a vertex with the
# three momentum arguments (P, k, q) and the channel fixed, so the caller only needs to
# supply frequency and spin arguments. The view must evaluate identically to calling the
# full vertex with the same fixed momenta explicitly.

using fdDGAsolver
using MatsubaraFunctions
using StaticArrays
using Test

@testset "fixed_momentum_view" begin
    # ---------------------------------------------------------------------------------------------------
    # Physical and numerical parameters
    # ---------------------------------------------------------------------------------------------------
    T  = 0.5                                            # temperature
    k1 = 2pi * SVector(1., 0.)                          # first reciprocal lattice vector
    k2 = 2pi * SVector(0., 1.)                          # second reciprocal lattice vector
    mK = BrillouinZoneMesh(BrillouinZone(3, k1, k2))    # 3×3 Brillouin zone mesh

    # ---------------------------------------------------------------------------------------------------
    # Construct three vertex types with random data
    # ---------------------------------------------------------------------------------------------------
    # Vertex: local (momentum-independent) full vertex built on top of a reference vertex
    F1 = fdDGAsolver.Vertex(RefVertex(T, 2.), T, 10, (4, 4), (3, 2))
    unflatten!(F1, rand(ComplexF64, length(flatten(F1))))

    # NL_Vertex: non-local vertex with one momentum argument per channel leg
    F2 = fdDGAsolver.NL_Vertex(F1, T, 8, (4, 4), (3, 2), mK)
    unflatten!(F2, rand(ComplexF64, length(flatten(F2))))

    # NL2_Vertex: non-local vertex with two independent momentum arguments per channel leg
    F3 = fdDGAsolver.NL2_Vertex(F2, T, 5, (3, 3), (2, 2), mK)
    unflatten!(F3, rand(ComplexF64, length(flatten(F3))))

    # ---------------------------------------------------------------------------------------------------
    # Test frequencies and momenta to fix in the view
    # ---------------------------------------------------------------------------------------------------
    Ω  = MatsubaraFrequency(T, -1, Boson)    # bosonic transfer frequency
    ν0 = MatsubaraFrequency(T, -1, Fermion)  # finite incoming fermionic frequency
    ω0 = MatsubaraFrequency(T,  1, Fermion)  # finite outgoing fermionic frequency
    P  = BrillouinPoint(-1, 2)               # bosonic transfer momentum
    k  = BrillouinPoint( 2, 7)               # incoming fermionic momentum
    q  = BrillouinPoint( 1, -2)              # outgoing fermionic momentum

    # ---------------------------------------------------------------------------------------------------
    # Consistency check: fixed_momentum_view == full vertex call with explicit momenta
    # ---------------------------------------------------------------------------------------------------
    # For each channel (a, p, t), construct the momentum-fixed views of all three vertex types
    # and verify that they reproduce the full vertex evaluator for all combinations of:
    #   - spin indices (pSp, xSp, dSp)
    #   - finite and infinite fermionic frequencies (ν0, νInf) × (ω0, νInf)
    #   - asymptotic component flags (F0, γa, γp, γt) toggled individually and all at once
    for Ch in (aCh, pCh, tCh)
        F1view = fdDGAsolver.fixed_momentum_view(F1, P, k, q, Ch)
        F2view = fdDGAsolver.fixed_momentum_view(F2, P, k, q, Ch)
        F3view = fdDGAsolver.fixed_momentum_view(F3, P, k, q, Ch)
        for Sp in (pSp, xSp, dSp), ν in [ν0, νInf], ω in [ω0, νInf]
            for (γa, γp, γt, F0) in [
                (true, true, true, true),       # all components
                (true, false, false, false),    # a-channel reducible part only
                (false, true, false, false),    # p-channel reducible part only
                (false, false, true, false),    # t-channel reducible part only
                (false, false, false, true),    # fully irreducible (bare) part only
            ]
                @test F1view(Ω, ν, ω, Ch, Sp; F0, γt, γp, γa) ≈ F1(Ω, ν, ω, P, k, q, Ch, Sp; F0, γt, γp, γa)
                @test F2view(Ω, ν, ω, Ch, Sp; F0, γt, γp, γa) ≈ F2(Ω, ν, ω, P, k, q, Ch, Sp; F0, γt, γp, γa)
                @test F3view(Ω, ν, ω, Ch, Sp; F0, γt, γp, γa) ≈ F3(Ω, ν, ω, P, k, q, Ch, Sp; F0, γt, γp, γa)
            end
        end
    end
end
