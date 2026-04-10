# Test suite for the Vertex type, which represents the full 2-particle vertex in the
# asymptotic channel decomposition
#   F(Ω, ν, ν') = F0 + γ_p(Ω, ν, ν') + γ_t(Ω, ν, ν') + γ_a(Ω, ν, ν').
# Each channel γ_r is itself decomposed into asymptotic classes K1, K2, K3 (see test_channel.jl).

using fdDGAsolver
using MatsubaraFunctions
using HDF5
using Test

@testset "Vertex" begin
    # ---------------------------------------------------------------------------------------------------
    # Physical and numerical parameters for constructing the Matsubara meshes
    # ---------------------------------------------------------------------------------------------------
    T     = 0.5         # temperature
    U     = 1.0         # bare Hubbard interaction
    numK1 = 10          # number of positive bosonic frequencies in the K1 mesh
    numK2 = (5, 5)      # (number of positive bosonic, number of positive fermionic) frequencies for the K2 mesh
    numK3 = (3, 3)      # (number of positive bosonic, number of positive fermionic) frequencies for the K3 mesh

    # ---------------------------------------------------------------------------------------------------
    # Number of frequencies
    # ---------------------------------------------------------------------------------------------------

    F0 = fdDGAsolver.RefVertex(T, U)
    F  = fdDGAsolver.Vertex(F0, T, numK1, numK2, numK3)

    # Verify that mesh metadata is correctly stored and retrievable via the getter methods
    @test MatsubaraFunctions.temperature(F) == T
    @test fdDGAsolver.numK1(F) == numK1
    @test fdDGAsolver.numK2(F) == numK2
    @test fdDGAsolver.numK3(F) == numK3

    # Total length is the sum over the three channels; each channel has the same length as in
    # test_channel.jl, so the vertex length is exactly 3 times a single channel's length.
    @test length(F) == 3 * ((2numK1 - 1) + (2numK2[1] - 1) * 2numK2[2] + (2numK3[1] - 1) * (2numK3[2])^2)

    # flatten serializes all channel data into a single vector; its length must match length(F)
    @test length(flatten(F)) == length(F)

    # ---------------------------------------------------------------------------------------------------
    # Copy
    # ---------------------------------------------------------------------------------------------------

    # Fill all three channels with random data to ensure a non-trivial comparison below
    F.γp.K1.data .= rand(size(F.γp.K1.data)...)
    F.γp.K2.data .= rand(size(F.γp.K2.data)...)
    F.γp.K3.data .= rand(size(F.γp.K3.data)...)
    F.γt.K1.data .= rand(size(F.γt.K1.data)...)
    F.γt.K2.data .= rand(size(F.γt.K2.data)...)
    F.γt.K3.data .= rand(size(F.γt.K3.data)...)
    F.γa.K1.data .= rand(size(F.γa.K1.data)...)
    F.γa.K2.data .= rand(size(F.γa.K2.data)...)
    F.γa.K3.data .= rand(size(F.γa.K3.data)...)

    # copy vertex
    F_copy = copy(F)
    # verify that both vertices are identical
    @test F == F_copy

    # Zero the copy
    set!(F_copy, 0)

    # verify that the vertices are now distinct and that only the copy was zeroed
    @test F != F_copy
    @test MatsubaraFunctions.absmax(F) > 0
    @test MatsubaraFunctions.absmax(F_copy) == 0

    # ---------------------------------------------------------------------------------------------------
    # Flatten / unflatten
    # ---------------------------------------------------------------------------------------------------

    # unflatten! restores F_copy from the flattened vector of F
    unflatten!(F_copy, flatten(F))
    # verify that the vertex is recovered
    @test F == F_copy

    # ---------------------------------------------------------------------------------------------------
    # Evaluation
    # ---------------------------------------------------------------------------------------------------
    # Test the spin-component evaluators and the crossing/density relations.
    # The pSp evaluator sums F0 and the three reducible channels after frequency conversion.
    # The xSp component is computed on the fly from crossing symmetry (types.jl):
    #   Γ^x_p(Ω,ν,ν') = -Γ^=_p(Ω,ν,Ω−ν')
    #   Γ^x_t(Ω,ν,ν') = -Γ^=_a(Ω,ν,ν')
    #   Γ^x_a(Ω,ν,ν') = -Γ^=_t(Ω,ν,ν')
    # The dSp component satisfies dSp = 2*pSp + xSp.

    # pick arbitrary in-bounds frequencies (inside the smallest K3 grid)
    Ω  = value(meshes(F.γp.K3, Val(1))[1])
    ν1 = value(meshes(F.γp.K3, Val(2))[1])
    ν2 = value(meshes(F.γp.K3, Val(3))[1])

    for Ch in [pCh, tCh, aCh]
        # dSp = 2*pSp + xSp must hold for every channel convention
        @test F(Ω, ν1, ν2, Ch, dSp) ≈ 2 * F(Ω, ν1, ν2, Ch, pSp) + F(Ω, ν1, ν2, Ch, xSp)
    end

    # crossing symmetry: xSp in pCh equals -pSp with ν' exchanged for Ω - ν'
    @test F(Ω, ν1, ν2, pCh, xSp) ≈ -F(Ω, ν1, Ω - ν2, pCh, pSp)
    # crossing symmetry: xSp in tCh equals -pSp in aCh
    @test F(Ω, ν1, ν2, tCh, xSp) ≈ -F(Ω, ν1, ν2, aCh, pSp)
    # crossing symmetry: xSp in aCh equals -pSp in tCh
    @test F(Ω, ν1, ν2, aCh, xSp) ≈ -F(Ω, ν1, ν2, tCh, pSp)

    # ---------------------------------------------------------------------------------------------------
    # I/O
    # ---------------------------------------------------------------------------------------------------

    testfile = dirname(@__FILE__) * "/test.h5"                          # create path to file
    file = h5open(testfile, "w")                                        # create and open file in write mode
    save!(file, "f", F)                                                 # save vertex under label "f"
    close(file)                                                         # close file

    file = h5open(testfile, "r")                                        # open file in read-only mode
    Fp = fdDGAsolver.load_vertex(fdDGAsolver.Vertex, file, "f")         # reconstruct vertex from file

    # verify equality and type
    @test F == Fp
    @test Fp isa fdDGAsolver.Vertex

    close(file)             # close file
    rm(testfile; force=true)  # delete testfile
end