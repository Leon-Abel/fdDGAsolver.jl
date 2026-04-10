# Test suite for the Channel type, which represents a single 2-particle reducible vertex in one
# diagrammatic channel (p, t, or a) using the asymptotic decomposition
#   γ(Ω, ν, ν') = K1(Ω) + K2(Ω, ν) + K2(Ω, ν') + K3(Ω, ν, ν').


using fdDGAsolver
using MatsubaraFunctions
using HDF5
using Test

@testset "Channel" begin
    # ---------------------------------------------------------------------------------------------------
    # Physical and numerical parameters for constructing the Matsubara meshes
    # ---------------------------------------------------------------------------------------------------
    T = 0.5         # temperature 
    numK1 = 10      # number of positive bosonic frequencies in the K1 mesh
    numK2 = (5, 5)  # (number of positive bosonic, number of positive fermionic) frequencies for the K2 mesh
    numK3 = (3, 3)  # (number of positive bosonic, number of positive fermionic) frequencies for the K3 mesh

    # ---------------------------------------------------------------------------------------------------
    # Number of frequencies
    # ---------------------------------------------------------------------------------------------------

    γ = fdDGAsolver.Channel(T, numK1, numK2, numK3)

    # Verify that mesh metadata is correctly stored and retrievable via the getter methods
    @test MatsubaraFunctions.temperature(γ) == T
    @test fdDGAsolver.numK1(γ) == numK1
    @test fdDGAsolver.numK2(γ) == numK2
    @test fdDGAsolver.numK3(γ) == numK3

    # Total number of frequencies across all three asymptotic classes.
    # K1 lives on a bosonic mesh of     2*numK1 - 1    points.
    # K2 lives on a bosonic mesh of     2*numK2[1] - 1 points and
    #             a fermionic mesh of   2*numK2[2]     points.
    # K3 lives on a (bosonic) x (fermionic)^2 mesh.
    @test length(γ) == (2numK1 - 1) + (2numK2[1] - 1) * 2numK2[2] + (2numK3[1] - 1) * (2numK3[2])^2

    # flatten serializes all K1/K2/K3 data into a single vector; its length must match length(γ)
    @test length(flatten(γ)) == length(γ)

    # ---------------------------------------------------------------------------------------------------
    # Copy
    # ---------------------------------------------------------------------------------------------------

    # Fill K1, K2, K3 with random data to ensure a non-trivial comparison below
    γ.K1.data .= rand(size(γ.K1.data)...)
    γ.K2.data .= rand(size(γ.K2.data)...)
    γ.K3.data .= rand(size(γ.K3.data)...)

    # copy channel
    γ_copy = copy(γ)
    # verify that both channels are identical
    @test γ == γ_copy

    # Zero the copy 
    set!(γ_copy.K1, 0)
    set!(γ_copy.K2, 0)
    set!(γ_copy.K3, 0)

    # verify that the channels are distinct from each other and that only the copy was zeroed 
    @test γ != γ_copy
    @test MatsubaraFunctions.absmax(γ) > 0
    @test MatsubaraFunctions.absmax(γ_copy) == 0

    # ---------------------------------------------------------------------------------------------------
    # Flatten / unflatten
    # ---------------------------------------------------------------------------------------------------

    # unflatten! restores γ_copy from the flattened vector of γ
    unflatten!(γ_copy, flatten(γ))
    # verify that the channel is recovered
    @test γ == γ_copy

    # ---------------------------------------------------------------------------------------------------
    # Reduce 
    # ---------------------------------------------------------------------------------------------------
    # reduce! subtracts the lower-order asymptotic contributions from the higher-order classes in-place,
    # converting from "cumulative" to "pure" (reduced) asymptotic classes:
    #
    #   K2_reduced(Ω, ν)      = K2(Ω, ν) − K1(Ω)
    #   K3_reduced(Ω, ν, ν')  = K3(Ω, ν, ν') − K1(Ω) − K2(Ω, ν) − K2'(Ω, ν')
    #
    # After this operation, each class stores only the increment not captured by the lower classes.
    #
    # The asymptotic limits νInf correspond to 
    # ν → ∞ K2, K3 → 0
    # ν' → ∞: K2', K3 → 0
    # ν, ν' → ∞: K2, K2', K3 → 0
    # We verify that the evaluator γ(Ω, ν, ω) reproduces the pre-reduction values
    # for all combinations of finite and infinite fermionic arguments.

    # arbitrary in-bounds frequencies
    Ω = MatsubaraFrequency(T, 1, Boson)
    ν = MatsubaraFrequency(T, 2, Fermion)
    ω = MatsubaraFrequency(T, -1, Fermion)

    # channels for different limits with only some classes contributing
    x1 = γ(Ω, ν, ω; K1 = false, K2 = false, K3 = true)          # x1 = K3  
    x2 = γ(Ω, νInf, ω; K1 = false, K2 = true, K3 = false)       # x2 = K2'
    x3 = γ(Ω, ν, νInf; K1 = false, K2 = true, K3 = false)       # x3 = K2
    x4 = γ(Ω, νInf, νInf; K1 = true, K2 = false, K3 = false)    # x4 = K1

    # reduce channels
    # K2_red = K2 − K1
    # K3_red = K3 − K1 − K2_red − K2'_red
    fdDGAsolver.reduce!(γ)

    # ensure the correct reduced channels are recoverd
    # γ = K1 + K2_red + K2'_red + K3_red
    #   = K1 + (K2−K1) + (K2'−K1) + (K3−K1−(K2−K1)−(K2'−K1)) = K3
    @test γ(Ω, ν, ω) ≈ x1
    # γ = K1 + K2'_red
    #   = K1 + (K2'−K1) = K2'
    @test γ(Ω, νInf, ω) ≈ x2
    # γ = K1 + K2_red
    #   = K1 + (K2−K1) = K2
    @test γ(Ω, ν, νInf) ≈ x3
    # γ = K1
    @test γ(Ω, νInf, νInf) ≈ x4

    # ---------------------------------------------------------------------------------------------------
    # Evaluation 
    # ---------------------------------------------------------------------------------------------------
    # Verify the asymptotic decomposition γ(Ω,ν,ν') = K1[Ω] + K2[Ω,ν] + K2[Ω,ν'] + K3[Ω,ν,ν']
    # for two representations of νInf: a large-index MatsubaraFrequency (outside the grid)
    # and the fdDGAsolver.νInf (InfiniteMatsubaraFrequency). Both must give identical
    # results, confirming that out-of-grid frequencies are treated as ν → ∞.

    for νInf in [MatsubaraFrequency(T, 10^10, Fermion), fdDGAsolver.νInf]

        # pick arbitrary in-bounds frequencies (inside the smallest K3 grid)
        Ω  = value(meshes(γ.K3, Val(1))[5])
        ν1 = value(meshes(γ.K3, Val(2))[3])
        ν2 = value(meshes(γ.K3, Val(3))[4])

        # verify that correct limits are recovered for both vInf and out of bounds frequencies
        @test γ(Ω, ν1,   ν2)   ≈ γ.K1[Ω] + γ.K2[Ω, ν1] + γ.K2[Ω, ν2] + γ.K3[Ω, ν1, ν2]
        @test γ(Ω, νInf, νInf) ≈ γ.K1[Ω]
        @test γ(Ω, ν1,   νInf) ≈ γ.K1[Ω] + γ.K2[Ω, ν1]
        @test γ(Ω, νInf, ν2)   ≈ γ.K1[Ω] + γ.K2[Ω, ν2]
    end
    @test fdDGAsolver.νInf === fdDGAsolver.InfiniteMatsubaraFrequency()

    # K2 can be evaluated as a 2-argument function γ.K2(Ω, ω); this must equal the full
    # channel evaluator with K1 excluded and the second fermionic argument set to νInf,
    # i.e. γ(Ω, ω, νInf; K1 = false) = K2[Ω, ω]. Tested over a wide range of indices
    # including out-of-grid frequencies to exercise the boundary logic.
    for iΩ in [-100, -1, 0, 1, 100], iω in [-100, -1, 0, 1, 100]
        Ω = MatsubaraFrequency(T, iΩ, Boson);
        ω = MatsubaraFrequency(T, iω, Fermion);
        @test γ.K2(Ω, ω) ≈ γ(Ω, ω, νInf; K1 = false)
    end

    # ---------------------------------------------------------------------------------------------------
    # I/O 
    # ---------------------------------------------------------------------------------------------------

    testfile = dirname(@__FILE__) * "/test.h5"                      # create path to file
    file = h5open(testfile, "w")                                    # create and opens file in write mode               
    save!(file, "f", γ)                                             # save channel under label "f"
    close(file)                                                     # close file

    file = h5open(testfile, "r")                                    # open file in read-only mode
    γp = fdDGAsolver.load_channel(fdDGAsolver.Channel, file, "f")   # reconstruct channel in γp from file 
    
    # verify equality and type
    @test γ == γp
    @test γp isa fdDGAsolver.Channel

    close(file)                 # close file
    rm(testfile; force=true)    # delete testfile
end
