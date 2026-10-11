# =============================================================================
# Bare propagator for a flow from the DMFT impurity to the lattice.
#
# The bare propagator depends on a flow parameter Λ:
#
#   G0_Λ(k, ν) = Θ_Λ(ν) G0_lat(k, ν) + Ξ_Λ(ν) G0_imp(ν),   Θ_Λ(ν) = ν² / (ν² + Λ²)
#
# Θ_Λ is a frequency regulator (as in the Ω-flow of functional RG): it
# suppresses the lattice propagator at |ν| below Λ. The local term
# Ξ_Λ(ν) G0_imp(ν) is fixed at each ν so that the DMFT self-consistency holds
# for every Λ: with the impurity self-energy, the local lattice propagator
# equals the impurity one,
#
#   1/N_k ∑_k [G0_Λ(k, ν)⁻¹ - Σ_imp(ν)]⁻¹ = G_imp(ν).
#
# At Λ = ∞ (Θ = 0) this gives Ξ = 1, i.e. G0_Λ = G0_imp, the impurity problem.
# At Λ = 0 (Θ = 1) it gives Ξ = 0, i.e. the lattice, provided the DMFT input
# is self-consistent. A flow in Λ thus starts from the DMFT solution and ends
# at the lattice model. Used in the tests (test_flow.jl).
#
# Propagators and Σ are stored as iG and iΣ, so the Dyson equation in the code
# reads iG = [(iG0)⁻¹ + iΣ]⁻¹ (dyson.jl).
# =============================================================================

"""
    bare_Green_Ω_flow(Λ, G0_imp, Σ_imp, G0_lat) => G0_Λ

Compute the bare Green function with the flow parameter `Λ`. `Λ = ∞` gives the impurity
Green function `G0_imp`, and `Λ = 0` gives the lattice Green function `G0_lat`.

DMFT self-consistency ``∫dk (G0_Λ(k, ν)⁻¹ - Σ_imp(ν))⁻¹ = G_imp(ν)`` is enforced.

`G0_imp` and `Σ_imp` are local, `G0_lat` lives on the lattice momentum mesh,
and the output `G0_Λ` has the mesh of `G0_lat` (all stored with a factor i).
At each frequency the complex weight `Ξ_Λ(ν)` of the local term is found with
a nonlinear solver (see the file header). The result is checked: `G0_Λ` with
`Σ_imp` at every k must reproduce `G_imp` locally to 1e-10.
"""
function bare_Green_Ω_flow(
    Λ      :: Float64,
    G0_imp :: MF_G{Q},
    Σ_imp  :: MF_G{Q},
    G0_lat :: NL_MF_G{Q}
    ) :: NL_MF_G{Q} where {Q}

    mG = meshes(G0_lat, Val(1))
    mK = meshes(G0_lat, Val(2))

    # impurity propagator G_imp from Dyson's equation
    G_imp = copy(G0_imp)
    Dyson!(G_imp, Σ_imp, G0_imp)

    G0_Λ = copy(G0_lat)
    set!(G0_Λ, 0)

    for iν in eachindex(mG)
        ν = value(mG[iν])
        # regulator Θ_Λ(ν) = ν² / (ν² + Λ²)
        Θ = value(ν)^2 / (value(ν)^2 + Λ^2)

        # residual of the DMFT self-consistency for a trial weight Ξ:
        # local part of [G0_Λ⁻¹ - Σ_imp]⁻¹ minus G_imp (in the iG convention)
        function _dmft_self_consistency(Ξ)
            G0_Λ_ν = @. Θ * G0_lat[ν, :] + Ξ * G0_imp[ν]
            return sum(@. 1 / (1 / G0_Λ_ν + Σ_imp[ν])) / length(mK) - G_imp[ν]
        end

        # solve for the complex Ξ (real and imaginary part as two unknowns)
        res = nlsolve(x -> (y = _dmft_self_consistency(complex(x[1], x[2]));
            [real(y), imag(y)]), [0.0, 0.0], show_trace = false, ftol = 1e-12)
        Ξ = complex(res.zero...)

        G0_Λ.data[iν, :] .= @. Θ * G0_lat[ν, :] + Ξ * G0_imp[ν]
    end

    # Sanity check of DMFT self-consistency
    # (G_Λ from G0_Λ and Σ_imp at every k; its local part must equal G_imp)
    Σ_lat = copy(G0_Λ)
    G_Λ = copy(G0_Λ)
    set!(Σ_lat, 0)
    for iν in eachindex(mG)
        ν = value(mG[iν])
        Σ_lat.data[iν, :] .= Σ_imp(ν)
    end

    Dyson!(G_Λ, Σ_lat, G0_Λ)
    y1 = [G_imp[value(x)] for x in mG]
    y2 = sum(G_Λ.data, dims=2) ./ length(mK)
    @assert maximum(abs.(y1 .- y2)) < 1e-10
    # @info "L2  error = $(norm(y1 .- y2))"
    # @info "Max error = $(maximum(abs.(y1 .- y2)))"

    return G0_Λ
end
