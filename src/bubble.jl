# =============================================================================
# Two-particle bubbles of the local solver.
#
# The bubble is the pair of propagators between two vertices in a
# Bethe-Salpeter equation (Lihm 2025, Eq. (S8)):
#
#   Π_p(Ω, ν) = G(ν) G(Ω - ν)    particle-particle bubble (p channel), code: Πpp
#   Π_a(Ω, ν) = Π_t(Ω, ν)
#             = G(ν) G(Ω + ν)    particle-hole bubble (a and t channels), code: Πph
#
# Ω is the bosonic frequency of the channel and ν the fermionic frequency that
# is summed over in the BSE. Since G is stored as iG (dyson.jl), the stored
# bubble is (iG)(iG) = -GG; the signs in the BSEs account for this
# (BSE_templates.jl).
#
# The target bubbles Πpp, Πph are rebuilt from the target G after every Dyson
# step. The reference bubbles Π0pp, Π0ph are built once from the reference
# propagator, in the solver constructor (ParquetSolver.jl).
# =============================================================================

"""
    bubbles!(S::AbstractSolver)
    bubbles!(Πpp, Πph, G; use_G_tail = true) -> nothing

Compute the particle-particle and particle-hole bubbles from the propagator G
(stored as iG) and write them into `Πpp` and `Πph` (Lihm 2025, Eq. (S8)):

    Πpp(Ω, ν) = G(ν) G(Ω - ν),    Πph(Ω, ν) = G(ν) G(Ω + ν)

The bubble mesh can reach beyond the G mesh. With `use_G_tail = true`, a G
outside its mesh is replaced by its high-frequency tail `iG(iν) ≈ 1/ν`. With
`false` the G mesh function is called directly, which gives zero outside its
mesh.

The solver method updates the target bubbles `S.Πpp`, `S.Πph` from `S.G`.
"""
function bubbles!(S :: AbstractSolver)
    bubbles!(S.Πpp, S.Πph, S.G)

    # Optional symmetrization
    # S.SGΠpp(S.Πpp)
    # S.SGΠph(S.Πph)
end

# Array method of bubbles!, see the docstring above.
function bubbles!(
    Πpp :: MF_Π{Q},
    Πph :: MF_Π{Q},
    G   :: MF_G{Q},
    ;
    use_G_tail :: Bool = true # Use asymptotic tail for out-of-bounds frequencies
    ) :: Nothing where {Q}

    for iΩ in eachindex(meshes(Πpp, Val(1))), iν in eachindex(meshes(Πpp, Val(2)))
        # frequency values from mesh indices
        Ω = value(meshes(Πpp, Val(1))[iΩ]) # Bosonic frequency
        ν = value(meshes(Πpp, Val(2))[iν]) # Fermionic frequency

        # particle-particle channel: Πpp[iΩ, iν] = G(ν) * G(Ω - ν)
        # particle-hole channel:     Πph[iΩ, iν] = G(Ω + ν) * G(ν)
        if use_G_tail
            # Inside the G mesh use the stored values, outside the tail
            # iG(iν) ≈ i/(iν) = 1/ν
            G_ν   = is_inbounds(ν,     meshes(G, Val(1))) ? G[ν]     : 1 / value(ν)
            G_Ωmν = is_inbounds(Ω - ν, meshes(G, Val(1))) ? G[Ω - ν] : 1 / value(Ω - ν) # G(Ω - ν): needed for pp channel
            G_Ωpν = is_inbounds(Ω + ν, meshes(G, Val(1))) ? G[Ω + ν] : 1 / value(Ω + ν) # G(Ω + ν): needed for ph channel
        else
            # Evaluate the mesh function directly at all frequencies
            G_ν   = G(ν)
            G_Ωmν = G(Ω - ν)
            G_Ωpν = G(Ω + ν)
        end

        # bubble diagrams as products of Green's functions
        Πpp[iΩ, iν] = G_ν * G_Ωmν
        Πph[iΩ, iν] = G_ν * G_Ωpν
    end

    return nothing
end
