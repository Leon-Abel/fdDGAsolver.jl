# bubble.jl
#
# Implements the calculation of particle-particle (pp) and 
# particle-hole (ph) bubble diagrams.

"""
    bubbles!(S::AbstractSolver)
 
Wrapper that extracts the particle-particle bubble (Πpp),
particle-hole bubble (Πph), and Green's function (G) from the solver and
calls the main bubbles! function.
"""
function bubbles!(S :: AbstractSolver)
    bubbles!(S.Πpp, S.Πph, S.G)

    # Optional symmetrization
    # S.SGΠpp(S.Πpp)
    # S.SGΠph(S.Πph)
end

"""
    bubbles!(Πpp::MF_Π{Q}, Πph::MF_Π{Q}, G::MF_G{Q}; use_G_tail::Bool=true)::Nothing where {Q}
 
Calculate particle-particle and particle-hole bubble diagrams.
 
This function computes two-particle bubble diagrams from the single-particle
Green's function G. The bubbles are defined as:

- Particle-particle (pp): Πpp(Ω,ν) = G(ν) * G(Ω - ν)
- Particle-hole (ph):     Πph(Ω,ν) = G(Ω + ν) * G(ν)

where Ω is the bosonic transfer frequency and ν is the fermionic frequency.
 
`use_G_tail::Bool=true`: If true, uses asymptotic tail (1/ν) for frequencies
outside the mesh range. If false, uses interpolation for all frequencies.
"""
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
            # For frequencies inside the mesh, use the actual G values
            # For frequencies outside, use the asymptotic form 1/ν
            G_ν   = is_inbounds(ν,     meshes(G, Val(1))) ? G[ν]     : 1 / value(ν)
            G_Ωmν = is_inbounds(Ω - ν, meshes(G, Val(1))) ? G[Ω - ν] : 1 / value(Ω - ν) # G(Ω - ν): needed for pp channel
            G_Ωpν = is_inbounds(Ω + ν, meshes(G, Val(1))) ? G[Ω + ν] : 1 / value(Ω + ν) # G(Ω + ν): needed for ph channel
        else
            # Use interpolation for all frequencies
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
