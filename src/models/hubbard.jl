# =============================================================================
# Bare propagator and band of the Hubbard model on the square lattice.
#
# Lihm 2025, Eq. (7): H = -∑_{ij,σ} (t_ij + μ δ_ij) c†_iσ c_jσ + U ∑_i n_i↑ n_i↓,
# with hopping t1 = t to nearest, t2 = t′ to next-nearest and t3 = t″ to
# third-nearest neighbors. The bare propagator G₀(k, iν) = 1/(iν + μ - ε(k))
# contains only the hopping and μ; the interaction U enters through the vertex.
#
# Conventions: the propagator is stored multiplied by i (see dyson.jl). The
# self-energy is stored without its half-filling Hartree value U/2, so the μ
# used here is the paper's μ minus U/2. With t2 = t3 = 0, μ = 0 is half
# filling.
# =============================================================================

"""
    hubbard_bare_Green(mesh_ν, mesh_k, Q = ComplexF64; μ, t1, t2 = 0, t3 = 0) -> MeshFunction

Bare propagator of the square-lattice Hubbard model on the fermionic Matsubara
mesh `mesh_ν` and the momentum mesh `mesh_k`, stored as `iG₀(iν, k)` with

    G₀(iν, k) = 1 / (iν + μ - ε(k))

and the band `ε(k)` from [`hubbard_band`](@ref).

- `μ`: chemical potential, without the Hartree term U/2 (see the file header).
  For t2 = t3 = 0, `μ = 0` is half filling (one electron per site), by the
  particle-hole symmetry of the nearest-neighbor square lattice.
- `t1`, `t2`, `t3`: hopping to the first, second and third neighbors
  (Lihm 2025: t1 = t = 1 sets the energy unit, t2 = t′).

Storing iG₀ instead of G₀ makes the stored values real at half filling.
"""
function hubbard_bare_Green(
    mesh_ν :: FMesh,
    mesh_k :: KMesh,
           :: Type{Q} = ComplexF64,
    ;
    μ, t1, t2 = 0., t3 = 0.,
    ) where {Q}

    G0 = MeshFunction(mesh_ν, mesh_k; data_t = Q)

    for k in mesh_k
        # Cartesian components of the momentum k = (k1, k2)
        k1, k2 = euclidean(k, mesh_k)

        # band energy ε(k) at this momentum
        εk = hubbard_band(k1, k2; t1, t2, t3)
        for ν in mesh_ν

            # the real number ν_n = πT(2n + 1)
            ν_value = plain_value(ν)

            # Bare Green function: G₀(k, iν) = 1 / (iν + μ - ε(k)).
            # Store im * G₀ so that the stored object is real at half filling.
            G0[ν, k] = 1 / (im * ν_value + μ - εk) * im
        end
    end

    G0 :: NL_MF_G{Q}
end

"""
    hubbard_band(k1, k2; t1, t2 = 0, t3 = 0) -> ε

Band energy of the square lattice with hopping to the first three neighbor
shells, at the momentum `(k1, k2)` (Cartesian components):

    ε(k) = -2t1 [cos(k1) + cos(k2)] - 4t2 cos(k1) cos(k2) - 2t3 [cos(2k1) + cos(2k2)]
"""
function hubbard_band(
    k1 :: Float64,
    k2 :: Float64
    ;
    t1 :: Float64,
    t2 :: Float64 = 0.,
    t3 :: Float64 = 0.,
    ) :: Float64

    # Nearest-neighbor (NN) contribution
    εk  = -2 * t1 * (cos(k1) + cos(k2))

    # Next-nearest-neighbor (NNN) contribution
    εk += -4 * t2 * cos(k1) * cos(k2)

    # Third nearest-neighbor (3NN) contribution
    εk += -2 * t3 * (cos(2k1) + cos(2k2))
    return εk
end

# make hubbard_bare_Green() publicly available to package
export hubbard_bare_Green
