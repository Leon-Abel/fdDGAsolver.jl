"""
    hubbard_bare_Green(mesh_ν::FMesh, mesh_k::KMesh, ::Type{Q} = ComplexF64; μ, t1, t2 = 0., t3 = 0.)

Bare (non-interacting) Green function of the 2D square lattice Hubbard model in
momentum-frequency space, defined as
 
    G₀(k, iν) = 1 / (iν + μ - ε(k))
 
- `μ`  : Chemical potential. For nearest-neighbor hopping only (t2=t3=0), μ=0
         corresponds to half filling (one electron per site) due to particle-hole
         symmetry of the square lattice.
- `t1` : First nearest-neighbor (NN) hopping amplitude.
- `t2` : Second nearest-neighbor (NNN) hopping amplitude.
- `t3` : Third nearest-neighbor (3NN) hopping amplitude.

The result is stored as `im * G₀` (i.e., the function returns `im * G₀`) to make
the stored quantity purely real at half filling, which simplifies symmetry enforcement
and numerical operations throughout the solver.
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
        # Obtain the Cartesian components of the BZ wavevector k = (k1, k2)
        k1, k2 = euclidean(k, mesh_k)

        # Evaluate the non-interacting dispersion ε(k) for this momentum point.
        εk = hubbard_band(k1, k2; t1, t2, t3)
        for ν in mesh_ν

            # extract the real frequency value iν_n = 2π(n + 1/2) * T as a Float.
            ν_value = plain_value(ν)

            # Bare Green function: G₀(k, iν) = 1 / (iν + μ - ε(k)).
            # Store im * G₀ so that the stored object is real.
            G0[ν, k] = 1 / (im * ν_value + μ - εk) * im
        end
    end

    G0 :: NL_MF_G{Q}
end

"""
    hubbard_band(k1::Float64, k2::Float64; t1, t2=0., t3=0.) :: Float64
 
Single-particle dispersion relation ε(k) of the 2D square lattice Hubbard model,
including up to third nearest-neighbor hopping:
 
    ε(k) = -2t₁[cos(k₁) + cos(k₂)] - 4t₂ cos(k₁)cos(k₂) - 2t₃[cos(2k₁) + cos(2k₂)]
 
- `k1`, `k2` : Cartesian components of the crystal momentum..
- `t1`       : First nearest-neighbor hopping.
- `t2`       : Second nearest-neighbor hopping. Default: 0.
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
