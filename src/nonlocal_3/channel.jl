# =============================================================================
# One channel's reducible vertex Φ_r on the lattice, solver NL3.
#
# NL3 keeps all momenta of the asymptotic classes (the paper's "Full"
# treatment, as opposed to the s-wave "s-K23" of NL):
#
#   Φ_r(Ω, ν, ν′; q, k, k′) = K1(Ω; q) + K2(Ω, ν; q, k) + K̄2(Ω, ν′; q, k′)
#                             + K3(Ω, ν, ν′; q, k, k′)
#
# (Lihm 2025, Eq. (S7) with momenta). K1 and K2 are stored as in NL2; K3 is an
# array over three frequencies and three momenta. A momentum kSW (averaged)
# gives the Brillouin-zone average (nonlocal/swave.jl).
#
# Naming in the evaluator: its arguments are (Ω, ν, ω, P, k, q), with ω = ν′ and
# q = k′ (the paper's k′, not the paper's bosonic q, which is P here).
# =============================================================================

"""
    NL3_Channel{Q} <: AbstractReducibleVertex{Q}

The reducible vertex `Φ_r` of one channel for the solver NL3: `K1(Ω, q)`,
`K2(Ω, ν, q, k)` and `K3(Ω, ν, ν′, q, k, k′)`.

Constructors: `NL3_Channel(K1, K2, K3)`, or
`NL3_Channel(T, numK1, numK2, numK3, meshK, Q = ComplexF64)` with all classes
zero on the momentum mesh `meshK` (used for all momenta).
"""
struct NL3_Channel{Q <: Number} <: AbstractReducibleVertex{Q}
    K1 :: NL_MF_K1{Q}
    K2 :: NL2_MF_K2{Q}
    K3 :: NL3_MF_K3{Q}

    # from existing mesh functions
    function NL3_Channel(K1 :: NL_MF_K1{Q}, K2 :: NL2_MF_K2{Q}, K3 :: NL3_MF_K3{Q}) where {Q}
        return new{Q}(K1, K2, K3)
    end

    # zero classes on the given frequency boxes and momentum mesh
    function NL3_Channel(
        T     :: Float64,
        numK1 :: Int64,
        numK2 :: NTuple{2, Int64},
        numK3 :: NTuple{2, Int64},
        meshK :: KMesh,
              :: Type{Q} = ComplexF64
        ) where {Q}

        # K1(Ω, q)
        mK1Ω = MatsubaraMesh(T, numK1, Boson)
        K1   = MeshFunction(mK1Ω, meshK; data_t = Q)
        set!(K1, 0)

        # K2(Ω, ν, q, k)
        @assert numK1 >= numK2[1] "No. bosonic frequencies in K1 must be larger than no. bosonic frequencies in K2"
        @assert numK1 >= numK2[2] "No. bosonic frequencies in K1 must be larger than no. fermionic frequencies in K2"
        mK2Ω = MatsubaraMesh(T, numK2[1], Boson)
        mK2ν = MatsubaraMesh(T, numK2[2], Fermion)
        K2   = MeshFunction(mK2Ω, mK2ν, meshK, meshK; data_t = Q)
        set!(K2, 0)

        # K3(Ω, ν, ν′, q, k, k′)
        @assert all(numK2 .>= numK3) "Number of frequencies in K2 must be larger than in K3"
        mK3Ω = MatsubaraMesh(T, numK3[1], Boson)
        mK3ν = MatsubaraMesh(T, numK3[2], Fermion)
        K3   = MeshFunction(mK3Ω, mK3ν, mK3ν, meshK, meshK, meshK; data_t = Q)
        set!(K3, 0)

        return new{Q}(K1, K2, K3)
    end
end

# -----------------------------------------------------------------------------
# Getters and copy
# -----------------------------------------------------------------------------

"""
    get_P_mesh(γ::NL3_Channel) -> KMesh

Mesh of the bosonic momentum q (code: P); the fermionic momenta use the same
mesh.
"""
function get_P_mesh(γ :: NL3_Channel) :: KMesh
    return meshes(γ.K1, Val(2))
end

"""
    numP(γ::NL3_Channel) -> Int64

Number of points of the momentum mesh.
"""
function numP(γ :: NL3_Channel) :: Int64
    return length(meshes(γ.K1, Val(2)))
end

"""
    copy(γ::NL3_Channel) -> NL3_Channel

Independent copy, with new arrays for K1, K2 and K3.
"""
function Base.:copy(γ :: NL3_Channel{Q}) :: NL3_Channel{Q} where {Q}
    return NL3_Channel(copy(γ.K1), copy(γ.K2), copy(γ.K3))
end

# -----------------------------------------------------------------------------
# Evaluation
# -----------------------------------------------------------------------------

"""
    (γ::NL3_Channel)(Ω, ν, ω, P, k, q; K1 = true, K2 = true, K3 = true) -> value

Evaluate `Φ_r` in the channel's own parametrization, with bosonic momentum `P`
and fermionic momenta `k` (left leg) and `q` (right leg; the paper's k′):
`K1(Ω, P) + K2(Ω, ν, P, k) + K̄2(Ω, ω, P, q) + K3(Ω, ν, ω, P, k, q)`, where `ω`
is the second fermionic frequency ν′. Classes outside their box give zero;
`ν` or `ω` = `νInf` is never inside a box, so this one method also gives the
high-frequency limits. The keywords switch classes off, and a momentum `kSW`
gives the average over it.
"""
@inline function (γ :: NL3_Channel{Q})(
    Ω  :: MatsubaraFrequency,
    ν  :: Union{MatsubaraFrequency, InfiniteMatsubaraFrequency},
    ω  :: Union{MatsubaraFrequency, InfiniteMatsubaraFrequency},
    P  :: Union{BrillouinPoint, SWaveBrillouinPoint},
    k  :: Union{BrillouinPoint, SWaveBrillouinPoint},
    q  :: Union{BrillouinPoint, SWaveBrillouinPoint},
    ;
    K1 :: Bool = true,
    K2 :: Bool = true,
    K3 :: Bool = true,
    )  :: Q where {Q}

    val = zero(Q)
    # mesh indices of the momenta (folded back; kSW stays kSW)
    iP = MatsubaraFunctions.mesh_index_bc(P, get_P_mesh(γ))
    ik = MatsubaraFunctions.mesh_index_bc(k, get_P_mesh(γ))
    iq = MatsubaraFunctions.mesh_index_bc(q, get_P_mesh(γ))

    if is_inbounds(Ω, meshes(γ.K1, Val(1)))
        K1 && (val += γ.K1[Ω, iP])

        if is_inbounds(Ω, meshes(γ.K2, Val(1)))
            ν1_inbounds = is_inbounds(ν, meshes(γ.K2, Val(2)))
            ν2_inbounds = is_inbounds(ω, meshes(γ.K2, Val(2)))

            if ν1_inbounds && ν2_inbounds
                # K2(Ω, ν; P, k) + K̄2(Ω, ν′; P, k′), and K3 if inside its box
                K2 && (val += γ.K2[Ω, ν, iP, ik] + γ.K2[Ω, ω, iP, iq])

                if (K3 && is_inbounds(Ω, meshes(γ.K3, Val(1))) &&
                          is_inbounds(ν, meshes(γ.K3, Val(2))) &&
                          is_inbounds(ω, meshes(γ.K3, Val(3))))
                    val += γ.K3[Ω, ν, ω, iP, ik, iq]
                end

            elseif ν1_inbounds
                K2 && (val += γ.K2[Ω, ν, iP, ik])

            elseif ν2_inbounds
                K2 && (val += γ.K2[Ω, ω, iP, iq])
            end
        end
    end

    return val
end
