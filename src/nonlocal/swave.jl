# =============================================================================
# The s-wave approximation: a marker for "averaged over the Brillouin zone".
#
# On the lattice the reducible vertices depend on a bosonic momentum q and two
# fermionic momenta k, k′. In the s-wave approximation (Lihm 2025, Eqs.
# (S17)-(S18)) a vertex keeps only its dependence on q; the dependence on k and
# k′ is replaced by the Brillouin-zone average, e.g.
#
#   F_s-wave(q, ν, ν′) = 1/N_k² ∑_{k,k′} F(q, k, k′),   Π_s-wave(q, ν) = 1/N_k ∑_k Π(q, k).
#
# kSW is a value used in place of a momentum that means "averaged".
#   - Any momentum plus or minus kSW is again kSW: a combination that contains
#     an averaged momentum is averaged too.
#   - Indexing a mesh function with kSW in a momentum slot returns the average
#     over that slot (the getindex methods below).
# This makes channel conversions cheap. Example: the a-channel K1 needed at the
# p-channel momenta (P, k, kSW) has the bosonic momentum k - kSW = kSW. Thus
# ∑_q K1(k - q) / N_q becomes K1(kSW), one average over the stored array.
#
# The channel and vertex evaluators (nonlocal*/channel.jl, vertex.jl) use this.
# NL uses it for all classes, NL2 for K3 only, NL3 not at all.
# =============================================================================

"""
    SWaveBrillouinPoint

Type of the marker [`kSW`](@ref) for a momentum that is averaged over the
Brillouin zone.
"""
struct SWaveBrillouinPoint <: AbstractValue; end
"""
    kSW

A momentum averaged over the Brillouin zone (s-wave approximation, Lihm 2025,
Eq. (S18)). Indexing a lattice mesh function with `kSW` in a momentum slot
gives the average over that momentum, and any momentum combined with `kSW`
stays `kSW`.
"""
const kSW = SWaveBrillouinPoint()

# Folding back into the Brillouin zone keeps the marker.
MatsubaraFunctions.fold_back(:: SWaveBrillouinPoint, :: KMesh) = SWaveBrillouinPoint()
MatsubaraFunctions.mesh_index_bc(:: SWaveBrillouinPoint, :: KMesh) = SWaveBrillouinPoint()

# Any momentum plus or minus an averaged momentum is averaged.
Base.:+(::BrillouinPoint, ::SWaveBrillouinPoint) = SWaveBrillouinPoint()
Base.:-(::BrillouinPoint, ::SWaveBrillouinPoint) = SWaveBrillouinPoint()
Base.:+(::SWaveBrillouinPoint, ::BrillouinPoint) = SWaveBrillouinPoint()
Base.:-(::SWaveBrillouinPoint, ::BrillouinPoint) = SWaveBrillouinPoint()
Base.:+(::SWaveBrillouinPoint, ::SWaveBrillouinPoint) = SWaveBrillouinPoint()
Base.:-(::SWaveBrillouinPoint, ::SWaveBrillouinPoint) = SWaveBrillouinPoint()

# -----------------------------------------------------------------------------
# Indexing with kSW: Brillouin-zone averages
# -----------------------------------------------------------------------------

"""
    getindex(f, w…, kSW) -> value

Index a lattice mesh function with [`kSW`](@ref) in a momentum slot: the value
is the average over that momentum, `1/N ∑_k f(…, k)`. The other arguments
(frequencies, other momenta) select one entry each. Methods exist for
- `G(ν, k)` and `K1(Ω, q)`;
- `K2(Ω, ν, q)` and `K3(Ω, ν, ν′, q)` of the s-wave solver NL;
- `K2(Ω, ν, q, k)` and the bubble `Π(Ω, ν, q, k)` of NL2, with kSW in the last,
  second-to-last or both momentum slots;
- `K3(Ω, ν, ν′, q, k, k′)` of NL3, with both fermionic momenta averaged.
"""
function Base.getindex(
    f  :: Union{NL_MF_K1{Q}, NL_MF_G{Q}},
    w1 :: Union{MeshPoint, <: AbstractValue, Int, UnitRange, Colon},
       :: SWaveBrillouinPoint,
    )  :: Q where {Q}

    i1 = MatsubaraFunctions.mesh_index(w1, meshes(f, Val(1)))

    return sum(view(f, i1, :)) / length(meshes(f, Val(2)))

end

# K2(Ω, ν, kSW) of NL (also the NL bubble Π(Ω, ν, kSW): same type)
function Base.getindex(
    f  :: NL_MF_K2{Q},
    w1 :: Union{MeshPoint, <: AbstractValue, Int},
    w2 :: Union{MeshPoint, <: AbstractValue, Int},
       :: SWaveBrillouinPoint,
    )  :: Q where {Q}

    i1 = MatsubaraFunctions.mesh_index(w1, meshes(f, Val(1)))
    i2 = MatsubaraFunctions.mesh_index(w2, meshes(f, Val(2)))

    return sum(view(f, i1, i2, :)) / length(meshes(f, Val(3)))

end

# K3(Ω, ν, ν′, kSW) of NL and NL2
function Base.getindex(
    f  :: NL_MF_K3{Q},
    w1 :: Union{MeshPoint, <: AbstractValue, Int},
    w2 :: Union{MeshPoint, <: AbstractValue, Int},
    w3 :: Union{MeshPoint, <: AbstractValue, Int},
       :: SWaveBrillouinPoint,
    )  :: Q where {Q}

    i1 = MatsubaraFunctions.mesh_index(w1, meshes(f, Val(1)))
    i2 = MatsubaraFunctions.mesh_index(w2, meshes(f, Val(2)))
    i3 = MatsubaraFunctions.mesh_index(w3, meshes(f, Val(3)))

    return sum(view(f, i1, i2, i3, :)) / length(meshes(f, Val(4)))

end

# Π(Ω, ν, P, kSW) for an NL bubble. NOTE: NL_MF_Π has only three dimensions, so
# this four-index method cannot work; it is only reachable from commented-out
# code. (The NL2 bubble is handled by the NL2_MF_K2 methods below.)
function Base.getindex(
    f  :: NL_MF_Π{Q},
    w1 :: Union{MeshPoint, <: AbstractValue, Int},
    w2 :: Union{MeshPoint, <: AbstractValue, Int},
    P  :: Union{MeshPoint, <: AbstractValue, Int},
       :: SWaveBrillouinPoint,
    )  :: Q where {Q}

    i1 = MatsubaraFunctions.mesh_index(w1, meshes(f, Val(1)))
    i2 = MatsubaraFunctions.mesh_index(w2, meshes(f, Val(2)))
    i3 = MatsubaraFunctions.mesh_index(P,  meshes(f, Val(3)))

    return sum(view(f, i1, i2, i3, :)) / length(meshes(f, Val(4)))

end


# K2(Ω, ν, q, kSW) of NL2 (also the NL2 bubble Π(Ω, ν, q, kSW): same type):
# average over the fermionic momentum
function Base.getindex(
    f  :: NL2_MF_K2{Q},
    w1 :: Union{MeshPoint, <: AbstractValue, Int},
    w2 :: Union{MeshPoint, <: AbstractValue, Int},
    k1 :: Union{MeshPoint, <: AbstractValue, Int},
       :: SWaveBrillouinPoint,
    )  :: Q where {Q}

    i1 = MatsubaraFunctions.mesh_index(w1, meshes(f, Val(1)))
    i2 = MatsubaraFunctions.mesh_index(w2, meshes(f, Val(2)))
    i3 = MatsubaraFunctions.mesh_index(k1, meshes(f, Val(3)))

    return sum(view(f, i1, i2, i3, :)) / length(meshes(f, Val(4)))

end

# K2(Ω, ν, kSW, k) of NL2: average over the bosonic momentum
function Base.getindex(
    f  :: NL2_MF_K2{Q},
    w1 :: Union{MeshPoint, <: AbstractValue, Int},
    w2 :: Union{MeshPoint, <: AbstractValue, Int},
       :: SWaveBrillouinPoint,
    k2 :: Union{MeshPoint, <: AbstractValue, Int},
    )  :: Q where {Q}

    i1 = MatsubaraFunctions.mesh_index(w1, meshes(f, Val(1)))
    i2 = MatsubaraFunctions.mesh_index(w2, meshes(f, Val(2)))
    i4 = MatsubaraFunctions.mesh_index(k2, meshes(f, Val(4)))

    return sum(view(f, i1, i2, :, i4)) / length(meshes(f, Val(3)))

end

# K2(Ω, ν, kSW, kSW) of NL2: average over both momenta
function Base.getindex(
    f  :: NL2_MF_K2{Q},
    w1 :: Union{MeshPoint, <: AbstractValue, Int},
    w2 :: Union{MeshPoint, <: AbstractValue, Int},
       :: SWaveBrillouinPoint,
       :: SWaveBrillouinPoint,
    )  :: Q where {Q}

    i1 = MatsubaraFunctions.mesh_index(w1, meshes(f, Val(1)))
    i2 = MatsubaraFunctions.mesh_index(w2, meshes(f, Val(2)))

    return sum(view(f, i1, i2, :, :)) / length(meshes(f, Val(3))) / length(meshes(f, Val(4)))

end


# K3(Ω, ν, ν′, q, kSW, kSW) of NL3: average over both fermionic momenta
function Base.getindex(
    f  :: NL3_MF_K3{Q},
    w1 :: Union{MeshPoint, <: AbstractValue, Int},
    w2 :: Union{MeshPoint, <: AbstractValue, Int},
    w3 :: Union{MeshPoint, <: AbstractValue, Int},
    w4 :: Union{MeshPoint, <: AbstractValue, Int},
       :: SWaveBrillouinPoint,
       :: SWaveBrillouinPoint,
    )  :: Q where {Q}

    i1 = MatsubaraFunctions.mesh_index(w1, meshes(f, Val(1)))
    i2 = MatsubaraFunctions.mesh_index(w2, meshes(f, Val(2)))
    i3 = MatsubaraFunctions.mesh_index(w3, meshes(f, Val(3)))
    i4 = MatsubaraFunctions.mesh_index(w4, meshes(f, Val(4)))

    return sum(view(f, i1, i2, i3, i4, :, :)) / length(meshes(f, Val(5))) / length(meshes(f, Val(6)))

end
