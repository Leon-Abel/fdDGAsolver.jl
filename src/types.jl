# types.jl
#
# Central type definitions:
#   - Diagrammatic channel tags (pCh, tCh, aCh) 
#   - Spin-component tags (pSp, xSp, dSp)
#   - Matsubara frequency and Brillouin-zone mesh aliases for local and nonlocal
#     MeshFunction types, covering the K1/K2/K3 asymptotic classes of the vertex.
#   - InfiniteMatsubaraFrequency sentinel type for handling the ν → ∞ asymptotics.
#   - k0, the Γ-point of the Brillouin zone.


# diagrammatic channels
#----------------------------------------------------------------------------------------------#
# In the parquet decomposition the full two-particle vertex is split as
#
#   Γ = Λ + Φ_p + Φ_t + Φ_a
#
# where Λ is the fully irreducible vertex and Φ_r is the reducible vertex in
# channel r ∈ {p, t, a}.
#
# The three channels are related by fermionic crossing symmetry.

"""
    abstract type ChannelTag end

Abstract type for diagrammatic channels
"""
abstract type ChannelTag end

"""
    struct pCh <: ChannelTag end

Parallel channel
"""
struct pCh <: ChannelTag end

"""
    struct tCh <: ChannelTag end

Transversal channel
"""
struct tCh <: ChannelTag end

"""
    struct aCh <: ChannelTag end

Antiparallel channel
"""
struct aCh <: ChannelTag end

# spin components
#----------------------------------------------------------------------------------------------#
# We define two spin components: parallel `pSp` and crossed `xSp`.
# All vertices are stored in the parallel spin component, and the crossed component is computed
# on the fly using the following crossing symmetry relations.
# ``Γp{xSp}(Ω, ν, ω) = -Γp{pSp}(Ω, ν, Ω - ω) = -Γp{pSp}(Ω, Ω - ν, ω)``
# ``Γt{xSp}(Ω, ν, ω) = -Γa{pSp}(Ω, ω, ν)``
# ``Γa{xSp}(Ω, ν, ω) = -Γt{pSp}(Ω, ω, ν)``

# We also define the density `dSp` component, and use it to simplify some evaluations.
# The density component is computed on the fly using the relation ``dSp = 2 * pSp + xSp``.
# Since the t-channel BSE is diagonal in the `xSp` and `dSp` channels but not in the
# `pSp` channel, we compute the t-channel BSE in `dSp`, and then subtract the `xSp` contribution
# (which is -1 times the `pSp` term in a channel by crossing symmetry) and divide by 2 to
# get the `pSp` component.

# The physical spin basis in terms of singlet (S), triplet (T), density (D), and magnetic (M)
# channels is given by
# ``S = pSp - xSp``
# ``T = pSp + xSp``
# ``D = 2 * pSp + xSp = dSp``
# ``M = xSp``

# pCh, pSp =  1/2 * (pCh, S) + 1/2 * (pCh, T)
# pCh, xSp = -1/2 * (pCh, S) + 1/2 * (pCh, T)
# pCh, dSp =  1/2 * (pCh, S) + 3/2 * (pCh, T)

# aCh, pSp = -1   * (tCh, M)
# aCh, xSp = -1/2 * (tCh, D) + 1/2 * (tCh, M)
# aCh, dSp = -1/2 * (tCh, D) - 3/2 * (tCh, M)

# tCh, pSp =  1/2 * (tCh, D) - 1/2 * (tCh, M)
# tCh, xSp = (tCh, M)
# tCh, dSp = (tCh, D)

"""
    abstract type SpinTag end

Abstract type for spin components
"""
abstract type SpinTag end

"""
    struct pSp <: SpinTag end

Parallel spin component
"""
struct pSp <: SpinTag end

"""
    struct xSp <: SpinTag end

Crossed spin component
"""
struct xSp <: SpinTag end

"""
    struct dSp <: SpinTag end

Density component
"""
struct dSp <: SpinTag end

# Mesh and MeshFunction defined in mesh_generalization_v3 branch of MatsubaraFunctions.jl package

# Mesh aliases
#----------------------------------------------------------------------------------------------#
"""
Fermionic Matsubara frequency mesh, with frequencies iν_n = iπ(2n+1)/β.
"""
const FMesh = Mesh{MeshPoint{MatsubaraFrequency{Fermion}}, MatsubaraDomain}
"""
Bosonic Matsubara frequency mesh, with frequencies iΩ_n = i2πn/β.
"""
const BMesh = Mesh{MeshPoint{MatsubaraFrequency{Boson}}, MatsubaraDomain}
"""
2D Brillouin zone momentum-mesh with 4-fold (C4) point-group symmetry
"""
const KMesh = Mesh{MeshPoint{BrillouinPoint{2}}, BrillouinDomain{2, 4}}

# MeshFunction aliases : local
#----------------------------------------------------------------------------------------------#
# Local (momentum-independent / DMFT-like) mesh functions on Matsubara frequency meshes.

"""
Local single-particle Green's function `G(iν)`: one fermionic frequency.
"""
const MF_G{Q}  = MeshFunction{1, Q, Tuple{FMesh}, Array{Q, 1}}
"""
Local two-particle bubble `Π(iΩ, iν)`: bosonic frequency, one fermionic frequency.
"""
const MF_Π{Q}  = MeshFunction{2, Q, Tuple{BMesh, FMesh}, Array{Q, 2}}
"""
Local K1 vertex class `K1(iΩ)`: bosonic frequency.
"""
const MF_K1{Q} = MeshFunction{1, Q, Tuple{BMesh}, Array{Q, 1}}
"""
Local K2 vertex class `K2(iΩ, iν)`: bosonic frequency, one fermionic frequency.
"""
const MF_K2{Q} = MeshFunction{2, Q, Tuple{BMesh, FMesh}, Array{Q, 2}}
"""
Local K3 vertex class `K3(iΩ, iν, iω)`: full frequency dependence.
"""
const MF_K3{Q} = MeshFunction{3, Q, Tuple{BMesh, FMesh, FMesh}, Array{Q, 3}}

# MeshFunction aliases : nonlocal Green's function and bubble
#----------------------------------------------------------------------------------------------#
# Analogues of the local objects above, extended with a momentum argument k
# on the Brillouin zone mesh.

"""
Nonlocal single-particle Green's function `G(iν, k)`: one fermionic frequency, momentum.
"""
const NL_MF_G{Q} = MeshFunction{2, Q, Tuple{FMesh, KMesh}, Array{Q, 2}}
"""
Nonlocal two-particle bubble `Π(iΩ, iν, q)`: bosonic frequency, one fermionic frequency, transfer momentum.
"""
const NL_MF_Π{Q} = MeshFunction{3, Q, Tuple{BMesh, FMesh, KMesh}, Array{Q, 3}}

# MeshFunction aliases : nonlocal vertex with only bosonic frequency dependence
#----------------------------------------------------------------------------------------------#
# Nonlocal K1/K2/K3 vertex classes with a *single* momentum argument q on the
# bosonic (transfer) channel. 

"""
Nonlocal K1 vertex class `K1(iΩ, q)`: bosonic frequency, transfer momentum only.
"""
const NL_MF_K1{Q} = MeshFunction{2, Q, Tuple{BMesh, KMesh}, Array{Q, 2}}
"""
Nonlocal K2 vertex class `K2(iΩ, iν, q)`: bosonic frequency, one fermionic frequency, transfer momentum only.
"""
const NL_MF_K2{Q} = MeshFunction{3, Q, Tuple{BMesh, FMesh, KMesh}, Array{Q, 3}}
"""
Nonlocal K3 vertex class `K3(iΩ, iν, iω, q)`: full frequency dependence, transfer momentum only.
"""
const NL_MF_K3{Q} = MeshFunction{4, Q, Tuple{BMesh, FMesh, FMesh, KMesh}, Array{Q, 4}}

# MeshFunction aliases : nonlocal vertex with bosonic and fermionic frequency dependences
#----------------------------------------------------------------------------------------------#
# Fully nonlocal objects where momenta on *both* fermionic legs are resolved.

"""
Fully nonlocal bubble `Π(iΩ, iν, k, k')`: bosonic frequency, one fermionic frequency, two independent momenta.
"""
const NL2_MF_Π{Q} = MeshFunction{4, Q, Tuple{BMesh, FMesh, KMesh, KMesh}, Array{Q, 4}}
"""
Fully nonlocal K2 vertex `K2(iΩ, iν, k, k')`: bosonic frequency, one fermionic frequency, independent momenta.
"""
const NL2_MF_K2{Q} = MeshFunction{4, Q, Tuple{BMesh, FMesh, KMesh, KMesh}, Array{Q, 4}}
"""
Fully nonlocal K3 vertex `K3(iΩ, iν, iω, k, k', k'')`: full frequency and full momentum dependence.
"""
const NL3_MF_K3{Q} = MeshFunction{6, Q, Tuple{BMesh, FMesh, FMesh, KMesh, KMesh, KMesh}, Array{Q, 6}}

# struct to describe high-frequency limit
#----------------------------------------------------------------------------------------------#

struct InfiniteMatsubaraFrequency <: AbstractValue end
const νInf = InfiniteMatsubaraFrequency()

Base.:+(::MatsubaraFrequency, ::InfiniteMatsubaraFrequency) = InfiniteMatsubaraFrequency()
Base.:-(::MatsubaraFrequency, ::InfiniteMatsubaraFrequency) = InfiniteMatsubaraFrequency()
Base.:+(::InfiniteMatsubaraFrequency, ::MatsubaraFrequency) = InfiniteMatsubaraFrequency()
Base.:-(::InfiniteMatsubaraFrequency, ::MatsubaraFrequency) = InfiniteMatsubaraFrequency()
Base.:+(::InfiniteMatsubaraFrequency, ::InfiniteMatsubaraFrequency) = InfiniteMatsubaraFrequency()
Base.:-(::InfiniteMatsubaraFrequency, ::InfiniteMatsubaraFrequency) = InfiniteMatsubaraFrequency()

MatsubaraFunctions.is_inbounds(::InfiniteMatsubaraFrequency, ::Mesh) = false
MatsubaraFunctions.is_inbounds_bc(::InfiniteMatsubaraFrequency, ::Mesh) = false

# origin of the Brillouin zone
#----------------------------------------------------------------------------------------------#

const k0 = BrillouinPoint(0, 0)
