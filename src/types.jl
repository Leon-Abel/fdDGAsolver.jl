# diagrammatic channels
#----------------------------------------------------------------------------------------------#

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
# FMesh : Fermionic Matsubara frequency mesh, with frequencies iν_n = iπ(2n+1)/β.
#
# BMesh : Bosonic Matsubara frequency mesh, with frequencies iΩ_n = i2πn/β.
#         Used for transfer frequencies of two-particle objects (bubbles and vertices).
#
# KMesh : 2D Brillouin zone mesh with 4-fold (C4) point-group symmetry, used for
#         the momentum dependence of nonlocal Green's functions and vertices.

const FMesh = Mesh{MeshPoint{MatsubaraFrequency{Fermion}}, MatsubaraDomain}
const BMesh = Mesh{MeshPoint{MatsubaraFrequency{Boson}}, MatsubaraDomain}
const KMesh = Mesh{MeshPoint{BrillouinPoint{2}}, BrillouinDomain{2, 4}}

# MeshFunction aliases : local
#----------------------------------------------------------------------------------------------#
# Local (momentum-independent / DMFT-like) mesh functions on Matsubara frequency meshes.
#
# MF_G  : Single-particle Green's function G(iν), defined on a single fermionic frequency mesh.
#
# MF_Π  : Two-particle bubble Π(iΩ, iν), defined on a bosonic and a fermionic frequency mesh.
#         The bosonic frequency iΩ is the transfer frequency; iν is the internal loop frequency.
#
# MF_K1 : Asymptotic vertex class K1(iΩ). Depends only on the bosonic transfer frequency iΩ.
#
# MF_K2 : Asymptotic vertex class K2(iΩ, iν). Depends on the bosonic transfer frequency iΩ
#         and one fermionic frequency iν.
#
# MF_K3 : Full vertex class K3(iΩ, iν, iν'). Depends on the bosonic transfer frequency iΩ
#         and both fermionic frequencies iν, iν'.

const MF_G{Q}  = MeshFunction{1, Q, Tuple{FMesh}, Array{Q, 1}}
const MF_Π{Q}  = MeshFunction{2, Q, Tuple{BMesh, FMesh}, Array{Q, 2}}
const MF_K1{Q} = MeshFunction{1, Q, Tuple{BMesh}, Array{Q, 1}}
const MF_K2{Q} = MeshFunction{2, Q, Tuple{BMesh, FMesh}, Array{Q, 2}}
const MF_K3{Q} = MeshFunction{3, Q, Tuple{BMesh, FMesh, FMesh}, Array{Q, 3}}

# MeshFunction aliases : nonlocal Green's function and bubble
#----------------------------------------------------------------------------------------------#
# Nonlocal extensions of MF_G and MF_Π, adding a single momentum dependence.
#
# NL_MF_G : Single-particle Green's function G(iν, k), with explicit momentum k.
#
# NL_MF_Π : Two-particle bubble Π(iΩ, iν, q), with explicit transfer momentum q.


const NL_MF_G{Q} = MeshFunction{2, Q, Tuple{FMesh, KMesh}, Array{Q, 2}}
const NL_MF_Π{Q} = MeshFunction{3, Q, Tuple{BMesh, FMesh, KMesh}, Array{Q, 3}}

# MeshFunction aliases : nonlocal vertex with only bosonic frequency dependence
#----------------------------------------------------------------------------------------------#
# Nonlocal extensions of the asymptotic vertex classes, adding a transfer momentum q.
# The fermionic momentum dependence of K1 and K2 is neglected (single-boson exchange), 
# retaining only the bosonic transfer momentum q.
#
# NL_MF_K1 : K1(iΩ, q)
# NL_MF_K2 : K2(iΩ, iν, q)
# NL_MF_K3 : K3(iΩ, iν, iν', q)

const NL_MF_K1{Q} = MeshFunction{2, Q, Tuple{BMesh, KMesh}, Array{Q, 2}}
const NL_MF_K2{Q} = MeshFunction{3, Q, Tuple{BMesh, FMesh, KMesh}, Array{Q, 3}}
const NL_MF_K3{Q} = MeshFunction{4, Q, Tuple{BMesh, FMesh, FMesh, KMesh}, Array{Q, 4}}

# MeshFunction aliases : nonlocal vertex with bosonic and fermionic frequency dependences
#----------------------------------------------------------------------------------------------#
# Fully nonlocal vertex classes, with explicit fermionic momenta k, k' in addition to
# the bosonic transfer momentum q.
#
# NL2_MF_Π  : Π(iΩ, iν, q, k)
# NL2_MF_K2 : K2(iΩ, iν, q, k)
# NL3_MF_K3 : K3(iΩ, iν, iν', q, k, k')

const NL2_MF_Π{Q} = MeshFunction{4, Q, Tuple{BMesh, FMesh, KMesh, KMesh}, Array{Q, 4}}
const NL2_MF_K2{Q} = MeshFunction{4, Q, Tuple{BMesh, FMesh, KMesh, KMesh}, Array{Q, 4}}
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
