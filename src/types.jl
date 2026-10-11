# =============================================================================
# Basic types: channel and spin tags, mesh and mesh-function aliases, and the
# marker for an infinite frequency.
#
# The parquet decomposition (Lihm 2025, Eq. (1)) writes the full two-particle
# vertex as
#
#   F = Λ + Φ_a + Φ_p + Φ_t
#
# with Λ fully two-particle irreducible and Φ_r reducible in channel r. The tags
# in this file name those channels (a, p, t) and the spin components a vertex is
# evaluated in. The mesh-function aliases fix how each object is stored: which
# Matsubara frequencies and, on the lattice, which momenta it depends on.
#
# Paper ↔ code (full dictionary in docs/physics_and_notation.md):
#   bosonic frequency ω          ↔  Ω
#   fermionic frequencies ν, ν′  ↔  ν, νp
#   bosonic momentum q           ↔  P
#   fermionic momenta k, k′      ↔  k, kp
#   Φ_r (Φ̃_r in fd runs)         ↔  the vertex fields `γa`, `γp`, `γt`
# =============================================================================

# -----------------------------------------------------------------------------
# Diagrammatic channels
# -----------------------------------------------------------------------------
# A diagram is reducible in channel r if cutting two internal lines splits it
# into two parts. The channels differ in which pair of lines is cut:
#   a  antiparallel particle-hole channel
#   p  parallel channel = particle-particle channel
#   t  transverse antiparallel particle-hole channel
# Each channel uses its own frequency parametrization (Ω, ν, ν′), see
# convention.jl and Lihm 2025, Eq. (S5). The three channels are related by
# crossing symmetry (Eq. (S4)).

"""
    ChannelTag

Supertype of the channel tags [`aCh`](@ref), [`pCh`](@ref) and [`tCh`](@ref).
Functions take a channel tag to know in which channel's frequency
parametrization their arguments are given (Lihm 2025, Eq. (S5)).
"""
abstract type ChannelTag end

"""
    pCh <: ChannelTag

The parallel channel p, i.e. the particle-particle channel.
"""
struct pCh <: ChannelTag end

"""
    tCh <: ChannelTag

The transverse antiparallel channel t, a particle-hole channel.
"""
struct tCh <: ChannelTag end

"""
    aCh <: ChannelTag

The antiparallel channel a, the other particle-hole channel. By crossing
symmetry it is tied to the t channel (Lihm 2025, Eq. (S11)).
"""
struct aCh <: ChannelTag end

# -----------------------------------------------------------------------------
# Spin components
# -----------------------------------------------------------------------------
# With SU(2) spin symmetry the vertex has only two independent spin structures
# (Lihm 2025, Eq. (S1)):
#
#   F_σ1σ2σ3σ4 = F^= δ_σ1σ2 δ_σ3σ4 + F^× δ_σ1σ4 δ_σ2σ3
#
#   pSp  "parallel" component  F^=  (paper: F^=)
#   xSp  "crossed" component   F^×  (paper: F^×)
#   dSp  2·pSp + xSp; in the t channel this is the density vertex F_D
#
# Only pSp is stored. xSp is computed when needed from crossing symmetry, which
# exchanges two legs (Eq. (S4a)):
#
#   F^×_p(Ω, ν, ν′) = -F^=_p(Ω, ν, Ω - ν′) = -F^=_p(Ω, Ω - ν, ν′)
#   F^×_t(Ω, ν, ν′) = -F^=_a(Ω, ν′, ν)
#   F^×_a(Ω, ν, ν′) = -F^=_t(Ω, ν′, ν)
#
# Since the vertex is also symmetric under ν ↔ ν′, the
# swap of ν and ν′ in the last two lines may be dropped; vertex.jl does that.
#
# dSp is used to simplify some evaluations. The t-channel BSE is diagonal in xSp
# and dSp but not in pSp (Eqs. (S9b), (S14)). So the t-channel BSE is solved
# for dSp, and pSp is recovered afterwards: subtract the xSp part (which is -1
# times the a-channel pSp, by crossing symmetry) and divide by 2.
#
# The physical channels are (Eq. (S2)): singlet S = pSp - xSp, triplet
# T = pSp + xSp, density D = 2·pSp + xSp = dSp, magnetic M = xSp. Written in
# them, the (channel, spin) combinations the code evaluates are
# (Eqs. (S2), (S10), (S11); D and M are the t-channel ones):
#
#   pCh, pSp =  1/2 * (pCh, S) + 1/2 * (pCh, T)
#   pCh, xSp = -1/2 * (pCh, S) + 1/2 * (pCh, T)
#   pCh, dSp =  1/2 * (pCh, S) + 3/2 * (pCh, T)
#
#   aCh, pSp = -1   * (tCh, M)
#   aCh, xSp = -1/2 * (tCh, D) + 1/2 * (tCh, M)
#   aCh, dSp = -1/2 * (tCh, D) - 3/2 * (tCh, M)
#
#   tCh, pSp =  1/2 * (tCh, D) - 1/2 * (tCh, M)
#   tCh, xSp = (tCh, M)
#   tCh, dSp = (tCh, D)
#
# Note that dSp equals the density vertex only in the t channel.

"""
    SpinTag

Supertype of the spin tags [`pSp`](@ref), [`xSp`](@ref) and [`dSp`](@ref),
which select the spin component a vertex is evaluated in (Lihm 2025, Eq. (S1)).
"""
abstract type SpinTag end

"""
    pSp <: SpinTag

The parallel spin component `F^=` (Lihm 2025, Eq. (S1)). This is the only
component that is stored; its bare value is `+U` (Eq. (S3)).
"""
struct pSp <: SpinTag end

"""
    xSp <: SpinTag

The crossed spin component `F^×` (Lihm 2025, Eq. (S1)). It is not stored but
obtained from `pSp` by crossing symmetry (Eq. (S4a)); its bare value is `-U`.
In the t channel it is the magnetic vertex `F_M` (Eq. (S2)).
"""
struct xSp <: SpinTag end

"""
    dSp <: SpinTag

The combination `2·pSp + xSp`. In the t channel this is the density vertex
`F_D` (Lihm 2025, Eq. (S2)), in which the t-channel BSE is diagonal (Eq. (S14)).
Its bare value is `+U`.
"""
struct dSp <: SpinTag end

# Mesh and MeshFunction come from the mesh_generalization_v3 branch of
# MatsubaraFunctions.jl.

# -----------------------------------------------------------------------------
# Mesh aliases
# -----------------------------------------------------------------------------

"""
Fermionic Matsubara frequency mesh, with frequencies iν_n = iπ(2n+1)/β.
"""
const FMesh = Mesh{MeshPoint{MatsubaraFrequency{Fermion}}, MatsubaraDomain}
"""
Bosonic Matsubara frequency mesh, with frequencies iΩ_n = i2πn/β.
"""
const BMesh = Mesh{MeshPoint{MatsubaraFrequency{Boson}}, MatsubaraDomain}
"""
Momentum mesh on the 2D Brillouin zone (the square lattice in the scripts).
"""
const KMesh = Mesh{MeshPoint{BrillouinPoint{2}}, BrillouinDomain{2, 4}}

# -----------------------------------------------------------------------------
# Mesh-function aliases: local objects
# -----------------------------------------------------------------------------
# Objects without momentum dependence: the impurity (Anderson model) solver and
# the DMFT reference. Arguments are listed in storage order. Green functions and
# self-energies are stored multiplied by i (see dyson.jl).

"""
Local Green function or self-energy, `G(ν)` or `Σ(ν)`: one fermionic frequency.
"""
const MF_G{Q}  = MeshFunction{1, Q, Tuple{FMesh}, Array{Q, 1}}
"""
Local bubble `Π(Ω, ν)`: bosonic frequency and the fermionic frequency of the
loop (Lihm 2025, Eq. (S8)).
"""
const MF_Π{Q}  = MeshFunction{2, Q, Tuple{BMesh, FMesh}, Array{Q, 2}}
"""
Local K1 class `K1(Ω)`: bosonic frequency only (Lihm 2025, Eq. (S7)).
"""
const MF_K1{Q} = MeshFunction{1, Q, Tuple{BMesh}, Array{Q, 1}}
"""
Local K2 class `K2(Ω, ν)`: bosonic frequency and one fermionic frequency.
`K̄2(Ω, ν′)` is read from the same array (Lihm 2025, Eq. (S7)).
"""
const MF_K2{Q} = MeshFunction{2, Q, Tuple{BMesh, FMesh}, Array{Q, 2}}
"""
Local K3 class `K3(Ω, ν, ν′)`: bosonic frequency and both fermionic
frequencies (Lihm 2025, Eq. (S7)).
"""
const MF_K3{Q} = MeshFunction{3, Q, Tuple{BMesh, FMesh, FMesh}, Array{Q, 3}}

# -----------------------------------------------------------------------------
# Mesh-function aliases: lattice Green function and bubble
# -----------------------------------------------------------------------------
# The lattice versions of the objects above, with a momentum on the Brillouin
# zone added as the last argument.

"""
Lattice Green function or self-energy `G(ν, k)`: one fermionic frequency and
one momentum.
"""
const NL_MF_G{Q} = MeshFunction{2, Q, Tuple{FMesh, KMesh}, Array{Q, 2}}
"""
Lattice bubble `Π(Ω, ν, P)`, averaged over the fermionic loop momentum: bosonic
frequency, fermionic frequency, bosonic (transfer) momentum `P` (paper: q).
Used by the s-wave solver NL (Lihm 2025, Eq. (S18b)).
"""
const NL_MF_Π{Q} = MeshFunction{3, Q, Tuple{BMesh, FMesh, KMesh}, Array{Q, 3}}

# -----------------------------------------------------------------------------
# Mesh-function aliases: lattice vertex depending on the bosonic momentum only
# -----------------------------------------------------------------------------
# K1/K2/K3 that keep only the bosonic (transfer) momentum P and are averaged
# over the fermionic momenta: the s-wave approximation of Lihm 2025, Eqs.
# (S17)-(S18). NL uses all three; NL2 uses K1 and K3; NL3 uses K1.

"""
Lattice K1 class `K1(Ω, P)`: bosonic frequency and bosonic momentum.
"""
const NL_MF_K1{Q} = MeshFunction{2, Q, Tuple{BMesh, KMesh}, Array{Q, 2}}
"""
Lattice K2 class `K2(Ω, ν, P)` in the s-wave approximation: bosonic
frequency, fermionic frequency, bosonic momentum.
"""
const NL_MF_K2{Q} = MeshFunction{3, Q, Tuple{BMesh, FMesh, KMesh}, Array{Q, 3}}
"""
Lattice K3 class `K3(Ω, ν, ν′, P)` in the s-wave approximation: all three
frequencies and the bosonic momentum.
"""
const NL_MF_K3{Q} = MeshFunction{4, Q, Tuple{BMesh, FMesh, FMesh, KMesh}, Array{Q, 4}}

# -----------------------------------------------------------------------------
# Mesh-function aliases: lattice objects that keep fermionic momenta
# -----------------------------------------------------------------------------
# NL2 keeps the fermionic momentum k in the bubble and in K2; NL3 also keeps
# both fermionic momenta k, k′ in K3 (the paper's "Full" calculation).

"""
Lattice bubble `Π(Ω, ν, P, k)` that keeps the fermionic loop momentum: bosonic
frequency, fermionic frequency, bosonic momentum `P`, fermionic momentum `k`.
"""
const NL2_MF_Π{Q} = MeshFunction{4, Q, Tuple{BMesh, FMesh, KMesh, KMesh}, Array{Q, 4}}
"""
Lattice K2 class `K2(Ω, ν, P, k)` that keeps the fermionic momentum: bosonic
frequency, fermionic frequency, bosonic momentum `P`, fermionic momentum `k`.
"""
const NL2_MF_K2{Q} = MeshFunction{4, Q, Tuple{BMesh, FMesh, KMesh, KMesh}, Array{Q, 4}}
"""
Lattice K3 class `K3(Ω, ν, ν′, P, k, k′)` with its full momentum dependence: all
three frequencies, the bosonic momentum `P` and both fermionic momenta.
"""
const NL3_MF_K3{Q} = MeshFunction{6, Q, Tuple{BMesh, FMesh, FMesh, KMesh, KMesh, KMesh}, Array{Q, 6}}

# -----------------------------------------------------------------------------
# Infinite frequency
# -----------------------------------------------------------------------------
# The asymptotic classes are defined by sending fermionic frequencies to
# infinity (Lihm 2025, Eqs. (S7), (S22)). Instead of a large number, the code
# passes the marker νInf.

"""
    InfiniteMatsubaraFrequency

Type of the marker [`νInf`](@ref) for a fermionic frequency at infinity.
Evaluating a vertex with `ν = νInf` or `ν′ = νInf` gives its high-frequency
limit; every class that needs that frequency to be finite drops out.
"""
struct InfiniteMatsubaraFrequency <: AbstractValue end
"""
    νInf

A fermionic frequency at infinity. For a vertex in channel r,

    F(Ω, ν, νInf) = F₀ + K1 + K2   (paper: F_012)
    F(Ω, νInf, ν′) = F₀ + K1 + K̄2  (paper: F_01̄2)

(Lihm 2025, Eq. (S22)). Any frequency plus or minus `νInf` is `νInf`.
"""
const νInf = InfiniteMatsubaraFrequency()

# Adding or subtracting a finite or infinite frequency to ∞ gives ∞.
Base.:+(::MatsubaraFrequency, ::InfiniteMatsubaraFrequency) = InfiniteMatsubaraFrequency()
Base.:-(::MatsubaraFrequency, ::InfiniteMatsubaraFrequency) = InfiniteMatsubaraFrequency()
Base.:+(::InfiniteMatsubaraFrequency, ::MatsubaraFrequency) = InfiniteMatsubaraFrequency()
Base.:-(::InfiniteMatsubaraFrequency, ::MatsubaraFrequency) = InfiniteMatsubaraFrequency()
Base.:+(::InfiniteMatsubaraFrequency, ::InfiniteMatsubaraFrequency) = InfiniteMatsubaraFrequency()
Base.:-(::InfiniteMatsubaraFrequency, ::InfiniteMatsubaraFrequency) = InfiniteMatsubaraFrequency()

# ∞ is never inside a mesh, so a class stored on that mesh contributes zero.
MatsubaraFunctions.is_inbounds(::InfiniteMatsubaraFrequency, ::Mesh) = false
MatsubaraFunctions.is_inbounds_bc(::InfiniteMatsubaraFrequency, ::Mesh) = false

# -----------------------------------------------------------------------------
# Brillouin zone
# -----------------------------------------------------------------------------

"""
    k0

The Γ point `k = (0, 0)` of the Brillouin zone. Used as a dummy momentum where
an argument does not matter, e.g. for a local vertex or at `ν = νInf`.
"""
const k0 = BrillouinPoint(0, 0)
