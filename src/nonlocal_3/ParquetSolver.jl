# =============================================================================
# The solver object of the lattice solver NL3.
#
# Same layout as the NL2 solver (nonlocal_2/ParquetSolver.jl), with the
# channels stored as NL3_Channel: K3 keeps all momenta, K3(Ω, ν, ν′; q, k, k′),
# and so do the K3 caches. The bubbles Π(Ω, ν; q, k), the SDE kernels and the
# K1 and K2 BSE kernels are those of NL2; only the K3 BSE has its own kernels
# (nonlocal_3/BSE_K3.jl). The K3 symmetry groups act on all three momenta
# (nonlocal_3/symmetries.jl).
#
# Differences from NL2 in the constructor: the bubbles always have nK1
# fermionic frequencies (no mΠν_factor), and init_sym_grp! builds the
# symmetry groups in parallel tasks, because the K3 groups are large.
# NL3 has no one-loop K3 kernel, so the strategy :fdPA_1loop does not work.
# =============================================================================

"""
    NL3_ParquetSolver{Q, VT, RefVT} <: AbstractSolver{Q}

Solver for the fd parquet equations of the Hubbard model in which K2, K3 and
the bubbles keep their fermionic momenta (solver NL3). The fields are those of
the [`NL2_ParquetSolver`](@ref), with `NL3_Channel` vertex channels and K3
caches over (Ω, ν, ν′, q, k, k′).

    NL3_ParquetSolver(nK1, nK2, nK3, mK_Γ, Gbare, G0, Σ0, F0, VT = NL3_Vertex; mode = :threads)

Build a solver with frequency boxes `nK1`, `nK2`, `nK3` and the momentum mesh
`mK_Γ` of the vertex and the bubbles. `Gbare`, `G0` and `Σ0` live on the fine
mesh; `F0` is the reference vertex and `VT` the type of the target vertex F.
The reference bubbles are computed once from `G0`, with `nK1` fermionic
frequencies. The target starts from Σ = Σ0 and zero fd channels. Call
[`init_sym_grp!`](@ref) before solving.
"""
mutable struct NL3_ParquetSolver{Q, VT, RefVT} <: AbstractSolver{Q}
    # Bare Green function of the target (fine momentum mesh)
    Gbare :: NL_MF_G{Q}

    # single-particle Green's function and bubbles for reference system
    G0::NL_MF_G{Q}
    Π0pp::NL2_MF_Π{Q}
    Π0ph::NL2_MF_Π{Q}

    # self-energy for reference system
    Σ0::NL_MF_G{Q}

    # two-particle vertex for reference system
    F0::RefVT

    # single-particle Green's function and bubbles for target system
    G::NL_MF_G{Q}
    Πpp::NL2_MF_Π{Q}
    Πph::NL2_MF_Π{Q}

    # self-energy for target system
    Σ::NL_MF_G{Q}

    # target vertex F = F0 + fd channels
    F::VT

    # output buffer of one BSE sweep (new fd channels)
    Fbuff::NL3_Vertex{Q, RefVertex{Q}}

    # left part of the BSE, FL = Ĩ π f
    FL::NL3_Vertex{Q, RefVertex{Q}}

    # SDE kernels (target and reference), shaped like a K2 class
    Lpp  :: NL2_MF_K2{Q}
    Lph  :: NL2_MF_K2{Q}
    L0pp :: NL2_MF_K2{Q}
    L0ph :: NL2_MF_K2{Q}

    # symmetry group for the self energy
    SGΣ::SymmetryGroup

    # symmetry groups for the particle-particle channel of F
    SGpp::Vector{SymmetryGroup}

    # symmetry groups for the particle-hole channels of F
    SGph::Vector{SymmetryGroup}

    # symmetry groups for the particle-particle channel of FL
    SGppL::Vector{SymmetryGroup}

    # symmetry groups for the particle-hole channels of FL
    SGphL::Vector{SymmetryGroup}

    # symmetry groups for the reference SDE kernels (fd runs)
    SG0pp2::SymmetryGroup
    SG0ph2::SymmetryGroup

    # Parallelization mode
    mode::Symbol

    # Pre-evaluated vertices. To be used in the BSE of the K3 class
    # See build_K3_cache.jl for details. Here over (Ω, ν, ν′, q, k, k′).
    cache_Γpx :: NL3_MF_K3{Q}
    cache_F0p :: NL3_MF_K3{Q}
    cache_F0a :: NL3_MF_K3{Q}
    cache_F0t :: NL3_MF_K3{Q}
    cache_Γpp :: NL3_MF_K3{Q}
    cache_Γa  :: NL3_MF_K3{Q}
    cache_Γt  :: NL3_MF_K3{Q}
    cache_Fp  :: NL3_MF_K3{Q}
    cache_Fa  :: NL3_MF_K3{Q}
    cache_Ft  :: NL3_MF_K3{Q}

    # constructor, see the docstring above
    function NL3_ParquetSolver(
        nK1   :: Int64,
        nK2   :: NTuple{2,Int64},
        nK3   :: NTuple{2,Int64},
        mK_Γ  :: KMesh,
        Gbare :: NL_MF_G{Q},
        G0    :: NL_MF_G{Q},
        Σ0    :: NL_MF_G{Q},
        F0    :: RefVT,
              :: Type{VT} = NL3_Vertex,
        ;
        mode::Symbol = :threads,
    ) where {Q, VT, RefVT}

        T = MatsubaraFunctions.temperature(meshes(G0, Val(1)))

        # bubbles π of the reference system, π(Ω, ν; q, k) on the coarse mesh
        mΠΩ = MatsubaraMesh(temperature(F0), nK1, Boson)
        mΠν = MatsubaraMesh(temperature(F0), nK1, Fermion)
        Π0pp = MeshFunction(mΠΩ, mΠν, mK_Γ, mK_Γ; data_t=Q)
        Π0ph = copy(Π0pp)

        bubbles_real_space!(Π0pp, Π0ph, G0)

        # single-particle Green's function and self-energy
        # Initialization: Σ = Σ0
        G = copy(G0)
        Σ = copy(Σ0)
        Dyson!(G, Σ, Gbare)

        # bubbles
        Πpp = copy(Π0pp)
        Πph = copy(Π0pp)
        bubbles_real_space!(Πpp, Πph, G)

        # target vertex (F0 plus zero channels) and BSE output buffer
        F = VT(F0, T, nK1, nK2, nK3, mK_Γ)
        Fbuff = NL3_Vertex(RefVertex(T, 0.0, Q), T, nK1, nK2, nK3, mK_Γ)

        # Vertices for the SDE
        Lpp = copy(F.γp.K2)
        Lph  = copy(Lpp)
        L0pp = copy(Lpp)
        L0ph = copy(Lpp)

        # placeholder symmetry groups; init_sym_grp! sets the real ones
        SGΣ = SymmetryGroup(Σ)
        SGpp = SymmetryGroup[SymmetryGroup(F.γp.K1), SymmetryGroup(F.γp.K2), SymmetryGroup(F.γp.K3)]
        SGph = SymmetryGroup[SymmetryGroup(F.γp.K1), SymmetryGroup(F.γp.K2), SymmetryGroup(F.γp.K3)]
        SGppL = SymmetryGroup[SymmetryGroup(F.γp.K1), SymmetryGroup(F.γp.K2), SymmetryGroup(F.γp.K3)]
        SGphL = SymmetryGroup[SymmetryGroup(F.γp.K1), SymmetryGroup(F.γp.K2), SymmetryGroup(F.γp.K3)]

        # Symmetry group of F0, needed for the BSE of the reference vertex in SDE for fdPA
        SG0pp2 = SymmetryGroup(L0pp)
        SG0ph2 = SymmetryGroup(L0ph)

        # Check consistency of meshes
        @assert meshes(Gbare) == meshes(G0) == meshes(Σ0) == meshes(Σ) == meshes(Σ)

        # Pre-evaluated caches for the BSE of the K3 class
        cache_Γpx = copy(F.γp.K3)
        cache_F0p = copy(F.γp.K3)
        cache_F0a = copy(F.γp.K3)
        cache_F0t = copy(F.γp.K3)
        cache_Γpp = copy(F.γp.K3)
        cache_Γa  = copy(F.γp.K3)
        cache_Γt  = copy(F.γp.K3)
        cache_Fp  = copy(F.γp.K3)
        cache_Fa  = copy(F.γp.K3)
        cache_Ft  = copy(F.γp.K3)

        return new{Q, typeof(F), RefVT}(Gbare, G0, Π0pp, Π0ph, Σ0, F0, G, Πpp, Πph, Σ, F, Fbuff, copy(Fbuff),
        Lpp, Lph, L0pp, L0ph,
        SGΣ, SGpp, SGph, SGppL, SGphL, SG0pp2, SG0ph2, mode, cache_Γpx, cache_F0p, cache_F0a,
        cache_F0t, cache_Γpp, cache_Γa, cache_Γt, cache_Fp, cache_Fa, cache_Ft) :: NL3_ParquetSolver{Q}
    end
end

# Printing: type, U, temperature, box sizes and number of momenta.
function Base.show(io::IO, S::NL3_ParquetSolver{Q}) where {Q}
    print(io, "$(nameof(typeof(S))){$Q}, U = $(real(bare_vertex(S.F0, aCh, pSp))), T = $(temperature(S))\n")
    print(io, "F0 K3 : $(numK3(S.F0))\n")
    print(io, "F  K1 : $(numK1(S.F))\n")
    print(io, "F  K2 : $(numK2(S.F))\n")
    print(io, "F  K3 : $(numK3(S.F))\n")
    print(io, "F  P  : $(numP(S.F))")
end


# -----------------------------------------------------------------------------
# Parquet approximation for the Hubbard model
# -----------------------------------------------------------------------------

"""
    parquet_solver_hubbard_parquet_approximation_NL3(nG, nK1, nK2, nK3, mK_G, mK_Γ, Q = ComplexF64; T, U, μ, t1, t2 = 0, t3 = 0, mode = :threads, VT = NL3_Vertex) -> NL3_ParquetSolver

Solver NL3 for the parquet approximation (PA) of the Hubbard model: bare
reference g = σ = 0 and f = U. Arguments as for
[`parquet_solver_hubbard_parquet_approximation`](@ref); `VT` is the type of the
target vertex.
"""
function parquet_solver_hubbard_parquet_approximation_NL3(
    nG::Int64,
    nK1::Int64,
    nK2::NTuple{2,Int64},
    nK3::NTuple{2,Int64},
    mK_G :: KMesh,
    mK_Γ :: KMesh,
    :: Type{Q} = ComplexF64,
    ;
    mode::Symbol = :threads,
    VT = NL3_Vertex,
    T,
    U,
    μ, t1, t2 = 0., t3 = 0.,
) where {Q}

    # Mesh for the Green functions and self-energy
    mG = MatsubaraMesh(T, nG, Fermion)

    Gbare = hubbard_bare_Green(mG, mK_G, Q; μ, t1, t2, t3)

    # Reference system: G0 = Σ0 = 0, F0 = U (parquet approximation)
    G0 = MeshFunction(mG, mK_G; data_t = Q)
    Σ0 = MeshFunction(mG, mK_G; data_t = Q)
    F0 = RefVertex(T, U, Q)
    set!(G0, 0)
    set!(Σ0, 0)

    NL3_ParquetSolver(nK1, nK2, nK3, mK_Γ, Gbare, G0, Σ0, F0, VT; mode)
end


# -----------------------------------------------------------------------------
# Symmetry groups
# -----------------------------------------------------------------------------

"""
    init_sym_grp!(S::NL3_ParquetSolver) -> nothing

Build the symmetry groups of the NL3 solver: as for NL2, except that the K3
groups use the symmetries of K3(Ω, ν, ν′; q, k, k′), which act on all three
momenta (`nonlocal_3/symmetries.jl`). The groups are built in parallel tasks.
The groups of the left part FL omit the exchange ν ↔ ν′. Call this once
before solving.
"""
function init_sym_grp!(
    S::NL3_ParquetSolver
    )::Nothing

    mK_Σ = meshes(S.Σ, Val(2))
    mK_Γ = meshes(S.F.γp.K1, Val(2))

    # The individual symmetry groups are built independently of each other and only read
    # from S, so they can be constructed concurrently. The four K3 groups dominate the
    # runtime (their meshes carry three momenta each), hence they are spawned first.
    t_SGpp3 = Threads.@spawn my_SymmetryGroup([
        Symmetry{6}(w -> sK3_NL3_pp1(w, mK_Γ)),
        Symmetry{6}(w -> sK3_NL3_pp2(w, mK_Γ)),
        Symmetry{6}(w -> sK3_NL3_pp3(w, mK_Γ)),
        Symmetry{6}(w -> sK3_NL3_ref(w, mK_Γ)),
        Symmetry{6}(w -> sK3_NL3_rot(w, mK_Γ)),
    ], S.F.γp.K3)

    t_SGppL3 = Threads.@spawn my_SymmetryGroup([
        Symmetry{6}(w -> sK3_NL3_pp1(w, mK_Γ)),
        Symmetry{6}(w -> sK3_NL3_pp3(w, mK_Γ)),
        Symmetry{6}(w -> sK3_NL3_ref(w, mK_Γ)),
        Symmetry{6}(w -> sK3_NL3_rot(w, mK_Γ)),
    ], S.F.γp.K3)

    t_SGph3 = Threads.@spawn my_SymmetryGroup([
        Symmetry{6}(w -> sK3_NL3_ph1(w, mK_Γ)),
        Symmetry{6}(w -> sK3_NL3_ph2(w, mK_Γ)),
        Symmetry{6}(w -> sK3_NL3_ph3(w, mK_Γ)),
        Symmetry{6}(w -> sK3_NL3_ref(w, mK_Γ)),
        Symmetry{6}(w -> sK3_NL3_rot(w, mK_Γ)),
    ], S.F.γt.K3)

    t_SGphL3 = Threads.@spawn my_SymmetryGroup([
        Symmetry{6}(w -> sK3_NL3_ph1(w, mK_Γ)),
        Symmetry{6}(w -> sK3_NL3_ph3(w, mK_Γ)),
        Symmetry{6}(w -> sK3_NL3_ref(w, mK_Γ)),
        Symmetry{6}(w -> sK3_NL3_rot(w, mK_Γ)),
    ], S.F.γt.K3)

    # self-energy
    t_SGΣ = Threads.@spawn my_SymmetryGroup(Symmetry{2}[
        Symmetry{2}(w -> sΣ_conj(w, mK_Σ)),
        Symmetry{2}(w -> sΣ_ref(w, mK_Σ)),
        Symmetry{2}(w -> sΣ_rot(w, mK_Σ))
    ], S.Σ)

    # Vertices in the particle-particle channel
    t_SGpp1 = Threads.@spawn my_SymmetryGroup(Symmetry{2}[
        Symmetry{2}(w -> sK1_conj(w, mK_Γ)),
        Symmetry{2}(w -> sK1_ref(w, mK_Γ)),
        Symmetry{2}(w -> sK1_rot(w, mK_Γ)),
    ], S.F.γp.K1)

    t_SGpp2 = Threads.@spawn my_SymmetryGroup([
        Symmetry{4}(w -> sK2_NL2_pp1(w, mK_Γ)),
        Symmetry{4}(w -> sK2_NL2_pp2(w, mK_Γ)),
        Symmetry{4}(w -> sK2_NL2_ref(w, mK_Γ)),
        Symmetry{4}(w -> sK2_NL2_rot(w, mK_Γ)),
    ], S.F.γp.K2)

    # particle-hole channels
    t_SGph2 = Threads.@spawn my_SymmetryGroup([
        Symmetry{4}(w -> sK2_NL2_ph1(w, mK_Γ)),
        Symmetry{4}(w -> sK2_NL2_ph2(w, mK_Γ)),
        Symmetry{4}(w -> sK2_NL2_ref(w, mK_Γ)),
        Symmetry{4}(w -> sK2_NL2_rot(w, mK_Γ)),
    ], S.F.γt.K2)

    # reference SDE kernels (SG0ph2 is built on L0pp, which has the same mesh as L0ph)
    t_SG0pp2 = Threads.@spawn my_SymmetryGroup([
        Symmetry{4}(w -> sK2_NL2_pp1(w, mK_Γ)),
        Symmetry{4}(w -> sK2_NL2_pp2(w, mK_Γ)),
        Symmetry{4}(w -> sK2_NL2_ref(w, mK_Γ)),
        Symmetry{4}(w -> sK2_NL2_rot(w, mK_Γ)),
    ], S.L0pp)

    t_SG0ph2 = Threads.@spawn my_SymmetryGroup([
        Symmetry{4}(w -> sK2_NL2_ph1(w, mK_Γ)),
        Symmetry{4}(w -> sK2_NL2_ph2(w, mK_Γ)),
        Symmetry{4}(w -> sK2_NL2_ref(w, mK_Γ)),
        Symmetry{4}(w -> sK2_NL2_rot(w, mK_Γ)),
    ], S.L0pp)

    # collect the results of the tasks
    S.SGΣ = fetch(t_SGΣ)

    S.SGpp[1] = fetch(t_SGpp1)
    S.SGpp[2] = fetch(t_SGpp2)
    S.SGpp[3] = fetch(t_SGpp3)

    # left part: without ν ↔ ν′
    S.SGppL[1] = S.SGpp[1]
    S.SGppL[2] = S.SGpp[2]
    S.SGppL[3] = fetch(t_SGppL3)

    # particle-hole channels (K1 symmetries as in the p channel)
    S.SGph[1] = S.SGpp[1]
    S.SGph[2] = fetch(t_SGph2)
    S.SGph[3] = fetch(t_SGph3)

    S.SGphL[1] = S.SGph[1]
    S.SGphL[2] = S.SGph[2]
    S.SGphL[3] = fetch(t_SGphL3)

    S.SG0pp2 = fetch(t_SG0pp2)
    S.SG0ph2 = fetch(t_SG0ph2)

    return nothing
end

# -----------------------------------------------------------------------------
# Getters and bubbles
# -----------------------------------------------------------------------------

# Number of momenta of the fine (G, Σ) and the coarse (vertex) mesh.
numP_G(S::NL3_ParquetSolver)::Int64 = length(meshes(S.G, Val(2)))
numP_Γ(S::NL3_ParquetSolver)::Int64 = numP(S.F)

"""
    bubbles!(S::NL3_ParquetSolver)

Update the target bubbles `S.Πpp`, `S.Πph` (Π(Ω, ν; q, k), as in NL2) from
`S.G` by real-space convolution ([`bubbles_real_space!`](@ref)).
"""
function bubbles!(S :: NL3_ParquetSolver)
    bubbles_real_space!(S)
    # bubbles_momentum_space!(S)
end
