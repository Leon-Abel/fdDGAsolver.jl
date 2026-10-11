# =============================================================================
# The solver object of the local (impurity) problem, and helpers shared by
# all solvers.
#
# A solver holds everything one parquet calculation needs:
#   reference system (known solution, lower case in Lihm 2025):
#       G0 = g, Σ0 = σ, F0 = f, bubbles Π0pp, Π0ph = π
#   target system (the unknowns, upper case in Lihm 2025):
#       G, Σ, F = f + Φ̃_a + Φ̃_p + Φ̃_t, bubbles Πpp, Πph
#   Gbare, the bare propagator of the target, used in the Dyson equation,
#   and work buffers: the output of a BSE sweep (Fbuff), the left part FL of
#   the BSE, the SDE kernels, symmetry groups and the K3 caches.
#
# The parquet approximation (PA) is the special case with the bare reference
# g = σ = 0 and f = U (parquet_solver_siam_parquet_approximation). Then
# Φ̃_r = Φ_r and the fd equations become the ordinary parquet equations.
#
# Paper ↔ code: f, g, σ, π ↔ F0, G0, Σ0, Π0 (suffix 0 = reference);
# F, G, Σ, Π ↔ F, G, Σ, Π. In Lihm 2025, F₀ is the bare vertex. In the code
# that is bare_vertex(F), while the field F0 is the reference vertex. G and Σ
# are stored as iG and iΣ (dyson.jl).
#
# The lattice solvers (nonlocal*/ParquetSolver.jl) have the same layout, with
# momentum-dependent fields.
# =============================================================================

# -----------------------------------------------------------------------------
# Abstract type
# -----------------------------------------------------------------------------

"""
    AbstractSolver{Q}

Supertype of all solvers, with number type `Q` (usually `ComplexF64`):
[`ParquetSolver`](@ref) (local) and `NL_ParquetSolver`, `NL2_ParquetSolver`,
`NL3_ParquetSolver` (lattice). Code written for `AbstractSolver` only uses
the fields all of them share (see `ParquetSolver`).
"""
abstract type AbstractSolver{Q}; end

# number type of the stored data
Base.eltype(::Type{<: AbstractSolver{Q}}) where {Q} = Q

# -----------------------------------------------------------------------------
# ParquetSolver struct
# -----------------------------------------------------------------------------

"""
    ParquetSolver{Q, VT, RefVT} <: AbstractSolver{Q}

Solver for the fd parquet equations of the impurity (no momenta), Lihm 2025,
Eqs. (5), (S29), (S38).

Type parameters: `Q` number type, `VT` type of the target vertex `F` (e.g.
`Vertex`, `MBEVertex`), `RefVT` type of the reference vertex `F0` (`RefVertex`
in the PA, or a vertex for an fd calculation with a non-trivial reference).

Reference system, fixed inputs: `G0`, `Σ0`, `F0` and the bubbles
`Π0pp`, `Π0ph`. Target system, the unknowns: `G`, `Σ`, `F` (with `F.F0 === F0`)
and the bubbles `Πpp`, `Πph`. `Gbare` is the target's bare propagator.

    ParquetSolver(nK1, nK2, nK3, Gbare, G0, Σ0, F0, VT = Vertex; mode = :threads, mΠν_factor = 6)

Build a solver with K1/K2/K3 box sizes `nK1`, `nK2`, `nK3`.
- The reference bubbles are computed once from `G0`, on a mesh with `nK1`
  bosonic and `nK1 * mΠν_factor` fermionic frequencies. The sum over the
  internal frequency in the K1 and K2 BSEs runs over this mesh.
- The target starts from `Σ = Σ0` (G from the Dyson equation) and
  `F = F0` with zero channels, i.e. Φ̃ = 0.
- `mode` selects the parallelization (`:serial`, `:threads`, `:hybrid`).

Call [`init_sym_grp!`](@ref) before solving.
"""
mutable struct ParquetSolver{Q, VT, RefVT} <: AbstractSolver{Q}
    # Non-interacting (bare) Green's function of the target system
    Gbare :: MF_G{Q}

    # -------------------------------------------------------------------------
    # Reference system (known solution; fixed during the solve)
    # -------------------------------------------------------------------------

    # single-particle Green's function g and bubbles π for the reference system
    G0::MF_G{Q}
    Π0pp::MF_Π{Q}
    Π0ph::MF_Π{Q}

    # self-energy σ for the reference system
    Σ0::MF_G{Q}

    # two-particle vertex f for the reference system
    F0::RefVT

    # -------------------------------------------------------------------------
    # Target system (the unknowns)
    # -------------------------------------------------------------------------

    # single-particle Green's function and bubbles for the target system
    G::MF_G{Q}
    Πpp::MF_Π{Q}
    Πph::MF_Π{Q}

    # self-energy for the target system
    Σ::MF_G{Q}

    # target vertex F = F0 + γp + γt + γa, i.e. f + Φ̃_p + Φ̃_t + Φ̃_a
    F::VT

    # -------------------------------------------------------------------------
    # Work buffers
    # -------------------------------------------------------------------------

    # Output of one BSE sweep: the new channels Φ̃_r. All BSEs of a sweep read
    # the old F and write here; afterwards the buffer is copied to F via
    # `set!(S.F, S.Fbuff)`. Its F0 is a RefVertex with U = 0 (unused).
    Fbuff::Vertex{Q, RefVertex{Q}}

    # Left part of the BSE, FL = Ĩ_r π_r f (K2 and K3 classes; no K1). It is
    # the "left" term in the code's mirrored form of Lihm 2025, Eq. (5), i.e.
    # the right-most term f π_r Ĩ_r in the paper's ordering. Computed once per
    # sweep by BSE_L_K2! and BSE_L_K3!.
    FL::Vertex{Q, RefVertex{Q}}

    # SDE kernels (SDE.jl): Lpp, Lph for the target, L0pp, L0ph for the
    # reference. They have the shape of a K2 class.
    Lpp  :: MF_K2{Q}
    Lph  :: MF_K2{Q}
    L0pp :: MF_K2{Q}
    L0ph :: MF_K2{Q}

    # Symmetry groups (symmetries.jl, set by init_sym_grp!). Each computes one
    # point per set of symmetry-related points and fills in the rest.
    # symmetry group for the self-energy
    SGΣ::SymmetryGroup

    # symmetry groups for K1, K2, K3 of the p channel of F
    SGpp::Vector{SymmetryGroup}

    # symmetry groups for K1, K2, K3 of the a and t channels of F
    SGph::Vector{SymmetryGroup}

    # the same for the left part FL (without the ν ↔ ν′ symmetry)
    SGppL::Vector{SymmetryGroup}
    SGphL::Vector{SymmetryGroup}

    # symmetry groups for the reference SDE kernels L0pp, L0ph (fd runs)
    SG0pp2::SymmetryGroup
    SG0ph2::SymmetryGroup

    # Parallelization mode
    mode::Symbol

    # Vertex slices for the K3 BSE, precomputed once per sweep on the K3 box
    # (build_K3_cache.jl): cache_Γr = Ĩ_r, cache_F0r = f - f(ν′ → ∞),
    # cache_Fr = F - F(ν → ∞) for channel r. cache_Γpx is Ĩ_p in the crossed
    # spin component; the t-channel caches hold the density combination.
    cache_Γpx :: MF_K3{Q}
    cache_F0p :: MF_K3{Q}
    cache_F0a :: MF_K3{Q}
    cache_F0t :: MF_K3{Q}
    cache_Γpp :: MF_K3{Q}
    cache_Γa  :: MF_K3{Q}
    cache_Γt  :: MF_K3{Q}
    cache_Fp  :: MF_K3{Q}
    cache_Fa  :: MF_K3{Q}
    cache_Ft  :: MF_K3{Q}

    # constructor, see the docstring above
    function ParquetSolver(
        nK1   :: Int64,
        nK2   :: NTuple{2,Int64},
        nK3   :: NTuple{2,Int64},
        Gbare :: MF_G{Q},
        G0    :: MF_G{Q},
        Σ0    :: MF_G{Q},
        F0    :: RefVT,
              :: Type{VT} = Vertex,
        ;
        mode::Symbol = :threads,
        mΠν_factor :: Int = 6,
    ) where {Q, VT, RefVT}

        T = MatsubaraFunctions.temperature(meshes(G0, Val(1)))

        # bubbles π of the reference system, computed once
        mΠΩ = MatsubaraMesh(temperature(F0), nK1, Boson)
        mΠν = MatsubaraMesh(temperature(F0), nK1 * mΠν_factor, Fermion)
        Π0pp = MeshFunction(mΠΩ, mΠν; data_t=Q)
        Π0ph = copy(Π0pp)

        bubbles!(Π0pp, Π0ph, G0)

        # single-particle Green's function and self-energy
        # Initialization: Σ = Σ0
        G = copy(G0)
        Σ = copy(Σ0)
        Dyson!(G, Σ, Gbare)

        # bubbles
        Πpp = copy(Π0pp)
        Πph = copy(Π0pp)
        bubbles!(Πpp, Πph, G)

        # target vertex: F0 plus three zero channels; BSE output buffer
        F = VT(F0, T, nK1, nK2, nK3)
        Fbuff = Vertex(RefVertex(T, 0.0, Q), T, nK1, nK2, nK3)

        # SDE kernels, shaped like a K2 class
        Lpp = copy(F.γp.K2)
        Lph = copy(Lpp)
        L0pp = copy(Lpp)
        L0ph = copy(Lpp)

        # placeholder symmetry groups (no symmetries); init_sym_grp! sets the real ones
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

        # FL starts as a copy of the (zero) buffer Fbuff
        return new{Q, typeof(F), RefVT}(Gbare, G0, Π0pp, Π0ph, Σ0, F0, G, Πpp, Πph, Σ, F, Fbuff, copy(Fbuff),
        Lpp, Lph, L0pp, L0ph,
        SGΣ, SGpp, SGph, SGppL, SGphL, SG0pp2, SG0ph2, mode, cache_Γpx, cache_F0p, cache_F0a,
        cache_F0t, cache_Γpp, cache_Γa, cache_Γt, cache_Fp, cache_Fa, cache_Ft) :: ParquetSolver{Q}
    end
end


# Printing: type, U, temperature and the box sizes.
function Base.show(io::IO, S::AbstractSolver{Q}) where {Q}
    print(io, "$(nameof(typeof(S))){$Q}, U = $(real(bare_vertex(S.F0, aCh, pSp))), T = $(temperature(S))\n")
    print(io, "F0 K3 : $(numK3(S.F0))\n")
    print(io, "F  K1 : $(numK1(S.F))\n")
    print(io, "F  K2 : $(numK2(S.F))\n")
    print(io, "F  K3 : $(numK3(S.F))")
end

# -----------------------------------------------------------------------------
# Parquet approximation for the Anderson impurity model
# -----------------------------------------------------------------------------

"""
    parquet_solver_siam_parquet_approximation(nG, nK1, nK2, nK3, Q = ComplexF64; e, Δ, D, T, U, mode = :threads, mΠν_factor = 6, VT = Vertex) -> ParquetSolver

Solver for the parquet approximation (PA) of the Anderson impurity model: the
fully irreducible vertex is replaced by the bare U. This is the fd setup with
the bare reference g = σ = 0, f = U (Lihm 2025, text below Eq. (6)).

`nG` is the number of positive frequencies of G and Σ; `e`, `Δ`, `D` define
the bath (see [`siam_bare_Green`](@ref)); `T` is the temperature and `U` the
interaction.
"""
function parquet_solver_siam_parquet_approximation(
    nG::Int64,
    nK1::Int64,
    nK2::NTuple{2,Int64},
    nK3::NTuple{2,Int64},
    :: Type{Q} = ComplexF64,
    ;
    mode::Symbol = :threads,
    mΠν_factor = 6,
    VT = Vertex,
    e,
    Δ,
    D,
    T,
    U,
) where {Q}

    # Mesh for the Green functions and self-energy
    mG = MatsubaraMesh(T, nG, Fermion)

    Gbare = siam_bare_Green(mG, Q; e, Δ, D)

    # Reference system: G0 = Σ0 = 0, F0 = U (parquet approximation)
    G0 = MeshFunction(mG; data_t = Q)
    Σ0 = MeshFunction(mG; data_t = Q)
    F0 = fdDGAsolver.RefVertex(T, U, Q)
    set!(G0, 0)
    set!(Σ0, 0)

    ParquetSolver(nK1, nK2, nK3, Gbare, G0, Σ0, F0, VT; mode, mΠν_factor)
end

# -----------------------------------------------------------------------------
# Symmetry groups
# -----------------------------------------------------------------------------

"""
    init_sym_grp!(S::ParquetSolver) -> nothing

Build the symmetry groups of the solver from the symmetries in symmetries.jl.
Call this once before solving.

- `SGΣ`: complex conjugation of Σ.
- `SGpp`, `SGph`: K1, K2, K3 of the p channel and of the a and t channels.
- `SGppL`, `SGphL`: the same for the left part FL, but without the ν ↔ ν′ swap
  in K3, because FL = Ĩ π f is not symmetric under ν ↔ ν′.
- `SG0pp2`, `SG0ph2`: the reference SDE kernels.
"""
function init_sym_grp!(
    S::ParquetSolver{Q}
    )::Nothing where {Q}

    # self-energy
    S.SGΣ = my_SymmetryGroup([Symmetry{1}(w -> sΣ(w, Q))], S.Σ)

    # particle-particle channel
    S.SGpp[1] = my_SymmetryGroup([Symmetry{1}(w -> sK1pp(w, Q))], S.F.γp.K1)
    S.SGpp[2] = my_SymmetryGroup([Symmetry{2}(w -> sK2pp1(w, Q)), Symmetry{2}(w -> sK2pp2(w, Q))], S.F.γp.K2)
    S.SGpp[3] = my_SymmetryGroup([Symmetry{3}(w -> sK3pp1(w, Q)), Symmetry{3}(w -> sK3pp2(w, Q)), Symmetry{3}(w -> sK3pp3(w, Q))], S.F.γp.K3)
    S.SGppL[1] = S.SGpp[1]
    S.SGppL[2] = S.SGpp[2]
    S.SGppL[3] = my_SymmetryGroup([Symmetry{3}(w -> sK3pp1(w, Q)), Symmetry{3}(w -> sK3pp3(w, Q))], S.F.γp.K3)

    # particle-hole channels
    S.SGph[1] = my_SymmetryGroup([Symmetry{1}(w -> sK1ph(w, Q))], S.F.γt.K1)
    S.SGph[2] = my_SymmetryGroup([Symmetry{2}(w -> sK2ph1(w, Q)), Symmetry{2}(w -> sK2ph2(w, Q))], S.F.γt.K2)
    S.SGph[3] = my_SymmetryGroup([Symmetry{3}(w -> sK3ph1(w, Q)), Symmetry{3}(w -> sK3ph2(w, Q)), Symmetry{3}(w -> sK3ph3(w, Q))], S.F.γt.K3)
    S.SGphL[1] = S.SGph[1]
    S.SGphL[2] = S.SGph[2]
    S.SGphL[3] = my_SymmetryGroup([Symmetry{3}(w -> sK3ph1(w, Q)), Symmetry{3}(w -> sK3ph3(w, Q))], S.F.γt.K3)

    # For the reference SDE kernels. (SG0ph2 is built on L0pp, which has the
    # same mesh as L0ph.)
    S.SG0pp2 = my_SymmetryGroup([Symmetry{2}(w -> sK2pp1(w, Q)), Symmetry{2}(w -> sK2pp2(w, Q))], S.L0pp)
    S.SG0ph2 = my_SymmetryGroup([Symmetry{2}(w -> sK2ph1(w, Q)), Symmetry{2}(w -> sK2ph2(w, Q))], S.L0pp)

    return nothing
end


"""
    reset_sym_grp!(S::AbstractSolver) -> nothing

Replace the symmetry groups of Σ and of the channels by trivial ones, which
contain no symmetry (every point is computed). Not called anywhere in the
package; useful for checking the symmetries.
"""
function reset_sym_grp!(S :: AbstractSolver)
    (; F, Σ) = S

    S.SGΣ = SymmetryGroup(Σ)
    S.SGpp = SymmetryGroup[SymmetryGroup(F.γp.K1), SymmetryGroup(F.γp.K2), SymmetryGroup(F.γp.K3)]
    S.SGph = SymmetryGroup[SymmetryGroup(F.γp.K1), SymmetryGroup(F.γp.K2), SymmetryGroup(F.γp.K3)]
    S.SGppL = SymmetryGroup[SymmetryGroup(F.γp.K1), SymmetryGroup(F.γp.K2), SymmetryGroup(F.γp.K3)]
    S.SGphL = SymmetryGroup[SymmetryGroup(F.γp.K1), SymmetryGroup(F.γp.K2), SymmetryGroup(F.γp.K3)]

    return nothing
end

"""
    symmetrize_solver!(S::AbstractSolver; verbose = false) -> (; S, err_list)

Impose the symmetries on Σ and on K1, K2, K3 of all three channels of `S.F`,
in place. `err_list` holds, for each object, the largest change made, i.e.
how strongly the symmetry was violated. Used after interpolation and in the
preconditioned solver.
"""
function symmetrize_solver!(S :: AbstractSolver; verbose = false)
    err_list = [S.SGΣ(S.Σ)]
    for Ch in (aCh, pCh, tCh)
        SG = Ch === pCh ? S.SGpp : S.SGph
        γ = get_reducible_vertex(S.F, Ch)
        push!(err_list, SG[1](γ.K1))
        push!(err_list, SG[2](γ.K2))
        push!(err_list, SG[3](γ.K3))
    end
    if verbose
        display(err_list)
    end
    return (; S, err_list)
end

# -----------------------------------------------------------------------------
# Getters
# -----------------------------------------------------------------------------

"""
    temperature(S::AbstractSolver) -> Float64

Temperature T = 1/β of the solver.
"""
function MatsubaraFunctions.temperature(
    S::AbstractSolver
    )::Float64

    return MatsubaraFunctions.temperature(S.F)
end

# Box sizes: positive frequencies of G, Σ and of the K1, K2, K3 classes.
numG(S::AbstractSolver)::Int64 = N(meshes(S.G, Val(1)))
numΣ(S::AbstractSolver)::Int64 = N(meshes(S.Σ, Val(1)))
numK1(S::AbstractSolver)::Int64 = numK1(S.F)
numK2(S::AbstractSolver)::NTuple{2,Int64} = numK2(S.F)
numK3(S::AbstractSolver)::NTuple{2,Int64} = numK3(S.F)

# -----------------------------------------------------------------------------
# Conversion to a flat vector
# -----------------------------------------------------------------------------
# The unknowns of the fixed-point problem are the three channels of F and the
# self-energy, written as one vector [F channels; Σ] (solve.jl, mfRG.jl).

"""
    flatten(S::AbstractSolver) -> Vector
    flatten!(S::AbstractSolver, x) -> nothing
    unflatten!(S::AbstractSolver, x) -> nothing

Convert between the solver's unknowns and one vector: first the three channels
of `S.F` (see `flatten` for vertices), then `S.Σ`. `unflatten!` writes a vector
back into `S.F` and `S.Σ`.
"""
function MatsubaraFunctions.flatten(
    S::AbstractSolver{Q}
)::Vector{Q} where {Q}

    return vcat(flatten(S.F), flatten(S.Σ))
end

# flatten! of a solver, see the docstring above
function MatsubaraFunctions.flatten!(
    S::AbstractSolver{Q},
    x::Vector{Q}
)::Nothing where {Q}

    lenΣ = length(S.Σ.data)
    flatten!(S.F, @view x[1:end-lenΣ])
    flatten!(S.Σ, @view x[end-lenΣ+1:end])

    return nothing
end

# unflatten! of a solver, see the docstring above
function MatsubaraFunctions.unflatten!(
    S::AbstractSolver{Q},
    x::Vector{Q}
)::Nothing where {Q}

    lenΣ = length(S.Σ.data)
    unflatten!(S.F, @view x[1:end-lenΣ])
    unflatten!(S.Σ, @view x[end-lenΣ+1:end])

    return nothing
end

# -----------------------------------------------------------------------------
# Saving and the switched-off symmetrization
# -----------------------------------------------------------------------------

"""
    save!(f, label, S::AbstractSolver) -> nothing

Write the solver to the HDF5 file `f`: the bare propagator, the reference
system (`G0`, `Σ0`, `F0`, `Π0pp`, `Π0ph`) and the target system (`G`, `Σ`, `F`,
`Πpp`, `Πph`), each as a top-level group. `label` is not used.
"""
function MatsubaraFunctions.save!(
    f::HDF5.File,
    label::String,
    S::AbstractSolver
)::Nothing

    MatsubaraFunctions.save!(f, "Gbare", S.Gbare)

    MatsubaraFunctions.save!(f, "G0", S.G0)
    MatsubaraFunctions.save!(f, "Σ0", S.Σ0)
    MatsubaraFunctions.save!(f, "F0", S.F0)
    MatsubaraFunctions.save!(f, "Π0pp", S.Π0pp)
    MatsubaraFunctions.save!(f, "Π0ph", S.Π0ph)

    MatsubaraFunctions.save!(f, "G", S.G)
    MatsubaraFunctions.save!(f, "Σ", S.Σ)
    MatsubaraFunctions.save!(f, "F", S.F)
    MatsubaraFunctions.save!(f, "Πpp", S.Πpp)
    MatsubaraFunctions.save!(f, "Πph", S.Πph)

    return nothing
end

"""
    my_symmetrize!(S::AbstractSolver)

Apply [`my_symmetrize!`](@ref) to every class of the three channels. That
scheme is switched off (`zero_out.jl`), so this changes nothing. Not called
anywhere.
"""
function my_symmetrize!(S :: AbstractSolver)
    my_symmetrize!(S.F.γp.K1, S.SGpp[1])
    my_symmetrize!(S.F.γp.K2, S.SGpp[2])
    my_symmetrize!(S.F.γp.K3, S.SGpp[3])
    my_symmetrize!(S.F.γt.K1, S.SGph[1])
    my_symmetrize!(S.F.γt.K2, S.SGph[2])
    my_symmetrize!(S.F.γt.K3, S.SGph[3])
    my_symmetrize!(S.F.γa.K1, S.SGph[1])
    my_symmetrize!(S.F.γa.K2, S.SGph[2])
    my_symmetrize!(S.F.γa.K3, S.SGph[3])
end
