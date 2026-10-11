# =============================================================================
# Schwinger-Dyson equation (SDE): the self-energy Σ from the full vertex F
# (local solver).
#
# The SDE links the one-particle self-energy to the two-particle vertex 
# (Lihm 2025, Eqs. (S34)-(S35)):
#
#   Σ(ν) - Σ_H = SDE[F],
#   SDE[A](ν) = -U ∑_{Ω,ν′} A(Ω, ν, ν′)_t G(ν + Ω) G(ν′) G(ν′ + Ω)
#
# A(…)_t is the vertex A in the t-channel parametrization (convention.jl).
# Each frequency sum ∑ carries a factor 1/β. Sums are written ∑ in this file
# to keep them apart from the self-energy Σ.
#
# Symbols used below:
#   reference, target
#              two systems with the same fully irreducible vertex. The
#              reference has a known solution (e.g. the DMFT impurity; lower
#              case f, g, σ, π); the target is the one being solved (upper
#              case F, G, Σ, Π).
#   Φ_r        vertex reducible in channel r = a, p, t (Lihm 2025, Eq. (1)).
#              A vertex stores them in its fields `γa`, `γp`, `γt`.
#   Φ̃_r        Φ_r of the target minus Φ_r of the reference.
#   K1, K2     asymptotic classes of Φ_r (channel.jl): K1(Ω) depends only on
#              the bosonic frequency, K2(Ω, ν) also on one fermionic one. Each
#              is stored on its own frequency box.
#   RefVertex  the part of a vertex that is not split into channels: the bare
#              interaction U, plus for DMFT input the "core" (everything
#              beyond K1 and K2). See refvertex.jl.
#   Π_p, Π_a   particle-particle and particle-hole bubble (code: Πpp, Πph;
#              Π_a = Π_t).
#   SG…        symmetry groups (symmetries.jl): they compute one value per set
#              of symmetry-related points and fill in the others.
#   mode       how the work is parallelized (:serial, :threads or :hybrid).
#
# SDE[·] is linear in the vertex, so it can be evaluated piece by piece.
# Writing F = U + (F - U):
#
#   Σ_H   Hartree term U(n - 1/2). Σ is stored without its half-filling value
#         U/2, so only the deviation from half filling appears.
#   Σ_U²  SDE[U]: the second-order diagram U² G G G (Eq. (S46)).
#   Σ_r   SDE[Φ_r] for each channel r, each evaluated in its own channel's
#         parametrization (Eq. (S36)). This uses the kernels
#           Lph(Ω, ν) = [(Φ_a + Φ_t) Π_a U](Ω, ν)   a and t channels together
#           Lpp(Ω, ν) = [U Π_p Φ_p](Ω, ν)           p channel
#         Lpp is "mirrored": written with U on the left instead of the right,
#         which is equal because the vertex is symmetric under ν ↔ ν′
#         (docs/physics_and_notation.md, Sec. 1.5). The kernels are then
#         closed with a single propagator:
#           Σ(ν) = 1/β ∑_Ω [ G(Ω - ν) Lpp(Ω, ν) + G(Ω + ν) Lph(Ω, ν) ]
#   core  the non-bare part F_ref - U of the RefVertex at the bottom of the
#         vertex, evaluated the same way. It is the DMFT core when the
#         reference comes from DMFT, and zero in the parquet approximation
#         (PA), where the reference vertex is just U.
#
# Finite-difference (fd) form. With a reference system (the `strategy`
# options :fdPA, :fdPA_new, :fdPA_1loop, see solve.jl) the SDE is written as a
# difference too:
#
#   Σ = σ + SDE[F](G) - SDE[f](g)
#
# with σ, f, g the reference self-energy, vertex and propagator. Since the
# reference obeys its own SDE, this equals Lihm 2025, Eq. (S38).
#
# The vertex is nested: the field F0 of a vertex holds the vertex it is built
# on. S.F = f + Φ̃_a + Φ̃_p + Φ̃_t stores the reference f as its F0, which in
# turn holds the DMFT channels and the core RefVertex. SDE! walks down
# F → F.F0 → F.F0.F0 → … and adds the channel terms of every level.
#
# Paper ↔ code: ω (bosonic) ↔ Ω; ν̃ (summed) ↔ ω; f, g, σ, π ↔ F0, G0, Σ0, Π0.
# G and Σ are stored as iG and iΣ, and the bubbles as Π = (iG)(iG) (see
# dyson.jl, bubble.jl). With that, the paper's minus signs become the plus
# signs used below.
# =============================================================================

# -----------------------------------------------------------------------------
# Solver interface
# -----------------------------------------------------------------------------

"""
    SDE!(S; strategy = :scPA, include_U² = true, include_Hartree = true, subtract_asymptotic = false)
    SDE!(Σ, G, Πpp, Πph, Lpp, Lph, F, SGΣ, SGpp, SGph; include_U² = true, include_Hartree = true, mode = :threads, subtract_asymptotic = false) -> Σ

Compute the self-energy from the Schwinger-Dyson equation (Lihm 2025,
Eqs. (S34)-(S38)).

Solver method: overwrites `S.Σ`.
- `:scPA`, `:scPA_new`: `Σ = SDE[F](G)`, the SDE of the target vertex.
- `:fdPA`, `:fdPA_new`, `:fdPA_1loop`: the fd form `Σ = σ + SDE[F](G) - SDE[f](g)`,
  with σ = `S.Σ0`, f = `S.F0`, g = `S.G0` and π = `S.Π0pp`/`S.Π0ph` from the
  reference system. The reference SDE uses its own kernel buffers `S.L0pp`,
  `S.L0ph`.

Array method: adds `SDE[F](G)` to `Σ` (stored as iΣ). The vertex is nested, so
it walks down `F → F.F0 → F.F0.F0 → …` and adds the channel terms of every level
(see `SDE_compute!`). The `Σ_U²` and Hartree terms belong to the vertex as a
whole and are added only once, at the top level.
"""
function SDE!(S :: AbstractSolver; strategy = :scPA, include_U² = true, include_Hartree = true, subtract_asymptotic = false)
    if strategy == :scPA || strategy == :scPA_new
        # Σ = SDE[F](G): the target vertex with the target propagator
        set!(S.Σ, 0)
        SDE!(S.Σ, S.G, S.Πpp, S.Πph, S.Lpp, S.Lph, S.F, S.SGΣ, S.SGpp[2], S.SGph[2]; include_U², include_Hartree, S.mode, subtract_asymptotic)

    elseif strategy == :fdPA || strategy == :fdPA_new || strategy == :fdPA_1loop
        # Σ = SDE[F](G): the target vertex with the target propagator
        set!(S.Σ, 0)
        SDE!(S.Σ, S.G, S.Πpp, S.Πph, S.Lpp, S.Lph, S.F, S.SGΣ, S.SGpp[2], S.SGph[2]; include_U², include_Hartree, S.mode, subtract_asymptotic)
        #   - SDE[f](g): the same for the reference system
        mult_add!(S.Σ, SDE!(copy(S.Σ)*0, S.G0, S.Π0pp, S.Π0ph, S.L0pp, S.L0ph, S.F0, S.SGΣ, S.SG0pp2, S.SG0ph2; include_U², include_Hartree, S.mode, subtract_asymptotic), -1)
        #   + σ: the reference self-energy
        add!(S.Σ, S.Σ0)

        # Not used: the same from the p-channel K1 + K2 only (see SDE_using_K12!)
        # SDE_using_K12!(S)
        # add!(S.Σ, SDE_using_K12!(copy(S.Σ), S.G - S.G0, S.F0, S.SGΣ; S.mode, include_Hartree = false))
        # add!(S.Σ, SDE_using_K12!(copy(S.Σ), S.G0, S.F0, S.SGΣ; S.mode, include_Hartree = false))
    else
        throw(ArgumentError("Calculation strategy $strategy unknown"))
    end
end

# Array method of SDE!, see the docstring above.
function SDE!(Σ, G, Πpp, Πph, Lpp, Lph, F, SGΣ, SGpp, SGph; include_U² = true, include_Hartree = true, mode = :threads, subtract_asymptotic = false)
    # Apply the SDE to F and all of its reference vertices F.F0, F.F0.F0, ...

    # current level (SDE_compute! fills its first argument, a scratch copy)
    add!(Σ, SDE_compute!(copy(Σ), G, Πpp, Πph, Lpp, Lph, F, SGΣ, SGpp, SGph; mode, include_U², include_Hartree, subtract_asymptotic))

    if F isa AbstractVertex
        # Next level down: the vertex F0 that F is built on.
        # Σ_U² and the Hartree term are already included, so switch them off.
        SDE!(Σ, G, Πpp, Πph, Lpp, Lph, F.F0, SGΣ, SGpp, SGph; include_U² = false, include_Hartree = false, mode, subtract_asymptotic)
    end

    return Σ
end


# Old solver-level wrappers, superseded by SDE!(S; strategy):
# function SDE!(Σ, S :: AbstractSolver; include_U² = true, include_Hartree = true)
#     SDE_compute!(Σ, S.G, S.Πpp, S.Πph, S.Lpp, S.Lph, S.F, S.SGΣ, S.SGpp[2], S.SGph[2]; S.mode, include_U², include_Hartree)
#     return Σ
# end

# function SDE(S :: AbstractSolver; include_U² = true, include_Hartree = true)
#     SDE!(copy(S.Σ), S; include_U², include_Hartree)
# end

"""
    SDE_U2(S) -> Σ_U²

Second-order self-energy of the solver `S`, see [`SDE_U2_using_G`](@ref).
"""
function SDE_U2(S :: AbstractSolver)
    SDE_U2_using_G(S)
end

# Solver method of SDE_U2_using_Π, see below.
function SDE_U2_using_Π(S :: AbstractSolver)
    SDE_U2_using_Π(S.Σ, S.G, S.Πpp, S.Πph, S.SGΣ, bare_vertex(S.F); S.mode)
end

# Solver method of SDE_U2_using_G, see below.
function SDE_U2_using_G(S :: AbstractSolver)
    SDE_U2_using_G(S.Σ, S.G, S.SGΣ, bare_vertex(S.F); S.mode)
end

# Solver method of SDE_using_K12!, see below.
function SDE_using_K12!(S :: AbstractSolver; include_Hartree = true)
    SDE_using_K12!(S.Σ, S.G, S.F, S.SGΣ; S.mode, include_Hartree)
end

# -----------------------------------------------------------------------------
# Channel kernels L (local vertex)
# -----------------------------------------------------------------------------
# Each kernel is a vertex closed with one bubble and one bare vertex U. It has
# the shape of a K2 class (bosonic Ω, fermionic ν) and lives on the K2 box.
# Not to be confused with FL, the "left part" of the Bethe-Salpeter equation
# (BSE_templates.jl).

"""
    SDE_channel_L_pp!(Lpp, Πpp, F, SGpp2; mode, subtract_asymptotic = false) -> nothing

Build the p-channel kernel of the SDE for one vertex level `F` and write it
into `Lpp`:

    Lpp(Ω, ν) = U · 1/β ∑_ν̃ Π_p(Ω, ν̃) Φ_p(Ω, Ω - ν̃, ν)

This is the mirrored form of `[Φ_p Π_p U](q, k)` in Lihm 2025, Eq. (S36c). In
the p channel the vertex's internal leg sits at Ω - ν̃ (Eq. (S15)). Closing the
kernel with `G(Ω - ν)` gives the p-channel part of Σ (see `SDE_compute!`).

- For a `Vertex`, `Φ_p` is the vertex's own p channel. Its `F0` and the other
  channels are left out; they are handled at their own level of the chain.
- For a `RefVertex`, which has no channels, its whole non-bare part
  `F_ref - U` is used.

`subtract_asymptotic = true` is not implemented for the local solver.
"""
function SDE_channel_L_pp!(
    Lpp   :: MF_K2{Q},
    Πpp   :: MF_Π{Q},
    F     :: AbstractVertex{Q},
    SGpp2 :: SymmetryGroup
    ;
    mode  :: Symbol,
    subtract_asymptotic :: Bool = false,
    )     :: Nothing where {Q}

    subtract_asymptotic && error("Subtracting asymptotic contribution not implemented")

    # Lpp(Ω, ν) for one (Ω, ν): sum over the internal frequency ν̃ (called ω)
    @inline function diagram(wtpl)

        Ω, ν    = wtpl
        val     = zero(Q)
        Πslice  = view(Πpp, Ω, :)

        for i in eachindex(Πslice)
            ω = value(meshes(Πpp, Val(2))[i])

            # U Π_p(Ω, ν̃) Φ_p(Ω, Ω - ν̃, ν), with only the vertex's own p
            # channel switched on. Older, equivalent form of the same line:
            # val += bare_vertex(F) * Πslice[i] * F.γp(Ω, Ω - ω, ν)
            val += bare_vertex(F) * Πslice[i] * (
                F(Ω, Ω - ω, ν, pCh, pSp; γp = true, F0 = false, γa = false, γt = false) )
        end

        return temperature(F) * val
    end

    # evaluate one point per symmetry class; the symmetry group fills in the rest
    SGpp2(Lpp, InitFunction{2, Q}(diagram); mode = mode)

    return nothing
end

"""
    SDE_channel_L_ph!(Lph, Πph, F, SGph2; mode, subtract_asymptotic = false) -> nothing

Build the particle-hole kernel of the SDE for one vertex level `F` and write it
into `Lph`. The a and t channels are added, since both are closed with the same
propagator `G(Ω + ν)` (Lihm 2025, Eqs. (S36a-b)):

    Lph(Ω, ν) = 1/β ∑_ν̃ [Φ_a(Ω, ν, ν̃) + Φ_t(Ω, ν, ν̃)] Π_a(Ω, ν̃) U

(`Π_a = Π_t`). For a `Vertex` only its own a and t channels are used; for a
`RefVertex` its non-bare part `F_ref - U`, once in the a and once in the t
parametrization. See [`SDE_channel_L_pp!`](@ref).
"""
function SDE_channel_L_ph!(
    Lph   :: MF_K2{Q},
    Πph   :: MF_Π{Q},
    F     :: AbstractVertex{Q},
    SGph2 :: SymmetryGroup
    ;
    mode  :: Symbol,
    subtract_asymptotic :: Bool = false,
    )     :: Nothing where {Q}

    subtract_asymptotic && error("Subtracting asymptotic contribution not implemented")

    # Lph(Ω, ν) for one (Ω, ν): sum over the internal frequency ν̃ (called ω)
    @inline function diagram(wtpl)

        Ω, ν    = wtpl
        val     = zero(Q)
        Πslice  = view(Πph, Ω, :)

        for i in eachindex(Πslice)
            ω = value(meshes(Πph, Val(2))[i])

            # U Π_a(Ω, ν̃) [Φ_a + Φ_t](Ω, ν, ν̃), with only the vertex's own a
            # and t channels switched on. Older, equivalent form of the same line:
            # val += bare_vertex(F) * Πslice[i] * (F.γt(Ω, ν, ω) + F.γa(Ω, ν, ω))
            val += bare_vertex(F) * Πslice[i] * (
                F(Ω, ν, ω, aCh, pSp; γa = true, F0 = false, γp = false, γt = false)
              + F(Ω, ν, ω, tCh, pSp; γt = true, F0 = false, γp = false, γa = false) )
        end

        return temperature(F) * val
    end

    # evaluate one point per symmetry class; the symmetry group fills in the rest
    SGph2(Lph, InitFunction{2, Q}(diagram); mode)

    return nothing
end

# RefVertex method of SDE_channel_L_pp!: U Π (F - U), using the whole non-bare
# part of the RefVertex.
function SDE_channel_L_pp!(
    Lpp   :: MF_K2{Q},
    Πpp   :: MF_Π{Q},
    F     :: RefVertex{Q},
    SGpp2 :: SymmetryGroup
    ;
    mode  :: Symbol,
    subtract_asymptotic :: Bool = false,
    )     :: Nothing where {Q}

    # Lpp(Ω, ν) for one (Ω, ν): sum over the internal frequency ν̃ (called ω)
    @inline function diagram(wtpl)

        Ω, ν    = wtpl
        val     = zero(Q)
        Πslice  = view(Πpp, Ω, :)

        for i in eachindex(Πslice)
            ω = value(meshes(Πpp, Val(2))[i])

            # U Π_p(Ω, ν̃) [F_ref - U](Ω, Ω - ν̃, ν) in the p parametrization
            val += bare_vertex(F) * Πslice[i] * (F(Ω, Ω - ω, ν, pCh, pSp) - bare_vertex(F, pCh, pSp))
        end

        return temperature(F) * val
    end

    # evaluate one point per symmetry class; the symmetry group fills in the rest
    SGpp2(Lpp, InitFunction{2, Q}(diagram); mode = mode)

    return nothing
end

# RefVertex method of SDE_channel_L_ph!: U Π (F - U), once in the a and once
# in the t parametrization.
function SDE_channel_L_ph!(
    Lph   :: MF_K2{Q},
    Πph   :: MF_Π{Q},
    F     :: RefVertex{Q},
    SGph2 :: SymmetryGroup
    ;
    mode  :: Symbol,
    subtract_asymptotic :: Bool = false,
    )     :: Nothing where {Q}

    # Lph(Ω, ν) for one (Ω, ν): sum over the internal frequency ν̃ (called ω)
    @inline function diagram(wtpl)

        Ω, ν    = wtpl
        val     = zero(Q)
        Πslice  = view(Πph, Ω, :)

        for i in eachindex(Πslice)
            ω = value(meshes(Πph, Val(2))[i])

            # U Π_a(Ω, ν̃) [(F_ref - U)_a + (F_ref - U)_t](Ω, ν, ν̃)
            val += bare_vertex(F) * Πslice[i] * (F(Ω, ν, ω, aCh, pSp) + F(Ω, ν, ω, tCh, pSp) - bare_vertex(F, aCh, pSp) - bare_vertex(F, tCh, pSp))
        end

        return temperature(F) * val
    end

    # evaluate one point per symmetry class; the symmetry group fills in the rest
    SGph2(Lph, InitFunction{2, Q}(diagram); mode)

    return nothing
end

# -----------------------------------------------------------------------------
# Self-energy of one vertex level
# -----------------------------------------------------------------------------

"""
    SDE_compute!(Σ, G, Πpp, Πph, Lpp, Lph, F, SGΣ, SGpp2, SGph2; mode, include_U² = true, include_Hartree = true, subtract_asymptotic = false) -> Σ

Self-energy contribution of one vertex level `F` (a `Vertex` or a `RefVertex`),
written into `Σ` (stored as iΣ):

1. Build the channel kernels `Lpp`, `Lph` of `F` (`SDE_channel_L_pp!`,
   `SDE_channel_L_ph!`).
2. Close them with one propagator (Lihm 2025, Eq. (S36)):

       iΣ(ν) = 1/β ∑_Ω [ iG(Ω - ν) Lpp(Ω, ν) + iG(Ω + ν) Lph(Ω, ν) ]

   Ω and ν run over the K2 box, where the kernels live. For ν outside it, the
   channel part is zero.
3. For a `RefVertex` the same vertex was used in all three channel forms, so
   the result is divided by 3.
4. Optionally add the second-order term `Σ_U²` (Eq. (S46)) and the Hartree term
   `U (n - 1/2)`.

`SGΣ`, `SGpp2`, `SGph2` are the symmetry groups of Σ and of the two kernels.
"""
function SDE_compute!(
    Σ     :: MF_G{Q},
    G     :: MF_G{Q},
    Πpp   :: MF_Π{Q},
    Πph   :: MF_Π{Q},
    Lpp   :: MF_K2{Q},
    Lph   :: MF_K2{Q},
    F     :: Union{AbstractVertex{Q}, RefVertex{Q}},
    SGΣ   :: SymmetryGroup,
    SGpp2 :: SymmetryGroup,
    SGph2 :: SymmetryGroup,
    ;
    mode  :: Symbol,
    include_U² = true,
    include_Hartree = true,
    subtract_asymptotic = false,
    )     :: MF_G{Q} where {Q}
    # contribution of this level's own channels Φ_a, Φ_p, Φ_t

    SDE_channel_L_pp!(Lpp, Πpp, F, SGpp2; mode, subtract_asymptotic)
    SDE_channel_L_ph!(Lph, Πph, F, SGph2; mode, subtract_asymptotic)

    # iΣ(ν) for one ν: close the kernels with one propagator, sum over Ω
    @inline function diagram(wtpl)

        ν   = wtpl[1]
        val = zero(Q)

        if is_inbounds(ν, meshes(Lpp, Val(2)))
            Lppslice = view(Lpp, :, ν)
            Lphslice = view(Lph, :, ν)

            for i in eachindex(Lppslice)
                Ω = value(meshes(Lpp, Val(1))[i])

                val += G(Ω - ν) * Lppslice[i]    # p channel: G(Ω - ν)
                val += G(Ω + ν) * Lphslice[i]    # a and t channels: G(Ω + ν)
            end
        end

        return temperature(F) * val
    end

    # evaluate one ν per symmetry class; the symmetry group fills in the rest
    SGΣ(Σ, InitFunction{1, Q}(diagram); mode)

    if F isa RefVertex
        # A RefVertex has no channel structure, so the same vertex was used in
        # the p, a and t forms, i.e. counted three times. Divide by 3.
        Σ.data .*= 1/3
    end

    if include_U²
        # second-order diagram SDE[U], Eq. (S46)
        Σ_U² = SDE_U2_using_G(Σ, G, SGΣ, bare_vertex(F); mode)
        add!(Σ, Σ_U²)
    end

    if include_Hartree
        # Hartree term U (n - 1/2); n is the density per spin
        n = compute_occupation(G)
        # We store im * Σ in S.Σ, so we multiply im.
        Σ.data .+= Q((n - 1/2) * bare_vertex(F) * im)
    end

    return Σ
end

# -----------------------------------------------------------------------------
# Second-order self-energy
# -----------------------------------------------------------------------------

"""
    SDE_U2_using_Π(Σ, G, Πpp, Πph, SGΣ, U; mode) -> Σ_U²
    SDE_U2_using_Π(S)

The second-order self-energy (see [`SDE_U2_using_G`](@ref)), built from the
bubbles. The diagram is closed once through the particle-particle bubble and
once through the particle-hole bubble, and the two forms are averaged:

    iΣ_U²(ν) = U²/2 · 1/β² ∑_{Ω,ν̃} [ iG(Ω - ν) Πpp(Ω, ν̃) + iG(Ω + ν) Πph(Ω, ν̃) ]

The two forms are the same diagram with relabeled frequencies and differ only
through the finite frequency boxes. Returns a new array (`Σ` only gives the
shape). Not used by the solvers, which call `SDE_U2_using_G`.
"""
function SDE_U2_using_Π(
    Σ   :: MF_G{Q},
    G   :: MF_G{Q},
    Πpp :: MF_Π{Q},
    Πph :: MF_Π{Q},
    SGΣ :: SymmetryGroup,
    U   :: Number,
    ;
    mode  :: Symbol,
    )   :: MF_G{Q} where {Q}
    # 2nd-order perturbative contribution to the self-energy

    T = temperature(meshes(Σ, Val(1)))
    Σ_U² = copy(Σ)
    set!(Σ_U², 0)

    # bubbles summed over their fermionic frequency: ∑_ν̃ Π(Ω, ν̃)
    Πppsum = MeshFunction((meshes(Πpp, Val(1)),), dropdims(sum(Πpp.data, dims=2), dims=2))
    Πphsum = MeshFunction((meshes(Πph, Val(1)),), dropdims(sum(Πph.data, dims=2), dims=2))

    # iΣ_U²(ν) for one ν
    @inline function diagram(wtpl)

        ν   = wtpl[1]
        val = zero(Q)

        for i in eachindex(meshes(Πppsum, Val(1)))

            Ω = value(meshes(Πppsum, Val(1))[i])
            val += G(Ω - ν) * Πppsum[Ω] + G(Ω + ν) * Πphsum[Ω]

        end

        # U² · (1/β)² for the two frequency sums, /2 for the average of the two forms
        return val * U^2 * T^2 / 2
    end

    # evaluate one ν per symmetry class; the symmetry group fills in the rest
    SGΣ(Σ_U², InitFunction{1, Q}(diagram); mode)

    return Σ_U²
end


"""
    SDE_U2_using_G(Σ, G, SGΣ, U; mode = :threads) -> Σ_U²
    SDE_U2_using_G(S)

Second-order self-energy, i.e. the SDE of the bare vertex U alone (Lihm 2025,
Eq. (S46)):

    iΣ_U²(ν) = U² · 1/β² ∑_{ν1,ν2} iG(ν1) iG(ν2) iG(ν1 - ν2 + ν)

computed directly from the propagator. G is stored as iG; with that the
paper's minus sign drops out. Both sums run over the G mesh. Returns a new
array (`Σ` only gives the shape).
"""
function SDE_U2_using_G(
    Σ   :: MF_G{Q},
    G   :: MF_G{Q},
    SGΣ :: SymmetryGroup,
    U   :: Number,
    ;
    mode  :: Symbol = :threads,
    )   :: MF_G{Q} where {Q}
    # 2nd-order perturbative contribution to the self-energy

    T = temperature(meshes(Σ, Val(1)))
    Σ_U² = copy(Σ)
    set!(Σ_U², 0)

    # iΣ_U²(ν) for one ν: double sum over the two internal frequencies
    @inline function diagram(wtpl)

        ν   = wtpl[1]
        val = zero(Q)

        for i2 in eachindex(meshes(G, Val(1))), i1 in eachindex(meshes(G, Val(1)))
            ω1 = value(meshes(G, Val(1))[i1])
            ω2 = value(meshes(G, Val(1))[i2])

            # the third propagator must also lie on the G mesh
            if is_inbounds(ω1 - ω2 + ν, meshes(G, Val(1)))
                val += G[ω1] * G[ω2] * G[ω1 - ω2 + ν]
            end
        end

        return val * U^2 * T^2
    end

    # evaluate one ν per symmetry class; the symmetry group fills in the rest
    SGΣ(Σ_U², InitFunction{1, Q}(diagram); mode)

    return Σ_U²
end

# -----------------------------------------------------------------------------
# Diagnostics
# -----------------------------------------------------------------------------
# Not used by the solvers; useful as independent checks of Σ.

"""
    SDE_using_K12!(Σ, G, F, SGΣ; mode, include_Hartree = true) -> Σ
    SDE_using_K12!(S; include_Hartree = true)

Self-energy from only the K1 and K2 classes of the p channel, written into `Σ`.

The SDE is linear, so it can be written entirely in the p-channel
parametrization (Lihm 2025, Eq. (S36c) applied to the full vertex). With the
full vertex F on one side, the kernel `[F Π_p U](Ω, ν)` is exactly
`K1_p(Ω) + K2_p(Ω, ν)`: this is the p-channel Bethe-Salpeter equation with one
outer leg at infinity (Eqs. (S23a-b), mirrored; compare Eq. (S57)). This gives

    iΣ(ν) = 1/β ∑_ν̃ iG(ν̃) [ K1_p(ν + ν̃) + K2_p(ν + ν̃, ν) ]   (+ Hartree)

It reads K1 and K2 of the vertex's own p channel directly, so for an fd vertex
only the difference `Φ̃_p` is used.
Only called from commented-out code in [`SDE!`](@ref).
"""
function SDE_using_K12!(
    Σ :: MF_G{Q},
    G :: MF_G{Q},
    F :: Vertex{Q},
    SGΣ :: SymmetryGroup,
    ;
    mode :: Symbol,
    include_Hartree = true,
    ) :: MF_G{Q} where {Q}

    # iΣ(ν) for one ν: sum over the internal frequency ν̃ (called ω)
    @inline function diagram(wtpl)

        ν   = wtpl[1]
        val = zero(Q)

        for iω in eachindex(meshes(G, Val(1)))
            ω = value(meshes(G, Val(1))[iω])
            # SDE using only K1 + K2 in p channel (bosonic frequency ν + ν̃)
            val += G[ω] * (F.γp.K1(ν + ω) + F.γp.K2(ν + ω, ν))
        end

        return temperature(F) * val
    end

    # evaluate one ν per symmetry class; the symmetry group fills in the rest
    SGΣ(Σ, InitFunction{1, Q}(diagram); mode)

    if include_Hartree
        n = compute_occupation(G)
        # We store im * Σ in S.Σ, so we multiply im.
        Σ.data .+= Q((n - 1/2) * bare_vertex(F) * im)
    end

    return Σ
end

"""
    self_energy_sanity_check(Σ) -> Bool

Check that a local self-energy is causal: Im Σ(iν) ≤ 0 for ν > 0 and ≥ 0 for
ν < 0. Since iΣ is stored, `real(Σ[ν]) = -Im Σ(iν)`, so the stored real part
must be ≥ 0 for ν > 0 and ≤ 0 for ν < 0. Warns at every frequency that
violates this and returns whether all passed.
"""
function self_energy_sanity_check(Σ)
    passed = true
    # sanity check
    for ν in meshes(Σ, Val(1))
        if plain_value(ν) > 0 && real(Σ[ν]) < 0
            passed = false
            @warn "Σ violates causality at n = $(index(ν))"
        elseif plain_value(ν) < 0 && real(Σ[ν]) > 0
            passed = false
            @warn "Σ violates causality at n = $(index(ν))"
        end
    end
    passed
end
