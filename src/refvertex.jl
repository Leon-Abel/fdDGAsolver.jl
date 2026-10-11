# =============================================================================
# RefVertex: the part of a vertex that is not split into channels.
#
# A vertex in this code is a RefVertex plus the reducible parts Φ_r of the
# three channels (vertex.jl). The RefVertex holds what belongs to no channel:
#   - the bare interaction U, i.e. the bare vertex F₀ (Lihm 2025, Eq. (S3)),
#   - optionally a frequency-dependent "core" on a K3-sized box.
#
# Parquet approximation (PA): the fully irreducible vertex Λ is replaced by U,
# so the RefVertex is just U with an empty core.
#
# DMFT input (utility/load_triqs.jl): the DMFT vertex is written as
#   f = U + core + ∑_r (K1_r + K2_r + K̄2_r).
# The core collects everything beyond the K1 and K2 classes, (Λ - U) + ∑_r K3_r,
# and is never split further. The RefVertex as a whole is then Λ + ∑_r K3_r. It
# is zero outside its box.
#
# Storage: the core is one function, but it is kept on two frequency boxes, in
# the p- and in the t-channel parametrization (convention.jl), each for the
# parallel (pSp) and the crossed (xSp) spin component:
#   Fp_p, Fp_x   p channel, F^= and F^×
#   Ft_p, Ft_x   t channel, F^= and F^×
# Evaluating in channel r reads the box of that channel. The a channel follows
# from the t channel by crossing symmetry (Eq. (S11)).
#
# Paper ↔ code: F₀ ↔ the field U (bare_vertex); ω ↔ Ω; ν′ ↔ νp.
# =============================================================================

"""
    RefVertex{Q}

The part of a vertex that is not split into channels: the bare interaction
`U` plus an optional core, with number type `Q` (see the file header).

Fields: `U`, and the core `Fp_p`, `Fp_x` (p-channel parametrization, F^= and
F^×) and `Ft_p`, `Ft_x` (t-channel parametrization). The a channel is not
stored; it follows from the t channel by crossing symmetry.

Constructors:
- `RefVertex(U, Fp_p, Fp_x, Ft_p, Ft_x)`: from a given core.
- `RefVertex(T, U, Q = ComplexF64)`: just the bare vertex U at temperature `T`
  (core zero, on one-point meshes). Used for the PA reference and for buffers.
"""
struct RefVertex{Q}
    U    :: Q           # bare interaction
    Fp_p :: MF_K3{Q}    # core in the p-channel parametrization, parallel spin component
    Fp_x :: MF_K3{Q}    # core in the p-channel parametrization, crossed spin component
    Ft_p :: MF_K3{Q}    # core in the t-channel parametrization, parallel spin component
    Ft_x :: MF_K3{Q}    # core in the t-channel parametrization, crossed spin component
    # The a channel is not stored independently; it is recovered via crossing symmetry

    # Construct from a given core.
    function RefVertex(
        U    :: Number,
        Fp_p :: MF_K3{Q},
        Fp_x :: MF_K3{Q},
        Ft_p :: MF_K3{Q},
        Ft_x :: MF_K3{Q},
        )    :: RefVertex{Q} where {Q}

        return new{Q}(Q(U), Fp_p, Fp_x, Ft_p, Ft_x)
    end

    # Construct the bare vertex U only: zero core on one-point Matsubara meshes.
    function RefVertex(
        T :: Float64,               # temperature
        U :: Float64,               # bare interaction
          :: Type{Q} = ComplexF64,  # number type of the data (default ComplexF64)
        ) where {Q}

        # Empty core: meshes with a single frequency, all values zero
        gΩ = MatsubaraMesh(T, 1, Boson)     # bosonic mesh with 1 frequency point
        gν = MatsubaraMesh(T, 1, Fermion)   # fermionic mesh with 1 frequency point
        Fp = MeshFunction(gΩ, gν, gν; data_t = Q)
        Fx = MeshFunction(gΩ, gν, gν; data_t = Q)
        set!(Fp, 0) # initialise all entries to zero
        set!(Fx, 0)

        # the p and t channels use meshes of the same shape; copy gives each its own storage
        return new{Q}(Q(U), Fp, Fx, copy(Fp), copy(Fx)) :: RefVertex{Q}
    end
end

# -----------------------------------------------------------------------------
# Utilities
# -----------------------------------------------------------------------------

# number type of the stored data
Base.eltype(::Type{<: RefVertex{Q}}) where {Q} = Q

# Printing: type, U, temperature and the size of the core's box.
function Base.show(io::IO, Γ::RefVertex{Q}) where {Q}
    print(io, "$(nameof(typeof(Γ))){$Q}, U = $(Γ.U), T = $(temperature(Γ)), K3 : $(numK3(Γ))")
end

"""
    numK3(Λ::RefVertex) -> (nΩ, nν)

Size of the core's box: the number of positive bosonic and fermionic
frequencies. It is named after K3 because the core has the shape of a K3 class.
"""
function numK3(
    Λ :: RefVertex
    ) :: NTuple{2, Int64}

    # Both Fp_p and Fp_x share the same mesh, so querying Fp_p is sufficient.
    # Val(1) → bosonic mesh, Val(2) → first fermionic mesh
    return N(meshes(Λ.Fp_p, Val(1))), N(meshes(Λ.Fp_p, Val(2)))
end

"""
    set!(F::RefVertex, val::Number) -> nothing

Set every entry of the four core arrays to `val`, e.g. `set!(F, 0)` to clear
the core. `U` is not changed.
"""
function MatsubaraFunctions.set!(
    F :: RefVertex,
    val :: Number,
    ) :: Nothing

    set!(F.Fp_p, val)
    set!(F.Fp_x, val)
    set!(F.Ft_p, val)
    set!(F.Ft_x, val)

    return nothing
end

# -----------------------------------------------------------------------------
# Getters
# -----------------------------------------------------------------------------

"""
    temperature(F::RefVertex) -> Float64

Temperature T = 1/β, read from the bosonic mesh of the core (all four arrays
use the same temperature).
"""
function MatsubaraFunctions.temperature(
    F :: RefVertex
    ) :: Float64

    return MatsubaraFunctions.temperature(meshes(F.Fp_p, Val(1)))
end

# -----------------------------------------------------------------------------
# Copy and comparison
# -----------------------------------------------------------------------------

"""
    copy(F::RefVertex) -> RefVertex

Independent copy, with new arrays for all four core components.
"""
function Base.:copy(
    F :: RefVertex{Q}
    ) :: RefVertex{Q} where {Q}

    return RefVertex(F.U, copy(F.Fp_p), copy(F.Fp_x), copy(F.Ft_p), copy(F.Ft_x))

end

"""
    ==(F1::RefVertex, F2::RefVertex) -> Bool

True if `U` and all four core arrays are equal entry by entry.
"""
function Base.:(==)(
    F1 :: RefVertex{Q},
    F2 :: RefVertex{Q},
    )  :: Bool where {Q}
    return (F1.U == F2.U) && (F1.Fp_p == F2.Fp_p) && (F1.Fp_x == F2.Fp_x) && (F1.Ft_p == F2.Ft_p) && (F1.Ft_x == F2.Ft_x)
end

# -----------------------------------------------------------------------------
# Evaluation: parallel spin component (pSp)
# -----------------------------------------------------------------------------
# A RefVertex is called like a function. The value is the bare part (+U for
# pSp, -U for xSp) plus the core, read from the box of the requested channel.
# Outside its box the core is zero. Keyword arguments (the channel switches of
# a full vertex) are accepted and ignored.

"""
    (F::RefVertex)(Ω, ν, νp, ::Type{Ch}, ::Type{Sp}; kwargs...) -> value

Value of the reference vertex at frequencies given in channel `Ch`'s
parametrization, for spin component `Sp`:

    pSp:  +U + core^=,     xSp:  -U + core^×,     dSp = 2·pSp + xSp

The p and t channels read their own core arrays. The a channel uses crossing
symmetry with the t channel (Lihm 2025, Eq. (S11)):
`F^=_a(Ω, ν, ν′) = -F^×_t(Ω, ν′, ν)` and `F^×_a(Ω, ν, ν′) = -F^=_t(Ω, ν′, ν)`.
If `ν` or `νp` is [`νInf`](@ref), only the bare part remains, since the core
vanishes at infinity.
"""
@inline function (F :: RefVertex{Q})(
    Ω  :: MatsubaraFrequency,
    ν  :: MatsubaraFrequency,
    νp :: MatsubaraFrequency,
       :: Type{pCh},
       :: Type{pSp},
    ; kwargs...
    )  :: Q where {Q}

    # p channel, parallel spin: core from the p-channel box plus U
    return F.Fp_p(Ω, ν, νp) + F.U
end

# t channel, parallel spin: core from the t-channel box plus U
@inline function (F :: RefVertex{Q})(
    Ω  :: MatsubaraFrequency,
    ν  :: MatsubaraFrequency,
    νp :: MatsubaraFrequency,
       :: Type{tCh},
       :: Type{pSp},
    ; kwargs...
    )  :: Q where {Q}

    return F.Ft_p(Ω, ν, νp) + F.U
end

# a channel, parallel spin: crossing symmetry with the t channel,
# F^=_a(Ω, ν, ν′) = -F^×_t(Ω, ν′, ν) (note the swap ν ↔ ν′ and the minus sign)
@inline function (F :: RefVertex{Q})(
    Ω  :: MatsubaraFrequency,
    ν  :: MatsubaraFrequency,
    νp :: MatsubaraFrequency,
       :: Type{aCh},
       :: Type{pSp},
    ; kwargs...
    )  :: Q where {Q}

    return -F.Ft_x(Ω, νp, ν) + F.U
end

# -----------------------------------------------------------------------------
# Evaluation: crossed spin component (xSp)
# -----------------------------------------------------------------------------
# The bare part of the crossed component is -U (Lihm 2025, Eq. (S3)).

# p channel, crossed spin
@inline function (F :: RefVertex{Q})(
    Ω  :: MatsubaraFrequency,
    ν  :: MatsubaraFrequency,
    νp :: MatsubaraFrequency,
       :: Type{pCh},
       :: Type{xSp},
    ; kwargs...
    )  :: Q where {Q}

    return F.Fp_x(Ω, ν, νp) - F.U
end

# t channel, crossed spin
@inline function (F :: RefVertex{Q})(
    Ω  :: MatsubaraFrequency,
    ν  :: MatsubaraFrequency,
    νp :: MatsubaraFrequency,
       :: Type{tCh},
       :: Type{xSp},
    ; kwargs...
    )  :: Q where {Q}

    return F.Ft_x(Ω, ν, νp) - F.U
end

# a channel, crossed spin: crossing symmetry with the t channel,
# F^×_a(Ω, ν, ν′) = -F^=_t(Ω, ν′, ν) (note the swap ν ↔ ν′ and the minus sign)
@inline function (F :: RefVertex{Q})(
    Ω  :: MatsubaraFrequency,
    ν  :: MatsubaraFrequency,
    νp :: MatsubaraFrequency,
       :: Type{aCh},
       :: Type{xSp},
    ; kwargs...
    )  :: Q where {Q}

    return -F.Ft_p(Ω, νp, ν) - F.U
end

# -----------------------------------------------------------------------------
# Evaluation: density component (dSp)
# -----------------------------------------------------------------------------
# dSp = 2·pSp + xSp (types.jl), built from the two evaluators above. In the t
# channel this is the density vertex F_D.

@inline function (F :: RefVertex{Q})(
    Ω  :: MatsubaraFrequency,
    ν  :: MatsubaraFrequency,
    νp :: MatsubaraFrequency,
       :: Type{Ch},
       :: Type{dSp},
    ; kwargs...
    )  :: Q where {Q, Ch <: ChannelTag}

    return 2 * F(Ω, ν, νp, Ch, pSp) + F(Ω, ν, νp, Ch, xSp)
end

# -----------------------------------------------------------------------------
# Evaluation: infinite frequencies
# -----------------------------------------------------------------------------
# If a fermionic frequency is νInf, the core is zero and only the bare vertex
# of the spin component remains.

# ν → ∞, ν′ finite
@inline function (F :: RefVertex{Q})(
    Ω  :: MatsubaraFrequency,
    ν  :: InfiniteMatsubaraFrequency,
    νp :: MatsubaraFrequency,
       :: Type{Ch},
       :: Type{Sp},
    ; kwargs...
    )  :: Q where {Q, Ch <: ChannelTag, Sp <: SpinTag}

    return bare_vertex(F, Ch, Sp)
end

# ν′ → ∞, ν finite
@inline function (F :: RefVertex{Q})(
    Ω  :: MatsubaraFrequency,
    ν  :: MatsubaraFrequency,
    νp :: InfiniteMatsubaraFrequency,
       :: Type{Ch},
       :: Type{Sp},
    ; kwargs...
    )  :: Q where {Q, Ch <: ChannelTag, Sp <: SpinTag}

    return bare_vertex(F, Ch, Sp)
end

# both fermionic frequencies → ∞
@inline function (F :: RefVertex{Q})(
    Ω  :: MatsubaraFrequency,
    ν  :: InfiniteMatsubaraFrequency,
    νp :: InfiniteMatsubaraFrequency,
       :: Type{Ch},
       :: Type{Sp},
    ; kwargs...
    )  :: Q where {Q, Ch <: ChannelTag, Sp <: SpinTag}

    return bare_vertex(F, Ch, Sp)
end

# -----------------------------------------------------------------------------
# Bare vertex
# -----------------------------------------------------------------------------

"""
    bare_vertex(F::RefVertex) -> U
    bare_vertex(F::RefVertex, ::Type{Sp}) -> value
    bare_vertex(F::RefVertex, ::Type{Ch}, ::Type{Sp}) -> value

The bare vertex F₀ in spin component `Sp` (Lihm 2025, Eq. (S3)): `+U` for
`pSp`, `-U` for `xSp`, and `2U - U = +U` for `dSp`. Without `Sp` it returns U.
It does not depend on the channel; the three-argument form exists only for
convenience (marked for removal, but still used by the SDE).
"""
@inline bare_vertex(F :: RefVertex) =  F.U                  # default: return U
@inline bare_vertex(F :: RefVertex, :: Type{pSp}) =  F.U    # parallel component: +U
@inline bare_vertex(F :: RefVertex, :: Type{xSp}) = -F.U    # crossed component:  -U
@inline bare_vertex(F :: RefVertex, :: Type{dSp}) =  F.U    # density: 2(+U) + (-U) = +U
@inline bare_vertex(F :: RefVertex, :: Type{Ch}, :: Type{Sp}) where {Ch <: ChannelTag, Sp <: SpinTag} = bare_vertex(F, Sp)   # TODO: remove

# -----------------------------------------------------------------------------
# HDF5 input and output
# -----------------------------------------------------------------------------

"""
    save!(file, label, F::RefVertex) -> nothing

Write a RefVertex to the HDF5 group `label`: `U` as an attribute of the group,
the four core arrays as subgroups `Fp_p`, `Fp_x`, `Ft_p`, `Ft_x`.
"""
function MatsubaraFunctions.save!(
    file  :: HDF5.File,
    label :: String,
    F     :: RefVertex
    )     :: Nothing

    grp = create_group(file, label)

    # Store scalar U as an HDF5 group attribute
    attributes(grp)["U"] = F.U

    # Save vertices
    MatsubaraFunctions.save!(file, label * "/Fp_p", F.Fp_p)
    MatsubaraFunctions.save!(file, label * "/Fp_x", F.Fp_x)
    MatsubaraFunctions.save!(file, label * "/Ft_p", F.Ft_p)
    MatsubaraFunctions.save!(file, label * "/Ft_x", F.Ft_x)

    return nothing
end

"""
    load_refvertex(file, label) -> RefVertex

Read a RefVertex written by `save!` from the HDF5 group `label`.
"""
function load_refvertex(
    file  :: HDF5.File,
    label :: String
    )     :: RefVertex

    U = read_attribute(file[label], "U")

    Fp_p = load_mesh_function(file, label * "/Fp_p")
    Fp_x = load_mesh_function(file, label * "/Fp_x")
    Ft_p = load_mesh_function(file, label * "/Ft_p")
    Ft_x = load_mesh_function(file, label * "/Ft_x")

    return RefVertex(U, Fp_p, Fp_x, Ft_p, Ft_x)
end
