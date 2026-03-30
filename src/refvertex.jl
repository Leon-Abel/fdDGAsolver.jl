# refvertex.jl
#
# Defines the `RefVertex` type, which represents the two-particle vertex in the
# parquet / multiloop fRG framework, decomposed into diagrammatic channels and
# spin components.

"""
    RefVertex{Q}
 
Container for the two-particle vertex used as a reference / input in the
parquet BSE or multiloop fRG equations.
"""
struct RefVertex{Q}
    U    :: Q           # bare interaction
    Fp_p :: MF_K3{Q}    # K3 vertex in the p-channel, parallel spin component
    Fp_x :: MF_K3{Q}    # K3 vertex in the p-channel, crossed spin component
    Ft_p :: MF_K3{Q}    # K3 vertex in the t-channel, parallel spin component
    Ft_x :: MF_K3{Q}    # K3 vertex in the t-channel, crossed spin component
    # The a-channel vertex is not stored independently; it is recovered via crossing symmetry 

    # Construct from pre-built K3 mesh functions.
    function RefVertex(
        U    :: Number,
        Fp_p :: MF_K3{Q},
        Fp_x :: MF_K3{Q},
        Ft_p :: MF_K3{Q},
        Ft_x :: MF_K3{Q},
        )    :: RefVertex{Q} where {Q}

        return new{Q}(Q(U), Fp_p, Fp_x, Ft_p, Ft_x)
    end

    # Construct a null (zero) RefVertex at temperature T with trivial (1-point) Matsubara meshes.
    function RefVertex(
        T :: Float64,               # temperature
        U :: Float64,               # bare interaction
          :: Type{Q} = ComplexF64,  # numeric type for vertex data (default ComplexF64)
        ) where {Q}

        # Null vertices: single-point meshes (no frequency dependence yet)
        gΩ = MatsubaraMesh(T, 1, Boson)     # bosonic mesh with 1 frequency point
        gν = MatsubaraMesh(T, 1, Fermion)   # fermionic mesh with 1 frequency point
        Fp = MeshFunction(gΩ, gν, gν; data_t = Q)
        Fx = MeshFunction(gΩ, gν, gν; data_t = Q)
        set!(Fp, 0) # initialise all entries to zero
        set!(Fx, 0)

        # p-channel and t-channel share the same mesh structure; copy ensures
        # independent storage
        return new{Q}(Q(U), Fp, Fx, copy(Fp), copy(Fx)) :: RefVertex{Q}
    end
end

#----------------------------------------------------------------------------------------------#
# Utility methods
#----------------------------------------------------------------------------------------------#

# Expose the scalar type Q
Base.eltype(::Type{<: RefVertex{Q}}) where {Q} = Q

# Pretty-print: show type, interaction U, temperature, and K3 grid size.
function Base.show(io::IO, Γ::RefVertex{Q}) where {Q}
    print(io, "$(nameof(typeof(Γ))){$Q}, U = $(Γ.U), T = $(temperature(Γ)), K3 : $(numK3(Γ))")
end

"""
    numK3(Λ :: RefVertex) -> NTuple{2, Int64}
 
Return the number of Matsubara frequency points in the bosonic (Ω) and
fermionic (ν) dimensions of the stored K3 vertex functions as a tuple (N_boson, N_fermion).
"""
function numK3(
    Λ :: RefVertex
    ) :: NTuple{2, Int64}

    # Both Fp_p and Fp_x share the same mesh, so querying Fp_p is sufficient.
    # Val(1) → bosonic mesh, Val(2) → first fermionic mesh
    return N(meshes(Λ.Fp_p, Val(1))), N(meshes(Λ.Fp_p, Val(2)))
end

"""
    set!(F :: RefVertex, val :: Number)
 
Fill all four K3 components of `F` with the constant `val` in place.
Useful for zeroing out the vertex before an iteration.
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

#----------------------------------------------------------------------------------------------#
# Getter methods
#----------------------------------------------------------------------------------------------#

"""
    temperature(F :: RefVertex) -> Float64
 
Extract the temperature from the bosonic Matsubara mesh of `Fp_p`.
All meshes are consistent by construction, so any component would give
the same result.
"""
function MatsubaraFunctions.temperature(
    F :: RefVertex
    ) :: Float64

    return MatsubaraFunctions.temperature(meshes(F.Fp_p, Val(1)))
end

#----------------------------------------------------------------------------------------------#
# Deep copy
#----------------------------------------------------------------------------------------------#

"""
    copy(F :: RefVertex{Q}) -> RefVertex{Q}
 
Return a deep copy of `F`, duplicating all four K3 data arrays so that the
original and the copy are independent in memory.
"""
function Base.:copy(
    F :: RefVertex{Q}
    ) :: RefVertex{Q} where {Q}

    return RefVertex(F.U, copy(F.Fp_p), copy(F.Fp_x), copy(F.Ft_p), copy(F.Ft_x))

end

#----------------------------------------------------------------------------------------------#
# Comparison
#----------------------------------------------------------------------------------------------#

"""
    ==(F1 :: RefVertex{Q}, F2 :: RefVertex{Q}) -> Bool
 
Two `RefVertex` objects are equal iff their bare interactions and all four K3
components are element-wise equal.
"""
function Base.:(==)(
    F1 :: RefVertex{Q},
    F2 :: RefVertex{Q},
    )  :: Bool where {Q}
    return (F1.U == F2.U) && (F1.Fp_p == F2.Fp_p) && (F1.Fp_x == F2.Fp_x) && (F1.Ft_p == F2.Ft_p) && (F1.Ft_x == F2.Ft_x)
end

#----------------------------------------------------------------------------------------------#
# Evaluators – parallel spin component (pSp)
#----------------------------------------------------------------------------------------------#
# The full vertex in each channel is the stored reducible part plus the bare
# vertex U (which acts as the asymptotic K1/K2 limit in the high-frequency regime).

# p-channel, parallel spin
@inline function (F :: RefVertex{Q})(
    Ω  :: MatsubaraFrequency,
    ν  :: MatsubaraFrequency,
    νp :: MatsubaraFrequency,
       :: Type{pCh},
       :: Type{pSp},
    ; kwargs...
    )  :: Q where {Q}

    return F.Fp_p(Ω, ν, νp) + F.U
end

## t-channel, parallel spin
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

# a-channel, parallel spin: obtained via crossing symmetry from the t-channel
# (note the swap ν ↔ ν′ and the overall minus sign from the crossing relation)
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

#----------------------------------------------------------------------------------------------#
# Evaluators – cossed spin component (xSp)
#----------------------------------------------------------------------------------------------#
# The crossed component carries an overall sign flip relative to pSp due to
# Fermi statistics (antisymmetry under exchange of the two outgoing legs).

# p-channel, crossed spin
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

# t-channel, crossed spin
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

# a-channel, crossed spin: obtained via crossing symmetry from the t-channel
# (note the swap ν ↔ ν′ and the overall minus sign from the crossing relation)
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

#----------------------------------------------------------------------------------------------#
# Evaluators – density component (dSp)
#----------------------------------------------------------------------------------------------#
# Defined as dSp = 2*pSp + xSp (see types.jl). Computed on the fly by
# combining the pSp and xSp evaluators above; no additional storage needed.

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

#----------------------------------------------------------------------------------------------#
# Evaluators – high-frequency / asymptotic limits
#----------------------------------------------------------------------------------------------#
# When any fermionic frequency is sent to ±∞ (represented by
# `InfiniteMatsubaraFrequency`), the vertex reduces to its bare value U
# (the K2/K1 asymptotic classes vanish).

# ν → ∞,  ν′ finite
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

# Both fermionic frequencies → ∞
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

#----------------------------------------------------------------------------------------------#
# Bare vertex helpers
#----------------------------------------------------------------------------------------------#
# The bare interaction U enters with a spin-dependent sign:
#   pSp (+U) : parallel spins see the full Hubbard repulsion
#   xSp (-U) : crossed component picks up a minus sign from anti-symmetry
#   dSp (+U) : density channel  2*(+U) + (-U) = +U  


@inline bare_vertex(F :: RefVertex) =  F.U                  # default: return U
@inline bare_vertex(F :: RefVertex, :: Type{pSp}) =  F.U    # parallel component: +U
@inline bare_vertex(F :: RefVertex, :: Type{xSp}) = -F.U    # crossed component:  -U
@inline bare_vertex(F :: RefVertex, :: Type{dSp}) =  F.U    # density: +U
@inline bare_vertex(F :: RefVertex, :: Type{Ch}, :: Type{Sp}) where {Ch <: ChannelTag, Sp <: SpinTag} = bare_vertex(F, Sp)   # TODO: remove

#----------------------------------------------------------------------------------------------#
# HDF5 I/O
#----------------------------------------------------------------------------------------------#

"""
    save!(file :: HDF5.File, label :: String, F :: RefVertex)
 
Write a `RefVertex` to an HDF5 file under the group `label`.
The bare interaction U is stored as an HDF5 attribute; the four K3 mesh
functions are stored as sub-groups via `MatsubaraFunctions.save!`.
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
 
Read a `RefVertex` previously written by `save!` from an HDF5 file.
Reconstructs U and all four K3 mesh functions from the group at `label`.
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