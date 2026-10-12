# =============================================================================
# The lattice vertex of the solver NL3, and averages of K2 and K3 over their
# fermionic momenta.
#
# As for NL_Vertex (nonlocal/vertex.jl), F = F0 + Φ_p + Φ_t + Φ_a (Lihm 2025,
# Eq. (1)), with the channels stored as NL3_Channel (all momenta kept). All
# evaluation uses the shared methods of nonlocal/vertex.jl: every channel is
# taken at its exactly converted frequencies and momenta.
#
# NL3_Vertex has no s-wave (kSW) methods of its own. With k or k′ = kSW and
# finite frequencies, the call goes to the νInf method of nonlocal/vertex.jl,
# which keeps only F0 and the channel Ch itself. The NL3 solver never does
# this; do not use such calls for NL3 vertices.
#
# Paper ↔ code: q ↔ P; k, k′ ↔ k, kp; Φ_r ↔ the fields `γp`, `γt`, `γa`.
# =============================================================================

"""
    NL3_Vertex{Q, VT} <: AbstractNonlocalVertex{Q}

Full lattice vertex of the solver NL3, `F = F0 + Φ_p + Φ_t + Φ_a`, with the
channels stored as [`NL3_Channel`](@ref) (all momenta kept).

- `F0` (type `VT`): the vertex the channels are added to: a `RefVertex` (bare U)
  in the PA, or the reference f in fd runs.
- `γp`, `γt`, `γa`: the channels `Φ_p`, `Φ_t`, `Φ_a` (`Φ̃_r` in fd runs).

Constructors: `NL3_Vertex(F0, γp, γt, γa)`, or
`NL3_Vertex(F0, T, numK1, numK2, numK3, meshK)` with three zero channels on the
momentum mesh `meshK`.
"""
struct NL3_Vertex{Q, VT} <: AbstractNonlocalVertex{Q}
    F0 :: VT
    γp :: NL3_Channel{Q}
    γt :: NL3_Channel{Q}
    γa :: NL3_Channel{Q}

    # from existing channels
    function NL3_Vertex(
        F0 :: VT,
        γp :: NL3_Channel{Q},
        γt :: NL3_Channel{Q},
        γa :: NL3_Channel{Q},
        )  :: NL3_Vertex{Q} where {Q, VT}

        return new{Q, VT}(F0, γp, γt, γa)
    end

    # three zero channels with the given boxes and momentum mesh
    function NL3_Vertex(
        F0    :: VT,
        T     :: Float64,
        numK1 :: Int64,
        numK2 :: NTuple{2, Int64},
        numK3 :: NTuple{2, Int64},
        meshK :: KMesh,
        ) where {VT}

        Q = eltype(F0)

        γ = NL3_Channel(T, numK1, numK2, numK3, meshK, Q)
        return new{Q, VT}(F0, γ, copy(γ), copy(γ)) :: NL3_Vertex{Q}
    end
end

# An NL3_Vertex stores its channels as NL3_Channel objects (used by load_vertex).
channel_type(::Type{NL3_Vertex}) = NL3_Channel

# Printing: type, U, temperature, F0 and the box sizes.
function Base.show(io::IO, Γ::NL3_Vertex{Q}) where {Q}
    print(io, "$(nameof(typeof(Γ))){$Q}, U = $(bare_vertex(Γ.F0)), T = $(temperature(Γ))\n")
    print(io, "F0 : $(Γ.F0)\n")
    print(io, "K1 : $(numK1(Γ))\n")
    print(io, "K2 : $(numK2(Γ))\n")
    print(io, "K3 : $(numK3(Γ))")
end

"""
    copy(F::NL3_Vertex) -> NL3_Vertex

Independent copy, including a copy of `F0`.
"""
function Base.:copy(
    F :: NL3_Vertex{Q}
    ) :: NL3_Vertex{Q} where {Q}

    return NL3_Vertex(copy(F.F0), copy(F.γp), copy(F.γt), copy(F.γa))
end

# -----------------------------------------------------------------------------
# Averages over the fermionic momenta
# -----------------------------------------------------------------------------

"""
    average_fermionic_momenta!(K2::NL2_MF_K2) -> nothing
    average_fermionic_momenta!(K3::NL3_MF_K3) -> nothing

Replace K2(Ω, ν; q, k) by its average over k, or K3(Ω, ν, ν′; q, k, k′) by its
average over k and k′, at every bosonic momentum q, in place. The arrays keep
their shape, so this turns an NL2/NL3 class into its s-wave form (Lihm 2025,
Eq. (S18)). Only called from commented-out lines in solve.jl.
"""
function average_fermionic_momenta!(K2 :: NL2_MF_K2)
    K2_avg = sum(K2.data, dims=4) ./ size(K2.data, 4)
    @views for i in axes(K2.data, 4)
        K2.data[:, :, :, i] .= K2_avg
    end
    return nothing
end


# K3 method of average_fermionic_momenta!, see the docstring above.
function average_fermionic_momenta!(K3 :: NL3_MF_K3)
    K3_avg = sum(K3.data, dims=(5, 6)) ./ size(K3.data, 5) ./ size(K3.data, 6)
    @views for j in axes(K3.data, 6), i in axes(K3.data, 5)
        K3.data[:, :, :, :, i, j] .= K3_avg
    end
    return nothing
end
