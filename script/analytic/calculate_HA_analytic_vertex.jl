# Analytic Hubbard-atom four-point vertex and its asymptotic decomposition.
#
# Builds a channel-decomposed `Vertex` object from the analytic expression 
# for the Hubbard-atom two-particle vertex at half-filling.
#
# Hubbard-atom vertex (u = U/2, β = 1/T, th = tanh(βu/2)):
#
#   F^↑↓ = 2u + (u³ Σᵢ(iνᵢ)² − 6u⁵) / Πᵢ(iνᵢ)
#         + βu² · [δ_{ω₁₂}·th + δ_{ω₁₃}·(th−1) + δ_{ω₁₄}·(th+1)]· Πᵢ(iνᵢ+u) / Πᵢ(iνᵢ)
#
#   F^↑↑ = βu² · (δ_{ω₁₄} − δ_{ω₁₂}) · Πᵢ(iνᵢ+u) / Πᵢ(iνᵢ)
#
# where ω_{ij} ≡ ν_i + ν_j. The Kronecker deltas on Matsubara frequencies are
# implemented numerically via `|ω| < πT`, since distinct Matsubaras differ by
# at least 2πT.
#
# Channel-leg conventions:
#
#   pCh : (ν₁,ν₂,ν₃,ν₄) = (ν', −ω+ν, ω−ν',   −ν )
#   tCh : (ν₁,ν₂,ν₃,ν₄) = (ν,  −ω−ν, ω+ν',   −ν')
#   aCh : (ν₁,ν₂,ν₃,ν₄) = (ν',   −ν, ω+ν,  −ω−ν')

using MPI
MPI.Init()
using fdDGAsolver
using MatsubaraFunctions
using HDF5

# -----------------------------------------------------------------------------
# Channel → four-leg frequency map (used by `full_vertex`).
# -----------------------------------------------------------------------------
 
"""
    F_legs(ω, ν, νp, ::Type{Ch}) :: NTuple{4, Float64}
 
Map the channel-natural arguments `(ω, ν, νp)` to the four 
fermionic leg frequencies `(ν₁, ν₂, ν₃, ν₄)`.
 
  `pCh : (ν', −ω+ν, ω−ν',   −ν )`

  `tCh : (ν,  −ω−ν, ω+ν',   −ν')`

  `aCh : (ν',   −ν, ω+ν,  −ω−ν')`
 
"""
@inline function F_legs(
    ω  :: Float64, 
    ν  :: Float64, 
    νp :: Float64, 
       ::Type{Ch}) :: NTuple{4, Float64} where {Ch <: fdDGAsolver.ChannelTag}
       
    if Ch === pCh
        return (νp, ν - ω, ω - νp, -ν)
    elseif Ch === tCh
        return (ν, -ν - ω, ω + νp, -νp)
    elseif Ch === aCh
        return (νp, -ν, ν + ω, -νp - ω)
    else
        error("unknown channel $Ch")
    end
end

# -----------------------------------------------------------------------------
# Full Hubbard-atom vertex at arbitrary (ω, ν, ν') in a given channel.
# -----------------------------------------------------------------------------
 
"""
    full_vertex(T, U, ω, ν, νp, ::Type{Ch}) :: NTuple{2, Float64}
 
Analytic Hubbard-atom four-point vertex at temperature `T` and on-site
interaction `U`, evaluated at channel bosonic frequency `ω` and channel
fermionic frequencies `ν, νp`, in channel `Ch ∈ {pCh, tCh, aCh}`.
 
Returns the tuple `(F_uu, F_ud)`:
  • `F_uu` — parallel-spin component F↑↑
  • `F_ud` — crossed-spin component F↑↓
 
Internally maps `(ω, ν, νp)` to the four fermionic leg frequencies
`(ν₁, ν₂, ν₃, ν₄)` via `F_legs`, then evaluates the closed-form expression
(see file header).
"""
function full_vertex(
    T  :: Float64, 
    U  :: Float64, 
    ω  :: Float64, 
    ν  :: Float64, 
    νp :: Float64, 
       ::Type{Ch}) :: NTuple{2, ComplexF64} where {Ch <: fdDGAsolver.ChannelTag}

    β  = 1 / T
    u  = U / 2
    th = tanh(β * u / 2)

    # Build the four-leg frequencies and the recurring products
    legs = F_legs(ω, ν, νp, Ch)
    iω  = collect(legs .* im)

    prod_iω        = prod(iω)                  # Πᵢ(iνᵢ)
    prod_iω_plus_u = prod(iω .+ u)             # Πᵢ(iνᵢ + u)
    an             = prod_iω_plus_u / prod_iω  # recurring anomalous prefactor
 
    # Kronecker δ on Matsubaras: distinct frequencies differ by ≥ 2πT, so
    # |ω_{ij}| < πT is a safe floating-point check for ω_{ij} = 0.
    δ12 = abs(legs[1] + legs[2]) < π*T
    δ13 = abs(legs[1] + legs[3]) < π*T
    δ14 = abs(legs[1] + legs[4]) < π*T
 
    # F↑↑: purely anomalous
    F_uu = β * u^2 * (δ14 - δ12) * an
 
    # F↑↓: regular + anomalous
    sum_sq = sum(iω .^ 2)
    reg    = 2u + (u^3 * sum_sq - 6u^5) / prod_iω
    anom   = β * u^2 * (δ12 * th + δ13 * (th - 1) + δ14 * (th + 1)) * an
    F_ud   = reg + anom

    return F_uu, F_ud
end


# -----------------------------------------------------------------------------
# K₁ asymptote
# -----------------------------------------------------------------------------
 
# K₁^Sp_r(ω) = lim_{ν_r, ν_r' → ∞} F^Sp(ω_r, ν_r, ν'_r) − F₀^Sp
# with bare F₀^↑↑ = 0 and F₀^↑↓ = U = 2u.
#   
# Channel-by-channel: at ν,ν' → ∞ only the single ω_{ij} that depends on
# neither fermionic frequency can be on the Kronecker line. From the leg
# table in the file header:
#
#   pCh: ω₁₃ =  ω   → anomalous coefficients: 0   (↑↑), (th−1) (↑↓)
#   tCh: ω₁₂ = −ω   → anomalous coefficients: −1  (↑↑),   th   (↑↓)
#   aCh: ω₁₄ = −ω   → anomalous coefficients: +1  (↑↑), (th+1) (↑↓)

"""
    K1_r(T, U, ω, ::Type{Ch}) :: NTuple{2, Float64}
 
Analytic K₁ asymptote of the Hubbard-atom four-point vertex in channel `Ch`.
Returns `(K1_uu, K1_ud)` — parallel-spin (↑↑) and antiparallel-spin (↑↓)
components — as functions of the channel bosonic frequency `ω` only.
 
K₁ is defined as  `K₁^Sp_r(ω) = lim_{ν_r, ν_r' → ∞} F^Sp(ω_r, ν_r, ν'_r) − F₀^Sp`, 
with bare `F₀^↑↑ = 0` and `F₀^↑↓ = U = 2u`.
"""
function K1_r(
    T :: Float64,
    U :: Float64,
    ω :: Float64,
      :: Type{Ch}
    ) :: NTuple{2, Float64} where {Ch <: fdDGAsolver.ChannelTag}

    β  = 1/T
    u  = U/2
    th = tanh(β * u / 2)

    # Kronecker-δ on Matsubara frequencies: two distinct frequencies differ
    # by ≥ 2πT, so |ω| < πT is a safe floating-point check for ω = 0.
    # In all three channels the surviving delta is ±ω, so the same δω applies.
    δω    = abs(ω) < π*T
    pref  = β * u^2 * δω        # common prefactor βu²·δ_ω
 
    if Ch === pCh
        # Surviving delta ω₁₃ = ω.  Coeffs: 0 (↑↑), (th − 1) (↑↓).
        K1_uu = 0.0
        K1_ud = pref * (th - 1)
 
    elseif Ch === tCh
        # Surviving delta ω₁₂ = −ω.  Coeffs: −1 (↑↑), th (↑↓).
        K1_uu = -pref
        K1_ud =  pref * th
 
    elseif Ch === aCh
        # Surviving delta ω₁₄ = −ω.  Coeffs: +1 (↑↑), (th + 1) (↑↓).
        K1_uu = pref
        K1_ud = pref * (th + 1)
 
    else
        error("unknown channel $Ch")
    end

    return K1_uu, K1_ud
end

# -----------------------------------------------------------------------------
# K₂ asymptote
# -----------------------------------------------------------------------------
 
# K₂^Sp_r(ω_r, ν_r) = lim_{ν_r' → ∞} F^Sp(ω_r, ν_r, ν'_r) − K₁^Sp_r(ω) − F₀.
#
# In the ν' → ∞ limit, two contributions remain:
#
#   (i)  "regular": the rational piece 2u + (u³ Σ(iν)² − 6u⁵)/Πᵢ(iνᵢ) has a finite
#        ν' → ∞ limit because the leading −2ν'² in the numerator cancels the
#        ν'² in the denominator.  Subtracting F₀ = 2u leaves a finite rational
#        function of (ω, ν), denoted `reg` below.
#
#   (ii) "anomalous": the same Kronecker-δ that survives the K₁ limit
#        (ω₁₃ for pCh, ω₁₂ for tCh, ω₁₄ for aCh) is still present, but now
#        multiplied by the ν' → ∞ limit of the prefactor Πᵢ(iνᵢ+u)/Πᵢ(iνᵢ),
#        denoted `P_K2_Ch` below.

"""
    K2_r(T, U, ω, ν, ::Type{Ch}) :: Tuple{ComplexF64, ComplexF64}
 
Analytic K₂ asymptote of the Hubbard-atom four-point vertex in channel `Ch`.
Returns `(K2_uu, K2_ud)` — parallel-spin (↑↑) and antiparallel-spin (↑↓)
components — as functions of the channel bosonic frequency `ω` and one
channel fermionic frequency `ν`.
 
K₂ is defined as `K₂_r^Sp(ω_r, ν_r) = lim_{ν'_r → ∞} F^Sp(ω_r, ν_r, ν'_r) − K₁_r^Sp(ω_r) − F₀`,
with bare `F₀^↑↑ = 0` and `F₀^↑↓ = U = 2u`
"""
function K2_r(
    T :: Float64,
    U :: Float64,
    ω :: Float64,
    ν :: Float64,
      :: Type{Ch}) :: NTuple{2, Float64} where {Ch <: fdDGAsolver.ChannelTag}

    β  = 1/T
    u  = U/2
    th = tanh(β * u / 2)

    # Kronecker-δ on Matsubara frequencies: two distinct frequencies differ
    # by ≥ 2πT, so |ω| < πT is a safe floating-point check for ω = 0.
    # In all three channels the surviving delta is ±ω, so the same δω applies.
    δω    = abs(ω) < π*T
    pref  = β * u^2 * δω        # common prefactor βu²·δ_ω

    if Ch === pCh
        # p-channel legs (ν', ν−ω, ω−ν', −ν); surviving delta is ω₁₃ = ω.
        # ν'→∞ limit of Πᵢ(iνᵢ+u)/Πᵢ(iνᵢ):
        P     = -((-im*ν + u) * (-im*ω + im*ν + u)) / (ν*ω - ν^2)
        # Regular remnant after F₀ = 2u cancels:
        reg   =  2u^3 / (ν*ω - ν^2)
        # Anomalous part: K₁ coefficient (th − 1) times P, then strip K₁.
        anom  = pref * (th - 1) * P

        K2_uu = 0
        K2_ud = reg + anom - K1_r(T, U, ω, pCh)[2]
 
    elseif Ch === tCh
        # t-channel legs (ν, −ν−ω, ω+ν', −ν'); surviving delta is ω₁₂ = −ω.
        P     = ((im*ν + u) * (-im*ω - im*ν + u)) / (ν*ω + ν^2)
        reg   = -2u^3 / (ν*ω + ν^2)
        anom  = pref * th * P

        K2_uu = -pref * P - K1_r(T, U, ω, tCh)[1]
        K2_ud = reg + anom - K1_r(T, U, ω, tCh)[2]
 
    elseif Ch === aCh
        # a-channel legs (ν', −ν, ω+ν, −ω−ν'); surviving delta is ω₁₄ = −ω.
        P     = ((u - im*ν) * (u + im*(ν + ω))) / (ν*ω + ν^2)
        reg   = -2u^3 / (ν*ω + ν^2)
        anom  = pref * (th + 1) * P

        K2_uu = pref * P - K1_r(T, U, ω, tCh)[1]
        K2_ud = reg + anom - K1_r(T, U, ω, aCh)[2]
 
    else
        error("unknown channel $Ch")
    end
 
    return K2_uu, K2_ud
end

# -----------------------------------------------------------------------------
# Assembly of the channel-decomposed `Vertex` object for the Hubbard atom.
# -----------------------------------------------------------------------------
 
"""
    Vertex_HubbardAtom(T, U, nK1, nK2, nK3, ::Type{Q} = ComplexF64) :: Vertex{ComplexF64, RefVertex{ComplexF64}}
 
Build a `Vertex(Λ, γp, γt, γa)` representing the analytic Hubbard-atom
two-particle vertex on Matsubara grids of size `nK1` (K₁ bosonic), `nK2`
(K₂ bosonic & fermionic), and `nK3` (K₃ bosonic & fermionic).
"""
function Vertex_HubbardAtom(
    T   :: Float64,
    U   :: Float64,
    nK1 :: Int64,
    nK2 :: NTuple{2, Int64},
    nK3 :: NTuple{2, Int64},
        :: Type{Q} = ComplexF64,
    ) :: Vertex{ComplexF64, RefVertex{ComplexF64}} where {Q}
 
    # ---- Step 1: build K1, K2 ---- 

    # allocate three Channel objects (γp, γt, γa), zero-initialised
    γp = fdDGAsolver.Channel(T, nK1, nK2, nK3, Q)
    γt = fdDGAsolver.Channel(T, nK1, nK2, nK3, Q)
    γa = fdDGAsolver.Channel(T, nK1, nK2, nK3, Q)
 
    # fill K1 and K2 of each channel with analytic data.
    # The channels only store the pSp component F^= = F^↑↓ 
    for (γ, Ch) in ((γp, pCh), (γt, tCh), (γa, aCh))
 
        # K1: depends on the channel bosonic frequency ω only.
        for iω in meshes(γ.K1, Val(1))
            ω = plain_value(iω)
            K1_uu, K1_ud = K1_r(T, U, ω, Ch)
            γ.K1[iω] = Q(K1_ud)
        end
 
        # K2: depends on (Ω, ν).
        # NOTE: by ν ↔ ν' symmetry of F in every channel, K₂̄ ≡ K₂ as stored functions;
        # the channel evaluator (see `channel.jl`) sums γ.K2[Ω,ν] + γ.K2[Ω,ν'] 
        # which correctly accounts for both legs.
        for iω in meshes(γ.K2, Val(1)),
            iν in meshes(γ.K2, Val(2))
            ω = plain_value(iω)
            ν = plain_value(iν)
            K2_uu, K2_ud = K2_r(T, U, ω, ν, Ch)
            γ.K2[iω, iν] = Q(K2_ud)
        end
 
        # K3 of each channel is left at zero; the K3 part of F (the "core")
        # is stored in the RefVertex `Λ` instead, see Step 4.
    end

    # ---- Step 2: build a K12-only vertex ----

    # Subtracts the K12 asymptote from F when extracting the core.
    # Its F0 is a *zero-K3* RefVertex carrying only the bare interaction `U`.
    Γ_K12 = Vertex(RefVertex(T, U, eltype(γa)), γp, γt, γa)

    # ---- Step 3: build vertex core ----
    
    # allocate the K3-shaped core MeshFunctions
    gΩ = MatsubaraMesh(T, nK3[1], Boson)  
    gν = MatsubaraMesh(T, nK3[2], Fermion)

    Fp_p = MeshFunction(gΩ, gν, gν; data_t = Q)
    Fp_x = MeshFunction(gΩ, gν, gν; data_t = Q)
    Ft_p = MeshFunction(gΩ, gν, gν; data_t = Q)
    Ft_x = MeshFunction(gΩ, gν, gν; data_t = Q)

    # fill them with (F_HA − Γ_K12) for each (channel, spin) combination 
    # that RefVertex stores explicitly: pCh/pSp, pCh/xSp, tCh/pSp, tCh/xSp.
    for ind in eachindex(Ft_p.data)
        # `plain_value` returns the bare Float64 frequency value;
        # `value` returns the MatsubaraFrequency object (needed by the
        # Vertex evaluator and `convert_frequency`).
        ω, ν, νp = plain_value.(to_meshes(Ft_p, ind))
        ω_, ν_, νp_ = value.(to_meshes(Ft_p, ind))

        # p Channel
        Fp_uu, Fp_ud = full_vertex(T, U, ω, ν, νp, pCh)
        Fp_p[ind] = Fp_ud - Γ_K12(ω_, ν_, νp_, pCh, pSp)
        Fp_x[ind] = (Fp_uu - Fp_ud)  - Γ_K12(ω_, ν_, νp_, pCh, xSp)

        # t Channel
        Ft_uu, Ft_ud = full_vertex(T, U, ω, ν, νp, tCh)
        Ft_p[ind] = Ft_ud - Γ_K12(ω_, ν_, νp_, tCh, pSp)
        Ft_x[ind] = (Ft_uu - Ft_ud)  - Γ_K12(ω_, ν_, νp_, tCh, xSp)
    end

    # ---- Step 4: build RefVertex ----
    # The RefVertex evaluator re-adds the bare ±U appropriately, so that the
    # final Vertex(Λ, γp, γt, γa) reproduces F_HA exactly on its grid.
    Λ = RefVertex(U, Fp_p, Fp_x, Ft_p, Ft_x)
 
    # return vertex composed of core RefVertex and decomposed channels
    return Vertex(Λ, γp, γt, γa)
end