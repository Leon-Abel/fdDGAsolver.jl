using MPI
MPI.Init()
using fdDGAsolver
using MatsubaraFunctions
using HDF5

# F↑↓ = 2u + u³·Σᵢ(iωᵢ)² / Πᵢ(iωᵢ) - 6u⁵ / Πᵢ(iωᵢ)
#       + βu²·[δ_{ω₁₂}·th + δ_{ω₁₃}·(th−1) + δ_{ω₁₄}·(th+1)] · Πᵢ(iωᵢ+u) / Πᵢ(iωᵢ)
#
# F↑↑ = βu²·(δ_{ω₁₄} − δ_{ω₁₂}) · Πᵢ(iωᵢ+u) / Πᵢ(iωᵢ)

# K1_r(ω_r) = F(ω_r, ν_r → ∞, ν'_r → ∞) - F0 
# F0↑↑ = 0
# F0↑↓ = U = 2u
function K1_r(T, U, ω, ::Type{Ch}) where {Ch <: fdDGAsolver.ChannelTag}
    β  = 1/T
    u  = U/2
    th = tanh(β * u / 2)

    if Ch === pCh
        K1_p_uu = 0
        K1_p_ud = β * u^2 * ((abs(ω) < π*T)*(th - 1))
        return K1_p_uu, K1_p_ud
    elseif Ch === tCh
        K1_t_uu = β * u^2 * (-(abs(ω) < π*T))
        K1_t_ud = β * u^2 * ((abs(ω) < π*T)*(th))
        return K1_t_uu, K1_t_ud
    elseif Ch === aCh
        K1_a_uu = β * u^2 * ((abs(ω) < π*T))
        K1_a_ud = β * u^2 * ((abs(ω) < π*T)*(th+1))
        return K1_a_uu, K1_a_ud
    else
        error("unknown channel $Ch")
    end 
end

# K2_r(ω_r, ν_r) = F(ω_r, ν_r, ν'_r → ∞) - Σ_r' K1_r'(ω_r) - F0 
# F0↑↑ = 0
# F0↑↓ = U = 2u
function K2_r(T, U, ω, ν, ::Type{Ch}) where {Ch <: fdDGAsolver.ChannelTag}
    β  = 1/T
    u  = U/2
    th = tanh(β * u / 2)

    if Ch === pCh
        K2_p_uu = 0
        # limit v' → ∞ of Πᵢ(iωᵢ+u) / Πᵢ(iωᵢ)
        P_K2_p = -((-im*ν + u)*(-im*ω + im*ν + u)) / (ν*ω - ν^2)
        reg  = 2*u^3 / (ν*ω - ν^2)
        anom = β * u^2 * ((abs(ω) < π*T)*(th - 1)) * P_K2_p
        K2_p_ud = reg + anom - K1_r(T, U, ω, pCh)[2]
        return K2_p_uu, K2_p_ud
    elseif Ch === tCh
        K2_t_uu = 0
        # limit v' → ∞ of Πᵢ(iωᵢ+u) / Πᵢ(iωᵢ)
        P_K2_t = ((im*ν + u)*(-im*ω - im*ν + u)) / (ν*ω + ν^2)
        reg  = -2*u^3 / (ν*ω + ν^2)
        anom = β * u^2 * ((abs(ω) < π*T)*(th)) * P_K2_t
        K2_t_ud = reg + anom - K1_r(T, U, ω, tCh)[2]
        return K2_t_uu, K2_t_ud
    elseif Ch === aCh
        K2_a_uu = 0
        # limit v' → ∞ of Πᵢ(iωᵢ+u) / Πᵢ(iωᵢ)
        P_K2_a = ((-im*ν + u)*(im*ω + im*ν + u)) / (ν*ω + ν^2)
        reg  = -2*u^3 / (ν*ω + ν^2)
        anom = β * u^2 * ((abs(ω) < π*T)*(th+1)) * P_K2_a
        K2_a_ud = reg + anom - K1_r(T, U, ω, aCh)[2]
        return K2_a_uu, K2_a_ud
    else
        error("unknown channel $Ch")
    end 
end

# # K2bar_r(ω_r, ν'_r) = F(ω_r, ν_r → ∞, ν'_r) - Σ_r' K1_r'(ω_r) - F0 
# # F0↑↑ = 0
# # F0↑↓ = U = 2u
# function K2bar_r(T, U, ω, νp, ::Type{Ch}) where {Ch <: fdDGAsolver.ChannelTag}
#     β  = 1/T
#     u  = U/2
#     th = tanh(β * u / 2)

#     if Ch === pCh
#         K2bar_p_uu = 0
#         # limit v' → ∞ of Πᵢ(iωᵢ+u) / Πᵢ(iωᵢ)
#         P_K2bar_p = ((im*νp + u)*(im*ω - im*νp + u)) / (νp*ω - νp^2)
#         reg  = 2*u^3 / (νp*ω - νp^2)
#         anom = β * u^2 * ((abs(ω) < π*T)*(th - 1)) * P_K2bar_p
#         K2bar_p_ud = reg + anom - K1_r(T, U, ω, pCh)[2]
#         return K2bar_p_uu, K2bar_p_ud
#     elseif Ch === tCh
#         K2bar_t_uu = 0
#         # limit v' → ∞ of Πᵢ(iωᵢ+u) / Πᵢ(iωᵢ)
#         P_K2bar_t = ((-im*νp + u)*(im*ω + im*νp + u)) / (νp*ω + νp^2)
#         reg  = -2*u^3 / (νp*ω + νp^2)
#         anom = β * u^2 * ((abs(ω) < π*T)*(th)) * P_K2bar_t
#         K2bar_t_ud = reg + anom - K1_r(T, U, ω, tCh)[2]
#         return K2bar_t_uu, K2bar_t_ud
#     elseif Ch === aCh
#         K2bar_a_uu = 0
#         # limit v' → ∞ of Πᵢ(iωᵢ+u) / Πᵢ(iωᵢ)
#         P_K2bar_a = ((im*νp + u)*(-im*ω - im*νp + u)) / (νp*ω + νp^2)
#         reg  = -2*u^3 / (νp*ω + νp^2)
#         anom = β * u^2 * ((abs(ω) < π*T)*(th+1)) * P_K2bar_a
#         K2bar_a_ud = reg + anom - K1_r(T, U, ω, aCh)[2]
#         return K2bar_a_uu, K2bar_a_ud
#     else
#         error("unknown channel $Ch")
#     end 
# end

function F_core(T, U, ω, ν, νp, ::Type{Ch}) where {Ch <: fdDGAsolver.ChannelTag}
    F_uu, F_ud         = full_vertex(T, U, ω, ν, νp, Ch)
    K1_uu, K1_ud       = K1_r(T, U, ω, Ch)
    K2_uu, K2_ud       = K2_r(T, U, ω, ν, Ch)
    K2bar_uu, K2bar_ud = K2bar_r(T, U, ω, νp, Ch)
    F_core_uu = F_uu - K1_uu - K2_uu - K2bar_uu
    F_core_ud = F_ud - K1_ud - K2_ud - K2bar_ud

    return F_core_uu, F_core_ud
end

"""
    F_HA(T, U, ω1, ω2, ω3, ω4; σ_eq_σp)

    Returns the analytic value of the four-point vertex in second-order pertubation 
    theory of the Hubbard atom. Temperature T, interaction strength U, incoming 
    frequencies -ω2 = ν + ω and -ω4 = ν' and outgoing frequencies ω1 = ν and ω3 = ν' + ω.

    σ_eq_σp differentiates between equal spins σ, σ' (true) and opposite spins σ, σ' (false).
"""
function F_HA(
    T ::Float64,
    U ::Float64, 
    ω ::Float64, 
    ν ::Float64, 
    νp::Float64, 
    σ_eq_σp::Bool,
           ::Type{Ch}) where {Ch <: fdDGAsolver.ChannelTag}

    # parameters
    β  = 1 / T
    u  = U / 2
    th = tanh(β * u / 2)

    function F_legs(ω, ν, νp, ::Type{Ch}) where {Ch <: fdDGAsolver.ChannelTag}
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
        
    # frequencies
    legs = F_legs(ω, ν, νp, Ch)
    iω  = collect(legs .* im)

    ω12 = legs[1] + legs[2]
    ω13 = legs[1] + legs[3]
    ω14 = legs[1] + legs[4]

    prod_iω        = prod(iω)
    prod_iω_plus_u = prod(iω .+ u)

    # two Matsubara frequencies ω, ω' are at least 2πT apart if ω ≠ ω'
    # to avoid numerical inaccuracies when checking ω == ω' we instead 
    # check abs(ω-ω') < πT 

    if σ_eq_σp
        # F_{↑↑}: purely anomalous
        return β * u^2 * ((abs(ω14) < π*T) - (abs(ω12) < π*T)) * prod_iω_plus_u / prod_iω
    else
        # F_{↑↓}: regular + anomalous
        sum_sq = sum(iω.^2)
        reg    = 2u + (u^3 * sum_sq - 6u^5) / prod_iω
        anom = β * u^2 * ((abs(ω12) < π*T) *  th +
                          (abs(ω13) < π*T) * (th - 1) +
                          (abs(ω14) < π*T) * (th + 1)
                         ) * prod_iω_plus_u / prod_iω
        return reg + anom
    end
end

function Vertex_HubbardAtom(
    T   :: Float64,
    U   :: Float64,
    nK1 :: Int64,
    nK2 :: NTuple{2, Int64},
    nK3 :: NTuple{2, Int64},
        :: Type{Q} = ComplexF64,
    ) where {Q}
 
    # ---- allocate three Channel objects (γp, γt, γa), zero-initialised
    γp = fdDGAsolver.Channel(T, nK1, nK2, nK3, Q)
    γt = fdDGAsolver.Channel(T, nK1, nK2, nK3, Q)
    γa = fdDGAsolver.Channel(T, nK1, nK2, nK3, Q)
 
    for (γ, Ch) in ((γp, pCh), (γt, tCh), (γa, aCh))
 
        # --- K1: depends on ω only
        for iω in meshes(γ.K1, Val(1))
            ω = plain_value(iω)
            K1_uu, K1_ud = K1_r(T, U, ω, Ch)
            γ.K1[iω] = Q(K1_ud)
        end
 
        # --- K2: depends on (Ω, ν).
        for iω in meshes(γ.K2, Val(1)),
            iν in meshes(γ.K2, Val(2))
            ω = plain_value(iω)
            ν = plain_value(iν)
            K2_uu, K2_ud = K2_r(T, U, ω, ν, Ch)
            γ.K2[iω, iν] = Q(K2_ud)
        end
 
        # --- K3: zero, included in RefVertex

    end

    # Define a K12-only vertex, used to subtract the asymptotic contribution from the core.
    Γ_K12 = Vertex(RefVertex(T, U, eltype(γa)), γp, γt, γa)

    gΩ = MatsubaraMesh(T, nK3[1], Boson)  
    gν = MatsubaraMesh(T, nK3[2], Fermion)

    Fp_p = MeshFunction(gΩ, gν, gν; data_t = Q)
    Fp_x = MeshFunction(gΩ, gν, gν; data_t = Q)
    Ft_p = MeshFunction(gΩ, gν, gν; data_t = Q)
    Ft_x = MeshFunction(gΩ, gν, gν; data_t = Q)

    for ind in eachindex(Ft_p.data)
        ω, ν, νp = plain_value.(to_meshes(Ft_p, ind))
        ω_, ν_, νp_ = value.(to_meshes(Ft_p, ind))

        # p Channel
        Fp_p[ind] = F_HA(T, U, ω, ν, νp, false, pCh) - Γ_K12(ω_, ν_, νp_, pCh, pSp)
        Fp_x[ind] = (F_HA(T, U, ω, ν, νp, true, pCh) - F_HA(T, U, ω, ν, νp, false, pCh))  - Γ_K12(ω_, ν_, νp_, pCh, xSp)

        # t Channel
        Ft_p[ind] = F_HA(T, U, ω, ν, νp, false, tCh) - Γ_K12(ω_, ν_, νp_, tCh, pSp)
        Ft_x[ind] = (F_HA(T, U, ω, ν, νp, true, tCh) - F_HA(T, U, ω, ν, νp, false, tCh))  - Γ_K12(ω_, ν_, νp_, tCh, xSp)
    end


    Λ = RefVertex(U, Fp_p, Fp_x, Ft_p, Ft_x)
 
    return Vertex(Λ, γp, γt, γa)
end

# """
#     Analytic_Vertex_HubbardAtom(T, U, nK3) where {Q}

#     Analytic vertex of the Hubbard atom.
# """
# function analytic_vertex_HA(
#     T   :: Float64,
#     U   :: Float64,
#     nK3 :: NTuple{2, Int64},
#         :: Type{Q} = ComplexF64,
#     ) where {Q}

#     # allocate Matsubara Meshes
#     gΩ = MatsubaraMesh(T, nK3[1], Boson)    # bosonic mesh, (2*nK3[1]-1) meshpoints
#     gν = MatsubaraMesh(T, nK3[2], Fermion)  # fermionic mesh, (2*nK3[1]) meshpoints

#     # allocate Mesh Functions for ↑↑ (uu) and ↑↓ (ud) spin components
#     F_uu = MeshFunction(gΩ, gν, gν; data_t = Q)
#     F_ud = MeshFunction(gΩ, gν, gν; data_t = Q)

#     # fill vertex
#     for idx in eachindex(F_uu.data)
#         Ω, ν, νp = value.(value.(to_meshes(F_uu, idx)))
#         ω_t = (ν, -ν - Ω, νp + Ω, -νp)

#         F_uu[idx] = F_HA(T, U, ω_t...; σ_eq_σp = true)    # ↑↑
#         F_ud[idx] = F_HA(T, U, ω_t...; σ_eq_σp = false)   # ↑↓
#     end

#     # save data
#     output_dir = joinpath(@__DIR__, "data")
#     mkpath(output_dir)
#     filename_out = joinpath(output_dir, "HA_an_vertex_nK3$(nK3)_T$(T)_U$(U).h5")

#     h5open(filename_out, "w") do f
#         f["T"]        = T
#         f["U"]        = U
#         f["nK3"]      = collect(nK3)
#         f["F_uu_data"]     = F_uu.data
#         f["F_ud_data"]     = F_ud.data
#     end

#     return F_uu, F_ud
# end

# """
#     F_HA(T, U, ω1, ω2, ω3, ω4; σ_eq_σp)

#     Returns the analytic value of the four-point vertex in second-order pertubation 
#     theory of the Hubbard atom. Temperature T, interaction strength U, incoming 
#     frequencies -ω2 = ν + ω and -ω4 = ν' and outgoing frequencies ω1 = ν and ω3 = ν' + ω.

#     σ_eq_σp differentiates between equal spins σ, σ' (true) and opposite spins σ, σ' (true).
# """
# function F_HA(
#     T::Float64, 
#     U::Float64, 
#     ω1::Float64, 
#     ω2::Float64, 
#     ω3::Float64, 
#     ω4::Float64; 
#     σ_eq_σp::Bool)

#     # parameters
#     β  = 1 / T
#     u  = U / 2
#     th = tanh(β * u / 2)
        
#     # frequencies
#     iω  = [im*ω1, im*ω2, im*ω3, im*ω4]

#     ω12 = ω1 + ω2
#     ω13 = ω1 + ω3
#     ω14 = ω1 + ω4

#     prod_iω        = prod(iω)
#     prod_iω_plus_u = prod(iω .+ u)

#     # two Matsubara frequencies ω, ω' are at least 2πT apart if ω ≠ ω'
#     # to avoid numerical inaccuracies when checking ω == ω' we instead 
#     # check abs(ω-ω') < πT 

#     if σ_eq_σp
#         # F_{↑↑}: purely anomalous
#         return β * u^2 * ((abs(ω14) < π*T) - (abs(ω12) < π*T)) * prod_iω_plus_u / prod_iω
#     else
#         # F_{↑↓}: regular + anomalous
#         sum_sq = sum(iω.^2)
#         reg    = 2u + (u^3 * sum_sq - 6u^5) / prod_iω
#         anom = β * u^2 * ((abs(ω12) < π*T) *  th +
#                           (abs(ω13) < π*T) * (th - 1) +
#                           (abs(ω14) < π*T) * (th + 1)
#                          ) * prod_iω_plus_u / prod_iω
#         return reg + anom
#     end
# end
