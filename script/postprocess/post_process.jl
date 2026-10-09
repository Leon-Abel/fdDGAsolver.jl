"""
    chi_D_from_K1(K1a_val, K1t_val, U) -> Float64

Physical density susceptibility from *already-evaluated* reducible K1
values (paper Eq. S19a, including the factor 1/2 that matches the paper convention):

    χ_D = (-2 K1t + K1a) / U² / 2.
"""
chi_D_from_K1(K1a_val, K1t_val, U) = (-2real(K1t_val) + real(K1a_val)) / U^2 / 2

"""
    chi_M_from_K1(K1a_val, U)          -> Float64

Physical magnetic susceptibility from *already-evaluated* reducible K1
values (paper Eq. S19a, including the factor 1/2 that matches the paper convention):

    χ_M = K1a / U² / 2.
"""
chi_M_from_K1(K1a_val, U) = real(K1a_val) / U^2 / 2

"""
    chi_K1(S, U; Ω) -> (χD_fine, χM_fine)

Physical density / magnetic susceptibility read directly from the converged
reducible vertex K1, through [`chi_M_from_K1`](@ref) / [`chi_D_from_K1`](@ref) —
so including the factor 1/2 those apply (paper Eq. S19a):

    χ_M(q) = K1a(q) / U² / 2,
    χ_D(q) = (-2 K1t(q) + K1a(q)) / U² / 2.

It is evaluated on the coarse vertex mesh `meshes(S.F.γa.K1, Val(2))` and
Fourier-interpolated onto the fine G k-mesh `meshes(S.G, Val(2))`, so it can be
indexed along the dense Γ-X-M-Γ path.
"""
function chi_K1(S, U :: Float64; Ω :: MatsubaraFrequency{Boson})
    mK_Γ = meshes(S.F.γa.K1, Val(2))     # coarse vertex mesh
    mK_G = meshes(S.G, Val(2))           # fine G k-mesh

    K1a  = S.F.γa.K1
    K1t  = S.F.γt.K1

    # ComplexF64 needed for _fourier_interpolate!
    χM = ComplexF64[ chi_M_from_K1(K1a[Ω, q], U)            for q in mK_Γ ]
    χD = ComplexF64[ chi_D_from_K1(K1a[Ω, q], K1t[Ω, q], U) for q in mK_Γ ]

    Lo = Int(sqrt(length(mK_G)))
    Li = Int(sqrt(length(mK_Γ)))
    return real.(fdDGAsolver._fourier_interpolate!(χD, Lo, Li)),
           real.(fdDGAsolver._fourier_interpolate!(χM, Lo, Li))
end

"""
    chi_K1_all(S, U) -> (Ωs, χD, χM)

Batched version of [`chi_K1`](@ref): evaluate the physical density / magnetic
susceptibility on **every** bosonic frequency of the solver's K1 mesh, instead of
a single `Ω`. The per-`Ω` read and Fourier interpolation are identical to
`chi_K1`; only the loop over `Ω` is added.

Returns
- `Ωs :: Vector{Float64}`      — the K1 bosonic Matsubara values (full ± range),
                                  length `nΩ`, in mesh order.
- `χD, χM :: Matrix{Float64}`   — size `(nΩ, nk²)`; row `iΩ` holds
                                  `χ_{D,M}(Ωs[iΩ], q)` on the fine G k-mesh, laid
                                  out like the vectors returned by `chi_K1`.

Like `chi_K1`, this reads the *stored* reducible K1, so for the fd solver it
returns the tilde (difference-to-reference) susceptibility and for the sc solver
the raw one.
"""
function chi_K1_all(S, U :: Float64)
    mΩ   = meshes(S.F.γa.K1, Val(1))     # bosonic K1 mesh
    mK_Γ = meshes(S.F.γa.K1, Val(2))     # coarse vertex k-mesh
    mK_G = meshes(S.G, Val(2))           # fine G k-mesh

    K1a  = S.F.γa.K1
    K1t  = S.F.γt.K1

    Lo = Int(sqrt(length(mK_G)))
    Li = Int(sqrt(length(mK_Γ)))

    Ωs = Float64[ plain_value(Ω) for Ω in mΩ ]
    nΩ = length(Ωs)

    χD = Matrix{Float64}(undef, nΩ, Lo^2)
    χM = Matrix{Float64}(undef, nΩ, Lo^2)

    for (iΩ, Ω) in enumerate(mΩ)
        # ComplexF64 needed for _fourier_interpolate!
        χM_c = ComplexF64[ chi_M_from_K1(K1a[Ω, q], U)            for q in mK_Γ ]
        χD_c = ComplexF64[ chi_D_from_K1(K1a[Ω, q], K1t[Ω, q], U) for q in mK_Γ ]
        χD[iΩ, :] .= real.(fdDGAsolver._fourier_interpolate!(χD_c, Lo, Li))
        χM[iΩ, :] .= real.(fdDGAsolver._fourier_interpolate!(χM_c, Lo, Li))
    end

    return Ωs, χD, χM
end