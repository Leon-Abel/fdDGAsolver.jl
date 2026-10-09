"""
Calculate the charge and magnetic susceptibility

  χ_D(T), χ_M(T)  vs  T

for the Anderson impurity model (AIM) at particle-hole symmetry,
  U = 5.75,  V = 2,  D = 10

using two methods implemented in this package, plus the Hubbard atom itself as
an analytic comparison:
  1. PA         : self-consistent parquet approximation (scPA)
  2. fd-DΓA(HA) : finite-difference parquet DΓA with the Hubbard atom
                  as the reference (seed) vertex
  3. HA         : the analytic Hubbard-atom vertex

Usage (from the repo root, multi-threaded):
  julia --project=script -t <nthreads> script/run/calculate_susceptibilities.jl [options]

Run with `--help` for the full option list and the current defaults. The three
methods are on by default; name `--no-<method>` to leave one out.

  --nmax           half-box size for the vertex grids
  --T-lo --T-hi    temperature range limits
  --nT             number of temperature points
  --no-sc          skip the self-consistent parquet approximation
  --no-fd          skip the finite-difference DΓA calculation
  --no-HA          skip the analytic Hubbard-atom calculation

  julia --project=script -t 8 script/run/calculate_susceptibilities.jl --nmax=48 --nT=41
  julia --project=script -t 8 script/run/calculate_susceptibilities.jl --no-HA --no-sc
"""

using MPI
MPI.Init()

using fdDGAsolver
using MatsubaraFunctions
using HDF5
using Printf
@isdefined(ScriptCommon) || include(joinpath(@__DIR__, "..", "common", "ScriptCommon.jl"))
using .ScriptCommon

include(joinpath(@__DIR__, "..", "analytic", "calculate_HA_analytic_vertex.jl"))

# -----------------------------------------------------------------------
# Command-line arguments
# -----------------------------------------------------------------------
# The three method toggles are opt-OUT flags, so the common case needs no
# arguments at all.
# ---------------------------------------------------------------------------
# What executing this file in the REPL does. Edit these and execute it again.
#
# Running the file as a script overrides them from the command line - and these
# are the defaults `--help` reports, so the two entry points cannot drift apart.
# ---------------------------------------------------------------------------
nmax  = 48     # half-box size for the vertex grids
T_lo  = 1.0    # lower temperature limit
T_hi  = 2.0    # upper temperature limit
nT    = 41     # number of temperature points
no_sc = false  # skip the self-consistent parquet approximation (scPA)
no_fd = false  # skip the finite-difference DΓA calculation
no_HA = false  # skip the analytic Hubbard-atom calculation

# ---------------------------------------------------------------------------
# CLI driver (skipped when this file is executed in the REPL or `include`d)
# ---------------------------------------------------------------------------
if abspath(PROGRAM_FILE) == @__FILE__
    settings = cli_settings(
        "Calculate χ_D(T) and χ_M(T) for the particle-hole-symmetric Anderson impurity model.")
    add_arg_table!(settings,
        ["--nmax"], Dict(:help => "half-box size for the vertex grids",
                         :arg_type => Int, :default => nmax, :range_tester => >(0)),
        ["--T_lo"], Dict(:help => "lower temperature limit",
                         :arg_type => Float64, :default => T_lo),
        ["--T_hi"], Dict(:help => "upper temperature limit",
                         :arg_type => Float64, :default => T_hi),
        ["--nT"],   Dict(:help => "number of temperature points",
                         :arg_type => Int, :default => nT, :range_tester => >(0)))
    add_flags!(settings,
        "no-sc" => "skip the self-consistent parquet approximation (scPA)",
        "no-fd" => "skip the finite-difference DΓA calculation",
        "no-HA" => "skip the analytic Hubbard-atom calculation")

    parsed = parse_args(ARGS, settings)

    nmax  = parsed["nmax"]
    T_lo  = parsed["T_lo"]
    T_hi  = parsed["T_hi"]
    nT    = parsed["nT"]
    no_sc = parsed["no_sc"]
    no_fd = parsed["no_fd"]
    no_HA = parsed["no_HA"]
end

sc   = !no_sc
fd   = !no_fd
HA   = !no_HA

T_lo <= T_hi || error("T_lo must be <= T_hi, got $T_lo > $T_hi.")
(sc || fd || HA) || error("Nothing to do: all three methods were skipped.")

# -----------------------------------------------------------------------
# Physical parameters
# -----------------------------------------------------------------------

# On-site Coulomb repulsion on the impurity
U = 5.75

# Hybridization
# We use hopping amplitude V = 2 and half-bandwidth D = 10.
# For a box-shaped bath: Δ = π V² / (2D) = π/5.
Δ = π / 5.0
D = 10.0

# Impurity on-site energy ε_d plus Hartree contribution of the self-energy.
# e = 0 corresponds to particle-hole symmetry (half filling of the impurity).
e = 0.0

# Temperature array 
T_values = collect(range(T_lo, T_hi, length = nT))

# -----------------------------------------------------------------------
# Frequency grid sizes (same for all temperatures)
# -----------------------------------------------------------------------

# nG  : half-box for the Green function and self-energy Matsubara mesh.
nG  = 4nmax
# nK1 : bosonic box for K1(iΩ) and for the bubble Π(iΩ, iν).
nK1 = 4nmax
# nK2 : (bosonic, fermionic) box for K2(iΩ, iν).
nK2 = (2nmax, 2nmax)
# nK3 : (bosonic, fermionic) box for K3(iΩ, iν, iν').
nK3 = (nmax, nmax)

# -----------------------------------------------------------------------
# Output storage
# -----------------------------------------------------------------------
if sc
    chiD_sc = zeros(Float64, nT)
    chiM_sc = zeros(Float64, nT)
end
if fd
    chiD_fd = zeros(Float64, nT)
    chiM_fd = zeros(Float64, nT)
end
if HA
    chiD_HA = zeros(Float64, nT)
    chiM_HA = zeros(Float64, nT)
end

# -----------------------------------------------------------------------
# Helper: extract χ_D and χ_M from a converged solver
#
# Storage convention:
#   t-channel pSp  =  - (K1_D − K1_M) / 2
#   a-channel pSp  =  K1_M
#
# Inverting:
#   K1_M  = K1a_pSp                              (magnetic K1)
#   K1_D  =  -2 * K1t_pSp + K1a_pSp             (density K1)
#
# Physical susceptibilities (normalised):
#   χ_D = K1_D(Ω=0) / U²
#   χ_M = K1_M(Ω=0) / U²
#
# The K1 mesh functions store complex values; for the PHS case they are
# real-valued (imaginary part is zero by symmetry).
# -----------------------------------------------------------------------
function extract_chi(S, T, U)
    Ω0 = MatsubaraFrequency(T, 0, Boson) # bosonic Matsubara frequency Ω = 0

    # Reducible-vertex K1 at Ω = 0, in the pSp (parallel-spin) convention.
    # For an scPA result the total channel K1 equals the stored channel K1 directly.
    K1t = real(S.F.γt.K1[Ω0])   # t-channel pSp
    K1a = real(S.F.γa.K1[Ω0])   # a-channel pSp

    # Invert the spin-channel relations
    K1_M = K1a                  # magnetic
    K1_D = -2K1t + K1a          # density

    # Normalise by U² to get the physical susceptibility
    return K1_D / U^2, K1_M / U^2
end

# Same helper for the fdPA result (F = δF relative to F0).
# The TOTAL channel K1 is the stored delta plus the reference system's K1:
#   K1_total = S_fd.F.γt.K1 + S_fd.F0.γt.K1
# Here S_fd.F0 is the full Vertex of the Hubbard atom reference.
function extract_chi_fd(S_fd, T, U)
    Ω0 = MatsubaraFrequency(T, 0, Boson)

    # δK1 (increment above reference) + reference K1 = total K1
    K1t_delta = real(S_fd.F.γt.K1[Ω0])
    K1a_delta = real(S_fd.F.γa.K1[Ω0])

    K1t_ref   = real(S_fd.F0.γt.K1[Ω0])
    K1a_ref   = real(S_fd.F0.γa.K1[Ω0])

    # Total K1 in each channel
    K1t = K1t_delta + K1t_ref
    K1a = K1a_delta + K1a_ref

    K1_M = K1a
    K1_D = -2K1t + K1a

    return K1_D / U^2, K1_M / U^2
end

# Same helper for the analytic HA result.
function extract_chi_HA(F_HA, T, U)
    Ω0 = MatsubaraFrequency(T, 0, Boson)

    K1t = real(F_HA.γt.K1[Ω0])
    K1a = real(F_HA.γa.K1[Ω0])

    K1_M = K1a
    K1_D = -2K1t + K1a

    return K1_D / U^2, K1_M / U^2
end

# -----------------------------------------------------------------------
# Main loop over temperatures
# -----------------------------------------------------------------------
for (iT, T) in enumerate(T_values)

    @printf "\n=== T = %.4f  (%d / %d) ===\n" T iT nT

    # Fermionic Matsubara mesh for G, Σ, and the bubble integration.
    # Contains 2*nG points: ν_n = 2π(n+½)T for n ∈ [−nG, nG−1].
    mG = MatsubaraMesh(T, nG, Fermion)

    # ------------------------------------------------------------------
    # 1) Parquet Approximation (scPA) for the SIAM
    # ------------------------------------------------------------------
    if sc 
        S_PA = parquet_solver_siam_parquet_approximation(nG, nK1, nK2, nK3;
            e, T, U, Δ, D)
        init_sym_grp!(S_PA)

        fdDGAsolver.solve!(S_PA;
            strategy      = :scPA,
            parallel_mode = :hybrid,
            verbose       = true,
            maxiter       = 200,
            tol           = 1e-8
        )
        chiD_sc[iT], chiM_sc[iT] = extract_chi(S_PA, T, U)
        @printf "  PA:      χ_D = %.6f  χ_M = %.6f\n" chiD_sc[iT] chiM_sc[iT]
    end

    # ------------------------------------------------------------------
    # 2) fd-DΓA(HA): finite-difference parquet DΓA with the Hubbard atom 
    #                as the reference system.
    # ------------------------------------------------------------------
    if fd
        # Build analytic Green function of Hubbard atom
        mG = MatsubaraMesh(T, nG, Fermion)  # fermionic mesh
        G_HA = MeshFunction(mG)             # Greens function defined on mesh
        for ν in mG
            ν_value = plain_value(ν)
            G_HA[ν] = 1/(ν_value + U^2/(4*ν_value))
        end

        # Build analytic self-energy of Hubbard atom
        Σ_HA = MeshFunction(mG) # Self-energy function defined on mesh
        for ν in mG
            ν_value = plain_value(ν)
            Σ_HA[ν] = 1/G_HA[ν] - ν_value
        end

        # Build analytic vertex
        F_HA = Vertex_HubbardAtom(T, U, nK1, nK2, nK3)

        # Build SIAM bare Green function
        Gbare_SIAM = siam_bare_Green(mG; e, Δ, D)

        # fdPA solver.
        S_fd = ParquetSolver(nK1, nK2, nK3, Gbare_SIAM, G_HA, Σ_HA, F_HA)
        init_sym_grp!(S_fd)

        fdDGAsolver.solve!(S_fd;
            strategy      = :fdPA,
            parallel_mode = :hybrid,
            verbose       = true,
            maxiter       = 200,
            tol           = 1e-8
        )
        chiD_fd[iT], chiM_fd[iT] =  extract_chi_fd(S_fd, T, U)
        @printf "  fd-DΓA(HA): χ_D = %.6f  χ_M = %.6f\n" chiD_fd[iT] chiM_fd[iT]
    end

    # ------------------------------------------------------------------
    # 3) HA: analytic solution of the Hubbard atom.
    # ------------------------------------------------------------------
    if HA
        F_HA = Vertex_HubbardAtom(T, U, nK1, nK2, nK3)
        chiD_HA[iT], chiM_HA[iT] =  extract_chi_HA(F_HA, T, U)
    end
end

# -----------------------------------------------------------------------
# Save results
# -----------------------------------------------------------------------
# `chi_scan_path` (common/paths.jl) is the one place this name is built, so the
# plot script that reads these files cannot spell it differently than they do.
if sc
    filename_out = chi_scan_path(:sc, nmax, T_lo, T_hi, nT)
    mkpath(dirname(filename_out))

    h5open(filename_out, "w") do f
        f["T_values"] = T_values
        f["chiD_sc"]  = chiD_sc
        f["chiM_sc"]  = chiM_sc
    end
    println("\n sc-Results written to $filename_out")
end
if fd
    filename_out = chi_scan_path(:fd, nmax, T_lo, T_hi, nT)
    mkpath(dirname(filename_out))

    h5open(filename_out, "w") do f
        f["T_values"] = T_values
        f["chiD_fd"]  = chiD_fd
        f["chiM_fd"]  = chiM_fd
    end
    println("\n fd-Results written to $filename_out")
end
if HA
    filename_out = chi_scan_path(:HA, nmax, T_lo, T_hi, nT)
    mkpath(dirname(filename_out))

    h5open(filename_out, "w") do f
        f["T_values"] = T_values
        f["chiD_HA"]  = chiD_HA
        f["chiM_HA"]  = chiM_HA
    end
    println("\n HA-Results written to $filename_out")
end