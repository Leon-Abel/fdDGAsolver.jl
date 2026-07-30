"""
Solve the 2D Hubbard model with the finite-difference parquet solver and save the
converged solver objects for post-processing.

Two methods are run (either can be switched off):
  PA   (sc) : parquet approximation — no DMFT input, bare vertex as reference
  pDΓA (fd) : finite-difference parquet DΓA — DMFT vertex as the reference

The DMFT reference (temperature T, chemical potential, hopping and local vertex) is
read from `filename_dmft_partial`; U and T are taken from it.

Usage (from the repo root, multi-threaded):
  julia -t <nthreads> script/run_hubbard_model.jl [nmax] [nq] [NL_method] [dmft_file] [do_PA] [do_fd]

  nmax      : half-box size for the vertex grids                     (default 14)
  nq        : linear k-grid size for the vertex mesh                 (default 8)
  NL_method : vertex momentum structure — 1 (K2 ≈ K2(q), K3 ≈ K3(q)),
              2 (K3 ≈ K3(q)), or 3 (full q dependence)               (default 1)
  dmft_file : DMFT reference data, relative to the repo root  (default data/high_temperature_U0.5.h5)
  do_PA     : run the parquet approximation                          (default true)
  do_fd     : run the finite-difference parquet DΓA                  (default true)

Output — one HDF5 file per method under `output/hubbard_model/`:
  hubbard_sc_U<U>_T<T>_nmax<nmax>_nq<nq>_NL<NL_method>.h5   (PA)
  hubbard_fd_U<U>_T<T>_nmax<nmax>_nq<nq>_NL<NL_method>.h5   (pDΓA)

Each file holds the full solver state written by `save!` (Gbare, G0, Σ0, F0, Π0pp,
Π0ph, G, Σ, F, Πpp, Πph) plus a `convergence` group from `save_convergence!`
(converged flag, residual_norm, iterations, tol, maxiter). Reload the solver with
`load_hubbard_solver`; read the convergence verdict with `load_convergence`
(see load_hubbard_data.jl). Post-processing lives in precompute_sigma_chi.jl and
the plot_sigma_chi_vs_*.jl scripts.
"""

using MPI
MPI.Init()

using fdDGAsolver
using MatsubaraFunctions
using StaticArrays
using HDF5
using Printf

# -----------------------------------------------------------------------
# Command-line arguments
# -----------------------------------------------------------------------
nmax                  = length(ARGS) >= 1 ? parse(Int, ARGS[1])  : 10
nq                    = length(ARGS) >= 2 ? parse(Int, ARGS[2])  : 4
NL_method             = length(ARGS) >= 3 ? parse(Int, ARGS[3])  : 2
filename_dmft_partial = length(ARGS) >= 4 ? ARGS[4]              : "data\\high_temperature_U0.5.h5"
do_PA                 = length(ARGS) >= 5 ? parse(Bool, ARGS[5]) : false
do_fd                 = length(ARGS) >= 6 ? parse(Bool, ARGS[6]) : true

# Solver stopping criteria (shared by both runs and recorded in the saved files)
maxiter = 200
tol     = 1e-8

# -----------------------------------------------------------------------
# Convergence metadata helper
# -----------------------------------------------------------------------
# Store, next to the saved solver, whether the parquet iteration converged.
# `result` is the named tuple returned by `fdDGAsolver.solve!`; `result.res`
# is the NLsolve `SolverResults`, whose fields carry the convergence verdict,
# the final residual norm and the iteration count. Booleans are written as
# Int (0/1) so the group is readable from h5py / TRIQS as well.
function save_convergence!(f, result; tol, maxiter)
    res = result.res
    g = create_group(f, "convergence")
    g["converged"]     = Int(res.f_converged || res.x_converged)
    g["f_converged"]   = Int(res.f_converged)
    g["x_converged"]   = Int(res.x_converged)
    g["residual_norm"] = res.residual_norm
    g["iterations"]    = res.iterations
    g["f_calls"]       = res.f_calls
    g["tol"]           = tol
    g["maxiter"]       = maxiter
    return nothing
end

# -----------------------------------------------------------------------
# Save one converged solver together with its convergence metadata.
# Returns the output path. Intended to be called on the main rank only.
# -----------------------------------------------------------------------
function save_solver!(output_dir, tag, S, result; U, T, nmax, nq, NL_method, tol, maxiter)
    fname = joinpath(output_dir,
        "hubbard_$(tag)_U$(U)_T$(T)_nmax$(nmax)_nq$(nq)_NL$(NL_method).h5")
    h5open(fname, "w") do f
        save!(f, "S", S)
        save_convergence!(f, result; tol, maxiter)
    end
    return fname
end

# -----------------------------------------------------------------------
# Load DMFT reference data (PHS Hubbard model near the 2PI vertex divergence)
# -----------------------------------------------------------------------
filename_dmft = normpath(joinpath(dirname(pathof(fdDGAsolver)), "..", filename_dmft_partial))
data_triqs    = load_triqs_data(filename_dmft)

(; T, U, μ, t1, t2, t3) = data_triqs.params

mpi_ismain() && begin
    println("=== PHS Hubbard model ===")
    @printf "T = %.4f,  U = %.4f,  μ = %.4f,  t1 = %.4f\n" T U μ t1
    @printf "nmax = %d,  nq = %d\n" nmax nq 
end

# -----------------------------------------------------------------------
# Frequency / momentum meshes
# -----------------------------------------------------------------------
k1_vec = 2π * SVector(1., 0.)
k2_vec = 2π * SVector(0., 1.)

nG  = 4nmax   # Matsubara box for G and Σ
nK1 = 4nmax   # bosonic box for K1 and bubble
nK2 = (nmax, nmax)
nK3 = (nmax, nmax)

# Dense k-mesh for single-particle quantities (G, Σ)
mk_G = BrillouinZoneMesh(BrillouinZone(48, k1_vec, k2_vec))

# Coarse k-mesh for the two-particle vertex (K1, K2, K3)
mk_F = BrillouinZoneMesh(BrillouinZone(nq, k1_vec, k2_vec))

Gbare = hubbard_bare_Green(MatsubaraMesh(T, nG, Fermion), mk_G; μ, t1, t2, t3)

# Bosonic Ω = 0 and lowest positive fermionic Matsubara frequencies
Ω0 = MatsubaraFrequency(T, 0, Boson)     # iω = 0
ν0 = MatsubaraFrequency(T, 0, Fermion)   # iν = iπT

# -----------------------------------------------------------------------
# Output directory — created up front so each solver can be saved as soon
# as it converges and then dropped from memory before the next is built.
# -----------------------------------------------------------------------
output_dir = joinpath(dirname(pathof(fdDGAsolver)), "../output/hubbard_model")
mpi_ismain() && mkpath(output_dir)


# -----------------------------------------------------------------------
# 1) Parquet Approximation (PA) — no DMFT input, bare vertex as F0
# -----------------------------------------------------------------------
if do_PA == true
    mpi_ismain() && println("\n=== Running PA ===")

    mpi_ismain() && println("Create solver.")
    # create solver object
    if NL_method == 1
        # approximate K2(q,k) ≈ K2(q) and K3(q, k, k') ≈ K3(q)
        S_PA = parquet_solver_hubbard_parquet_approximation(
            nG, nK1, nK2, nK3, mk_G, mk_F;
            T, U, μ, t1, t2, t3, mode = :hybrid,
        )
    elseif NL_method == 2
        # approximate K3(q, k, k') ≈ K3(q)
        S_PA = parquet_solver_hubbard_parquet_approximation_NL2(
            nG, nK1, nK2, nK3, mk_G, mk_F;
            T, U, μ, t1, t2, t3, mode = :hybrid,
        )
    elseif NL_method == 3
        # full K2 and K3 momentum dependence
        S_PA = parquet_solver_hubbard_parquet_approximation_NL3(
            nG, nK1, nK2, nK3, mk_G, mk_F;
            T, U, μ, t1, t2, t3, mode = :hybrid,
        )
    else
        error("NL method not recognized!")
    end

    mpi_ismain() && println("Initialize symmetry group.")
    # fill symmertries
    init_sym_grp!(S_PA)

    mpi_ismain() && println("Start solver.")
    # solve
    result_PA = fdDGAsolver.solve!(S_PA;
        strategy = :scPA,
        maxiter,
        tol,
        verbose  = mpi_ismain(),
    )

    # Save immediately, then release the PA solver so it does not occupy
    # memory while the (equally large) fd solver is built and solved.
    if mpi_ismain()
        println("\n=== Saving sc-data ===")
        fname_PA = save_solver!(output_dir, "sc", S_PA, result_PA;
            U, T, nmax, nq, NL_method, tol, maxiter)
        println("sc-data saved to $fname_PA")
    end

    # Drop references on every rank (each rank holds a full solver copy) and
    # reclaim the arrays before continuing.
    global S_PA      = nothing
    global result_PA = nothing
end

# -----------------------------------------------------------------------
# 2) fd-parquet DΓA (pDΓA) — DMFT vertex as reference
# -----------------------------------------------------------------------
if do_fd == true
    mpi_ismain() && println("\n=== Running pDΓA ===")

    mpi_ismain() && println("Load reference system.")
    # Reference single-particle quantities from DMFT (k-independent / local)
    G0 = copy(Gbare); set!(G0, 0)
    Σ0 = copy(Gbare); set!(Σ0, 0)
    for ν in meshes(G0, Val(1))
        view(G0, ν, :) .= data_triqs.G(value(ν))
        view(Σ0, ν, :) .= data_triqs.Σ(value(ν))
    end
    occ_target = data_triqs.occ

    mpi_ismain() && println("Create solver.")
    if NL_method == 1
        F0 = NL_Vertex(data_triqs.Γ, T, nK1, nK2, nK3, mk_F)
        # approximate K2(q,k) ≈ K2(q) and K3(q, k, k') ≈ K3(q)
        S_fd = NL_ParquetSolver(
            nK1, nK2, nK3, mk_F, copy(Gbare), G0, Σ0, F0;
            mode = :hybrid,
        )
    elseif NL_method == 2
        F0 = NL2_Vertex(data_triqs.Γ, T, nK1, nK2, nK3, mk_F)
        # approximate K3(q, k, k') ≈ K3(q)
        S_fd = NL2_ParquetSolver(
            nK1, nK2, nK3, mk_F, copy(Gbare), G0, Σ0, F0;
            mode = :hybrid,
        )
    elseif NL_method == 3
        F0 = NL3_Vertex(data_triqs.Γ, T, nK1, nK2, nK3, mk_F)
        # full K2 and K3 momentum dependence
        S_fd = NL3_ParquetSolver(
            nK1, nK2, nK3, mk_F, copy(Gbare), G0, Σ0, F0;
            mode = :hybrid,
        )
    else
        error("NL method not recognized!")
    end

    # release large reference vertex (F0) and reference single-particle inputs.
    global F0        = nothing
    global G0        = nothing
    global Σ0        = nothing

    mpi_ismain() && println("Initialize symmetry group.")
    # fill symmertries
    @time init_sym_grp!(S_fd)

    mpi_ismain() && println("Start solver.")
    # solve
    @time result_fd = fdDGAsolver.solve!(S_fd;
        strategy = :fdPA,
        maxiter,
        tol,
        verbose  = mpi_ismain(),
    )

    # Save immediately, then release the fd solver
    if mpi_ismain()
        println("\n=== Saving fd-data ===")
        fname_fd = save_solver!(output_dir, "fd", S_fd, result_fd;
            U, T, nmax, nq, NL_method, tol, maxiter)
        println("fd-data saved to $fname_fd")
    end

    global S_fd      = nothing
    global result_fd = nothing
    
end