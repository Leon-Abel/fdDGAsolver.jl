# =============================================================================
# Helpers shared by the plot scripts.
# =============================================================================

using HDF5

# -----------------------------------------------------------------------------
# Curve selection
# -----------------------------------------------------------------------------

"""
    selected(what, toggles...) -> Vector

Return the values whose switch is on, in the given order (the order the curves
are drawn in). Errors if none is on.

    selected("NL method", (NL1, 1), (NL2, 2), (NL3, 3))
"""
function selected(what, toggles...)
    v = [x for (on, x) in toggles if on]
    isempty(v) && error("Select at least one $what.")
    return v
end

# -----------------------------------------------------------------------------
# Momentum labels and grid index
# -----------------------------------------------------------------------------

"""
    k_index(qx_pi, qy_pi, nk) -> Int

Index of the momentum `(qx_pi, qy_pi)` (in units of π) on the `nk × nk` k-grid,
which covers 0..2π in each direction with kx running fastest.
"""
function k_index(qx_pi, qy_pi, nk)
    step(a) = mod(round(Int, a * nk / 2), nk)
    return step(qy_pi) * nk + step(qx_pi) + 1
end

_near(a, b) = isapprox(a, b; atol = 1e-6)
_fmt_pi(a)  = replace(string(round(a, digits = 3)), r"\.0$" => "")   # 1.0 -> "1", 0.5 -> "0.5"

"Return `:Γ`, `:X` or `:M` if `(qx_pi, qy_pi)` is that point, otherwise `nothing`."
function _high_symmetry(qx_pi, qy_pi)
    _near(qx_pi, 0) && _near(qy_pi, 0) && return :Γ
    ((_near(qx_pi, 1) && _near(qy_pi, 0)) || (_near(qx_pi, 0) && _near(qy_pi, 1))) && return :X
    _near(qx_pi, 1) && _near(qy_pi, 1) && return :M
    return nothing
end

"""
    q_label(qx_pi, qy_pi) -> String

Text label for a momentum (in units of π), e.g. `"M (1π, 1π)"` or `"(0.5π, 0.5π)"`.
"""
function q_label(qx_pi, qy_pi)
    coord = "($(_fmt_pi(qx_pi))π, $(_fmt_pi(qy_pi))π)"
    hs = _high_symmetry(qx_pi, qy_pi)
    return hs === nothing ? coord : "$hs $coord"
end

"""
    q_tex(qx_pi, qy_pi) -> String

LaTeX label for a momentum, for figures (without the surrounding `\$`).
"""
function q_tex(qx_pi, qy_pi)
    hs = _high_symmetry(qx_pi, qy_pi)
    hs === :Γ && return "\\Gamma"
    hs === :X && return "\\text{X}"
    hs === :M && return "\\text{M}"
    return "($(_fmt_pi(qx_pi))\\pi, $(_fmt_pi(qy_pi))\\pi)"
end

# -----------------------------------------------------------------------------
# Picking one frequency from the cache
# -----------------------------------------------------------------------------

"Index of the lowest positive fermionic frequency ν = πT in the cached `nu` values."
nu0_row(nu, T) = argmin(abs.(nu .- π * T))

"Index of Ω = 0 in the cached `omega` values."
omega0_row(omega) = argmin(abs.(omega))

# -----------------------------------------------------------------------------
# Susceptibility at U = 0 (the U = 0 point of the χ curves)
# -----------------------------------------------------------------------------
# At U = 0, χ_D = χ_M = χ₀/2, where χ₀ is the static bare bubble. It only needs
# T, μ and the hoppings, so it is a quick k-sum, not a solver run.

_fermi(ξ, T) = 1 / (1 + exp(ξ / T))

"Band energy minus μ at k = 2π(i,j)/nk (same grid as `k_index`)."
_disp(i, j, nk, μ, t1, t2, t3) =
    -2t1 * (cos(2π*i/nk) + cos(2π*j/nk)) -
     4t2 *  cos(2π*i/nk) * cos(2π*j/nk) -
     2t3 * (cos(4π*i/nk) + cos(4π*j/nk)) - μ

"[f(ξk) − f(ξk')]/(ξk' − ξk), or −f'(ξk) when ξk = ξk'."
_lindhard(ξk, ξkp, T) = abs(ξkp - ξk) < 1e-10 ?
    _fermi(ξk, T) * (1 - _fermi(ξk, T)) / T :
    (_fermi(ξk, T) - _fermi(ξkp, T)) / (ξkp - ξk)

"U = 0 susceptibility χ₀(q)/2 at momentum q (used for the lattice curves)."
function bare_chi_q(qx_pi, qy_pi; T, μ, t1, t2, t3, nk = 48)
    sqx = mod(round(Int, qx_pi * nk / 2), nk)
    sqy = mod(round(Int, qy_pi * nk / 2), nk)
    s = 0.0
    for j in 0:nk-1, i in 0:nk-1
        ξk  = _disp(i, j, nk, μ, t1, t2, t3)
        ξkq = _disp(mod(i + sqx, nk), mod(j + sqy, nk), nk, μ, t1, t2, t3)
        s  += _lindhard(ξk, ξkq, T)
    end
    return s / nk^2 / 2
end

"U = 0 local susceptibility χ₀,loc/2, averaged over q (used for the DMFT curve)."
function bare_chi_local(; T, μ, t1, t2, t3, nk = 48)
    ξ  = [_disp(i, j, nk, μ, t1, t2, t3) for j in 0:nk-1 for i in 0:nk-1]
    fv = _fermi.(ξ, T)
    N  = length(ξ)
    s  = 0.0
    for b in 1:N
        ξb, fb = ξ[b], fv[b]
        for a in 1:N
            d  = ξb - ξ[a]
            s += abs(d) < 1e-10 ? fv[a] * (1 - fv[a]) / T : (fv[a] - fb) / d
        end
    end
    return s / N^2 / 2
end

"""
    dmft_temperature(U) -> Float64

Temperature T from the DMFT file for interaction `U`. Reads only T, not the
whole file.
"""
function dmft_temperature(U)
    f = dmft_file(U)
    isfile(f) || error("DMFT reference data not found:\n  $f")
    return h5read(f, "params/T")
end

"""
    dmft_hopping(U) -> Float64

Nearest-neighbour hopping |t1| from the DMFT file for interaction `U`. The
figures give U and T in units of it (U/t, T/t). The file stores t1 as a negative
number, so this returns its absolute value.
"""
function dmft_hopping(U)
    f = dmft_file(U)
    isfile(f) || error("DMFT reference data not found:\n  $f")
    return abs(Float64(h5read(f, "params/t1")))
end

"""
    dmft_model_params(Us) -> NamedTuple or nothing

Model parameters (T, μ, t1, t2, t3) from the first DMFT file found for the
values in `Us`. They are the same in every file. Returns `nothing` if no file
exists.
"""
function dmft_model_params(Us)
    for U in Us
        f = dmft_file(U)
        isfile(f) || continue
        p = h5open(f) do fh
            Dict(k => read(fh["params"][k]) for k in keys(fh["params"]))
        end
        return (; T  = p["T"], μ = get(p, "μ", get(p, "mu", 0.0)),
                  t1 = p["t1"], t2 = get(p, "t2", 0.0), t3 = get(p, "t3", 0.0))
    end
    return nothing
end

# -----------------------------------------------------------------------------
# Finding available runs
# -----------------------------------------------------------------------------

"""
    scan_available(nmax, nq, NL_method) -> Dict{String, Set{Tuple{Symbol,Symbol}}}

Which runs exist for the given `nmax`, `nq` and `NL_method`, as
`U => {(strategy, iteration_scheme)}` (U as a string, as in the file name).

Looks for both cache files and saved solvers, since a solver file is often
deleted once its cache exists.
"""
function scan_available(nmax, nq, NL_method)
    available = Dict{String, Set{Tuple{Symbol, Symbol}}}()
    dirs = (postprocess_dir(), solver_dir())
    for dir in dirs
        isdir(dir) || continue
        for f in readdir(dir)
            r = parse_run_basename(f)
            r === nothing && continue
            (r.nmax == nmax && r.nq == nq && r.NL == NL_method) || continue
            (r.strategy === nothing || r.iteration_scheme === nothing) && continue
            push!(get!(available, r.U_str, Set{Tuple{Symbol, Symbol}}()),
                  (r.strategy, r.iteration_scheme))
        end
    end
    return available
end

"""
    windowed_U(available_by_nl, nls; Umin, Umax, extra = String[]) -> Vector{String}

The U values to plot: all U found for the selected NL methods, plus `extra`,
kept only if between `Umin` and `Umax`, sorted. Returned as strings, as in the
file names.
"""
function windowed_U(available_by_nl, nls; Umin = -Inf, Umax = Inf, extra = String[])
    all_U = Set{String}()
    for nl in nls, U in keys(available_by_nl[nl])
        push!(all_U, U)
    end
    union!(all_U, extra)
    Us = sort!(collect(all_U); by = u -> parse(Float64, u))
    filter!(u -> Umin <= parse(Float64, u) <= Umax, Us)
    return Us
end

# -----------------------------------------------------------------------------
# Accuracy check
# -----------------------------------------------------------------------------

"""
    cache_residual(path) -> Float64

Final residual of the run behind a cache file: the largest change of any vertex
or self-energy value in the last iteration. Used to leave out runs that are not
accurate enough.

Returns `Inf` if the cache file does not exist, so such a run is always left out.
"""
cache_residual(path) = isfile(path) ? load_convergence(path).residual_norm : Inf
