"""
Plot χ_D(T) and χ_M(T) vs T from the HDF5 output of
`script/run/calculate_susceptibilities.jl`.

All THREE scans are drawn together — that comparison is what the figure is — so
all three files have to be present. The producer can be told to skip any of them
(`--no-sc` / `--no-fd` / `--no-HA`), and this says which one is missing and how to
make it rather than failing inside HDF5.

Usage (from anywhere):
  julia --project=script script/plot/plot_susceptibilities.jl [options]

Run with `--help` for the full option list.

  --sc --fd --HA   the three input files, as paths relative to the repo root.
                   The defaults are `chi_scan_path` (common/paths.jl) at the
                   producer's own default box — the one rule both scripts spell
                   the name with — i.e. output/Chi_{sc,fd,HA}_48_1.0_2.0_41.h5

Relative paths are resolved against the repo root, not the current directory, so
the script runs from anywhere.
"""

using HDF5
using CairoMakie   # swap to GLMakie for an interactive window
using Printf

@isdefined(ScriptCommon) || include(joinpath(@__DIR__, "..", "common", "ScriptCommon.jl"))
using .ScriptCommon

# -----------------------------------------------------------------------
# Locate the HDF5 files
# -----------------------------------------------------------------------
# ---------------------------------------------------------------------------
# What executing this file in the REPL does. Edit these and execute it again.
#
# Running the file as a script overrides them from the command line - and these
# are the defaults `--help` reports, so the two entry points cannot drift apart.
# ---------------------------------------------------------------------------
# The box `calculate_susceptibilities.jl` scans by default; naming it here rather
# than the finished file names is what keeps the two in step.
scan_box = (48, 1.0, 2.0, 41)               # (nmax, T_lo, T_hi, nT)

sc_file = chi_scan_path(:sc, scan_box...)   # PA input
fd_file = chi_scan_path(:fd, scan_box...)   # pDΓA input
HA_file = chi_scan_path(:HA, scan_box...)   # atomic-limit input

# ---------------------------------------------------------------------------
# CLI driver (skipped when this file is executed in the REPL or `include`d)
# ---------------------------------------------------------------------------
if abspath(PROGRAM_FILE) == @__FILE__
    settings = cli_settings("Plot χ_D(T) and χ_M(T) from the saved susceptibility scans.")
    add_arg_table!(settings,
        ["--sc"], Dict(:help => "PA input, relative to the repo root",
                       :arg_type => String, :default => sc_file),
        ["--fd"], Dict(:help => "pDΓA input, relative to the repo root",
                       :arg_type => String, :default => fd_file),
        ["--HA"], Dict(:help => "atomic-limit input, relative to the repo root",
                       :arg_type => String, :default => HA_file))

    parsed = parse_args(ARGS, settings)

    sc_file = parsed["sc"]
    fd_file = parsed["fd"]
    HA_file = parsed["HA"]
end

resolve(p) = isabspath(p) ? p : repo_path(p)

"""
    load_data(; sc_file, fd_file, HA_file) -> NamedTuple

Read the three susceptibility scans plus the NRG reference. Each keyword defaults to
the parameter block above, so `load_data()` loads exactly what executing this file
would.
"""
function load_data(; sc_file = sc_file, fd_file = fd_file, HA_file = HA_file)
    filename_sc = resolve(sc_file)
    filename_fd = resolve(fd_file)
    filename_HA = resolve(HA_file)

    # The producer writes one file per method and can be told to skip any of them, so
    # a missing input here is an ordinary outcome rather than a broken install. Say
    # which one and what makes it, instead of letting `h5read` fail on the first.
    for (tag, f, flag) in ((:sc, filename_sc, "--no-sc"),
                           (:fd, filename_fd, "--no-fd"),
                           (:HA, filename_HA, "--no-HA"))
        isfile(f) && continue
        error("""
              No $(tag) susceptibility scan at
                $f
              Produce it with
                julia --project=script -t <nthreads> script/run/calculate_susceptibilities.jl
              (that script writes all three unless told otherwise with $(flag)), or point
              --$(tag) at an existing file.""")
    end

    # One (T, χ_D, χ_M) series per method.
    function read_scan(file, suffix)
        println("Reading: $file")
        return (T  = h5read(file, "T_values"),
                χD = h5read(file, "chiD_$suffix"),
                χM = h5read(file, "chiM_$suffix"))
    end

    sc = read_scan(filename_sc, "sc")
    fd = read_scan(filename_fd, "fd")
    HA = read_scan(filename_HA, "HA")

    # NRG reference data. χ_D and χ_M were digitized at slightly different T, so each
    # carries its own T axis.
    NRG_D = (T = [1.0483466045764112, 1.5322254472458896, 2.0161042899153676,
                  2.9838448993873428, 3.951585508859316],
             χ = [0.04238998215808115, 0.04782324195575556, 0.048834430508802304,
                  0.04524003093771149, 0.040075718291043005])
    NRG_M = (T = [1.067347560731358, 1.5480902420879996,
                  2.028937470602603, 2.990422833315886, 3.9520127431871295],
             χ = [0.37984341637010666, 0.24745907473309592,
                  0.1796697508896795, 0.11264056939501776, 0.08048398576512461])

    # The figure is written next to the pDΓA input.
    outfile = replace(filename_fd, ".h5" => ".svg")

    return (; sc, fd, HA, NRG_D, NRG_M, outfile)
end

"""
    plot_data(d) -> Figure

Draw and save the figure from `load_data`'s result. Nothing here re-reads a file,
so retuning the layout costs the drawing alone:

    julia> data = load_data()
    julia> plot_data(data)
"""
function plot_data(d)
    sc, fd, HA, NRG_D, NRG_M = d.sc, d.fd, d.HA, d.NRG_D, d.NRG_M

    println("Plotting Figure")
    # font, line and marker sizes tuned for projection on a presentation slide
    figure = with_theme(theme_latexfonts();
        fontsize = 22,
        Axis   = (titlesize = 26, xlabelsize = 26, ylabelsize = 26,
                  xticklabelsize = 22, yticklabelsize = 22),
        Legend = (labelsize = 20,),
        Lines        = (linewidth = 3,),
        ScatterLines = (linewidth = 3, markersize = 12),
        VLines       = (linewidth = 2,),
    ) do
        # -----------------------------------------------------------------------
        # Figure layout: main panel (χ_D) + inset (χ_M)
        # -----------------------------------------------------------------------
        fig = Figure(size = (700, 500))

        # Main axes for χ_D (charge susceptibility)
        # (title is long, so it stays below the theme's titlesize to fit the figure width)
        ax_main = Axis(fig[1, 1];
            xlabel       = L"T",
            ylabel       = L"\chi_D",
            title        = L"$U = 5.75$, $V=2$, $D = 10$, $\text{half-filling}$",
            limits       = (0.95, 4.05, 0.034, 0.066),
            xtickalign   = 1.0,
            ytickalign   = 1.0,
            xticksmirrored = true,
            yticksmirrored = true,
            xticks       = ([1,1.58,2,3,4], ["1", L"T_c", "2", "3", "4"]),
            yticks       = 0.035:0.010:0.065
        )

        # Inset axes for χ_M (magnetic susceptibility): sizes scaled down from the theme
        ax_inset = Axis(fig[1, 1];
            width        = Relative(0.4),
            height       = Relative(0.4),
            halign       = 0.95,
            valign       = 0.95,
            xlabel       = L"T",
            ylabel       = L"\chi_M",
            xlabelsize   = 22,
            ylabelsize   = 22,
            xticklabelsize = 18,
            yticklabelsize = 18,
            limits       = (0.95, 4.05, 0.075, 0.425),
        )

        # -----------------------------------------------------------------------
        # Plot χ_D on the main axes
        # -----------------------------------------------------------------------
        vlines!(ax_main, [1.58];
            color = :gray,
        )

        # line and marker sizes come from the theme above
        lines!(ax_main, HA.T, HA.χD;
            color = :black,
            linestyle = :dash,
            label     = "HA",
        )

        scatterlines!(ax_main, sc.T, sc.χD;
            label     = "PA",
            marker = :rect,
        )

        scatterlines!(ax_main, fd.T, fd.χD;
            label     = "fd-DΓA(HA)",
            marker = :circle,
        )

        scatterlines!(ax_main, NRG_D.T, NRG_D.χ;
            label     = "NRG",
            marker = :diamond,
        )

        axislegend(ax_main; position = :cb, framevisible = false, rowgap = 6, padding = (8, 8, 8, 8))

        # -----------------------------------------------------------------------
        # Plot χ_M on the inset axes
        # -----------------------------------------------------------------------
        vlines!(ax_inset, [1.58];
            color = :gray,
            linewidth = 1.5
        )

        lines!(ax_inset, HA.T, HA.χM;
            color = :black,
            linewidth = 2,
            linestyle = :dash,
        )

        scatterlines!(ax_inset, sc.T, sc.χM;
            linewidth = 2,
            marker = :rect,
            markersize = 8
        )

        scatterlines!(ax_inset, fd.T, fd.χM;
            linewidth = 2,
            marker = :circle,
            markersize = 8
        )

        scatterlines!(ax_inset, NRG_M.T, NRG_M.χ;
            linewidth = 2,
            marker = :diamond,
            markersize = 8,
        )

        Makie.translate!(ax_inset.blockscene, 0, 0, 150)

        # -----------------------------------------------------------------------
        # Save
        # -----------------------------------------------------------------------
        println("Saving figure")
        save(d.outfile, fig)
        println("Figure written to $(d.outfile)")

        return fig
    end

    return figure
end

# ---------------------------------------------------------------------------
# Executing this file does the whole thing. The two halves are also callable on
# their own, so a layout tweak need not re-read the scans:
#
#   julia> data = load_data()    # once
#   julia> plot_data(data)       # as often as you like
# ---------------------------------------------------------------------------
data = load_data()
fig  = plot_data(data)
