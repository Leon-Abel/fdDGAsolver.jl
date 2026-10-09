# =============================================================================
# Printing helpers: print only once when running with MPI, and format tables.
# =============================================================================

using Printf

"""
    @mainprintf(fmt, args...)

Like `@printf`, but only the main MPI process prints. Without this, every MPI
process would print the same message.

Needs `using MatsubaraFunctions` in the calling script (for `mpi_ismain`).
"""
macro mainprintf(fmt, args...)
    quote
        $(esc(:mpi_ismain))() && Printf.@printf($fmt, $(map(esc, args)...))
    end
end

# -----------------------------------------------------------------------------
# Report formatting
# -----------------------------------------------------------------------------

"""
    format_duration(seconds) -> String

Format a time in seconds for printing: `12.3 s`, `4 m 05 s` or `1 h 12 m`.
"""
function format_duration(s)
    s < 60    && return @sprintf("%.1f s", s)
    s < 3600  && return @sprintf("%d m %02d s", s ÷ 60, round(Int, s % 60))
    return @sprintf("%d h %02d m", s ÷ 3600, (s % 3600) ÷ 60)
end

"""
    print_table(header, rows; io = stdout)

Print a table with aligned columns and a `---` line under the header.
Each row is a list of strings, one per column.
"""
function print_table(header, rows; io = stdout)
    rows = collect(rows)
    if isempty(rows)
        println(io, join(header, "  "))
        return nothing
    end
    widths = [maximum(length, (header[j], (r[j] for r in rows)...)) for j in eachindex(header)]
    println(io, join((rpad(header[j], widths[j]) for j in eachindex(header)), "  "))
    println(io, join(("-"^widths[j] for j in eachindex(header)), "  "))
    for r in rows
        println(io, join((rpad(r[j], widths[j]) for j in eachindex(header)), "  "))
    end
    return nothing
end

"""
    write_tsv(path, header, rows)

Save the same table as `print_table` to a tab-separated file.
"""
function write_tsv(path, header, rows)
    mkpath(dirname(path))
    open(path, "w") do io
        println(io, join(header, '\t'))
        for r in rows
            println(io, join(r, '\t'))
        end
    end
    return path
end
