"""
    ScriptRuns

Helpers to build, solve and save one run. Kept apart from [`ScriptCommon`](@ref)
because it loads `fdDGAsolver`, which is slow to load and not needed by batch
scripts.

Load `ScriptCommon` first, since this module uses it:

    include(joinpath(@__DIR__, "common", "ScriptCommon.jl"))
    using .ScriptCommon
    include(joinpath(@__DIR__, "common", "ScriptRuns.jl"))
    using .ScriptRuns
"""
module ScriptRuns

using ..ScriptCommon

include("runs.jl")

# Names a script can use directly after `using .ScriptRuns`.
export build_solver, build_solver_from_dmft
export solve_run!, save_run!

end # module
