# =============================================================================
# Bethe-Salpeter equation (BSE) kernels of the s-wave lattice solver NL.
#
# The same fd-BSE as for the local solver (BSE.jl; Lihm 2025, Eq. (5) mirrored,
# Eq. (S29) per class), at each bosonic momentum q, with all vertices and
# bubbles averaged over the fermionic momenta (s-wave approximation,
# Eqs. (S17)-(S18)). The channel, spin component and sign are chosen by the
# wrappers in BSE_templates.jl. BSE_1loop.jl holds the one-loop variants for
# the strategy :fdPA_1loop.
# =============================================================================

include("BSEa/BSEa_K1.jl")
include("BSEa/BSEa_K2.jl")
include("BSEa/BSEa_K3.jl")

include("BSEa/BSE_1loop.jl")
