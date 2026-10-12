# =============================================================================
# Bethe-Salpeter equation (BSE) kernels of the lattice solver NL2.
#
# The same fd-BSE as for the NL solver (nonlocal/BSE.jl; Lihm 2025, Eq. (5)
# mirrored, Eq. (S29) per class), at each bosonic momentum q. In NL2 the K2
# class and the bubbles keep the fermionic momentum, so the K1 and K2 kernels
# sum over the internal momentum k̃ together with ν̃. The K3 kernels stay
# s-wave. The channel, spin component and sign are chosen by the wrappers in
# BSE_templates.jl. BSE_1loop.jl holds the one-loop variants for the strategy
# :fdPA_1loop.
# =============================================================================

include("BSEa/BSEa_K1.jl")
include("BSEa/BSEa_K2.jl")
include("BSEa/BSEa_K3.jl")

include("BSEa/BSE_1loop.jl")
