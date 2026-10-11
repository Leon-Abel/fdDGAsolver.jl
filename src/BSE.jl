# =============================================================================
# Bethe-Salpeter equation (BSE) kernels of the local solver, one file per
# asymptotic class (K1, K2, K3).
#
# The kernels solve the finite-difference BSE of Lihm 2025, Eq. (5), in its
# left-right mirrored form, class by class (Eq. (S29)). They are written once
# for all channels. The channel, spin component and sign are chosen by the
# wrappers in BSE_templates.jl. The folder name "BSEa" is historical: these
# kernels started as the a-channel versions.
# =============================================================================

include("BSEa/BSEa_K1.jl")
include("BSEa/BSEa_K2.jl")
include("BSEa/BSEa_K3.jl")
