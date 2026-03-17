# symmetries.jl
#
# Symmetry transformations for Green's functions and vertices:
#   - self-energy symmetries (sΣ)
#   - particle-particle channel symmetries (sK1pp, sK2pp, sK3pp)
#   - particle-hole channel symmetries (sK1ph, sK2ph, sK3ph)
#
# These functions return a transformed frequency NTuple and an `Operation` 
# describing sign changes (sgn) or complex conjugation (con).

# self-energy symmetries
#----------------------------------------------------------------------------------------------#
"""
    function sΣ(w::NTuple{1, MatsubaraFrequency},)::Tuple{NTuple{1, MatsubaraFrequency}, Operation}

Time reversal symmetry for the self-energy 'Σ': 'Σ(iw) = Σ*(-iw)'.
Since the implementation stores iΣ, the relation transforms as 'iΣ(iw) = -(iΣ)*(-iw)'.
"""
function sΣ(
    w :: NTuple{1, MatsubaraFrequency},
    ) :: Tuple{NTuple{1, MatsubaraFrequency}, Operation}

    # Maps w -> -w, flips sign, and applies conjugation.
    # We store iΣ, so the symmetry Σ -> Σ* becomes iΣ -> -(iΣ)*.
    return (-w[1],), Operation(sgn = true, con = true)
end

# particle-particle (pp) symmetries
#----------------------------------------------------------------------------------------------#
"""
    function sK1pp(w::NTuple{1, MatsubaraFrequency},)::Tuple{NTuple{1, MatsubaraFrequency}, Operation}

pp-channel K1 class symmetry: K1pp(iw) = K1pp*(-iw)
"""
function sK1pp(
    w :: NTuple{1, MatsubaraFrequency},
    ) :: Tuple{NTuple{1, MatsubaraFrequency}, Operation}

    return (-w[1],), Operation(sgn = false, con = true)
end

"""
    function sK2pp1(w::NTuple{2, MatsubaraFrequency},)::Tuple{NTuple{2, MatsubaraFrequency}, Operation}

pp-channel K2 class symmetry (1): K2pp(iw1, iw2) = K2pp*(-iw1, -iw2)
"""
function sK2pp1(
    w :: NTuple{2, MatsubaraFrequency},
    ) :: Tuple{NTuple{2, MatsubaraFrequency}, Operation}

    return (-w[1], -w[2]), Operation(sgn = false, con = true)
end

"""
    function sK2pp2(w::NTuple{2, MatsubaraFrequency},)::Tuple{NTuple{2, MatsubaraFrequency}, Operation}

pp-channel K2 class symmetry (2): K2pp(iw1, iw2) = K2pp(iw1, iw1 - iw2)
"""
function sK2pp2(
    w :: NTuple{2, MatsubaraFrequency},
    ) :: Tuple{NTuple{2, MatsubaraFrequency}, Operation}

    return (w[1], w[1] - w[2]), Operation()
end

"""
    function sK3pp1(w::NTuple{3, MatsubaraFrequency},)::Tuple{NTuple{3, MatsubaraFrequency}, Operation}

pp-channel K3 class symmetry (1): K3pp(iw1, iw2, iw3) = K3pp*(-iw1, -iw2, -iw3)
"""
function sK3pp1(
    w :: NTuple{3, MatsubaraFrequency},
    ) :: Tuple{NTuple{3, MatsubaraFrequency}, Operation}

    return (-w[1], -w[2], -w[3]), Operation(sgn = false, con = true)
end

"""
    function sK3pp2(w::NTuple{3, MatsubaraFrequency},)::Tuple{NTuple{3, MatsubaraFrequency}, Operation}

pp-channel K3 class symmetry (2): K3pp(iw1, iw2, iw3) = K3pp(iw1, iw3, iw2)
"""
function sK3pp2(
    w :: NTuple{3, MatsubaraFrequency},
    ) :: Tuple{NTuple{3, MatsubaraFrequency}, Operation}

    return (w[1], w[3], w[2]), Operation()
end

"""
    function sK3pp3(w::NTuple{3, MatsubaraFrequency},)::Tuple{NTuple{3, MatsubaraFrequency}, Operation}

pp-channel K3 class symmetry (3): K3pp(iw1, iw2, iw3) = K3pp(iw1, iw1 - iw2, iw1 - iw3)
"""
function sK3pp3(
    w :: NTuple{3, MatsubaraFrequency},
    ) :: Tuple{NTuple{3, MatsubaraFrequency}, Operation}

    return (w[1], w[1] - w[2], w[1] - w[3]), Operation()
end

# particle-hole symmetries
#----------------------------------------------------------------------------------------------#
# many identical to pp symmetries
"""
    function sK1ph(w::NTuple{1, MatsubaraFrequency},)::Tuple{NTuple{1, MatsubaraFrequency}, Operation}

ph-channel K1 class symmetry: K1ph(iw) = K1ph*(-iw)
"""
function sK1ph(
    w :: NTuple{1, MatsubaraFrequency},
    ) :: Tuple{NTuple{1, MatsubaraFrequency}, Operation}

    return sK1pp(w)
end

"""
    function sK2ph1(w::NTuple{2, MatsubaraFrequency},)::Tuple{NTuple{2, MatsubaraFrequency}, Operation}

ph-channel K2 class symmetry (1): K2ph(iw1, iw2) = K2ph*(-iw1, -iw2)
"""
function sK2ph1(
    w :: NTuple{2, MatsubaraFrequency},
    ) :: Tuple{NTuple{2, MatsubaraFrequency}, Operation}

    return sK2pp1(w)
end

"""
    function sK2ph2(w::NTuple{2, MatsubaraFrequency},)::Tuple{NTuple{2, MatsubaraFrequency}, Operation}

ph-channel K2 class symmetry (2): K2pp(iw1, iw2) = K2pp(-iw1, iw1 + iw2)
"""
function sK2ph2(
    w :: NTuple{2, MatsubaraFrequency},
    ) :: Tuple{NTuple{2, MatsubaraFrequency}, Operation}

    return (-w[1], w[1] + w[2]), Operation()
end

"""
    function sK3ph1(w::NTuple{3, MatsubaraFrequency},)::Tuple{NTuple{3, MatsubaraFrequency}, Operation}

ph-channel K3 class symmetry (1): K3php(iw1, iw2, iw3) = K3ph*(-iw1, -iw2, -iw3)
"""
function sK3ph1(
    w :: NTuple{3, MatsubaraFrequency},
    ) :: Tuple{NTuple{3, MatsubaraFrequency}, Operation}

    return sK3pp1(w)
end

"""
    function sK3ph2(w::NTuple{3, MatsubaraFrequency},)::Tuple{NTuple{3, MatsubaraFrequency}, Operation}

ph-channel K3 class symmetry (2): K3ph(iw1, iw2, iw3) = K3ph(iw1, iw3, iw2)
"""
function sK3ph2(
    w :: NTuple{3, MatsubaraFrequency},
    ) :: Tuple{NTuple{3, MatsubaraFrequency}, Operation}

    return sK3pp2(w)
end

"""
    function sK3ph3(w::NTuple{3, MatsubaraFrequency},)::Tuple{NTuple{3, MatsubaraFrequency}, Operation}

ph-channel K3 class symmetry (3): K3ph(iw1, iw2, iw3) = K3ph(-iw1, iw1 + iw2, iw1 + iw3)
"""
function sK3ph3(
    w :: NTuple{3, MatsubaraFrequency},
    ) :: Tuple{NTuple{3, MatsubaraFrequency}, Operation}

    return (-w[1], w[1] + w[2], w[1] + w[3]), Operation()
end
