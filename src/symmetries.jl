# symmetries.jl
#
# Symmetry transformations for Green's functions and vertices:
#   - self-energy symmetries (sΣ)
#   - particle-particle channel symmetries (sK1pp, sK2pp, sK3pp)
#   - particle-hole channel symmetries (sK1ph, sK2ph, sK3ph)
#
# These functions return a transformed frequency NTuple and an `Operation`
# describing sign changes (sgn) or complex conjugation (con).
# All functions take an explicit ::Type{Q} argument so the returned Operation{Q}
# matches the data type of the mesh function they will be applied to.

# self-energy symmetries
#----------------------------------------------------------------------------------------------#
"""
    function sΣ(w::NTuple{1, MatsubaraFrequency}, ::Type{Q})::Tuple{NTuple{1, MatsubaraFrequency}, Operation{Q}}

Time reversal symmetry for the self-energy 'Σ': 'Σ(iw) = Σ*(-iw)'.
Since the implementation stores iΣ, the relation transforms as 'iΣ(iw) = -(iΣ)*(-iw)'.
"""
function sΣ(
    w :: NTuple{1, MatsubaraFrequency},
    :: Type{Q},
    ) :: Tuple{NTuple{1, MatsubaraFrequency}, Operation{Q}} where {Q <: Number}

    # Maps w -> -w, flips sign, and applies conjugation.
    # We store iΣ, so the symmetry Σ -> Σ* becomes iΣ -> -(iΣ)*.
    return (-w[1],), Operation{Q}(sgn = true, con = true)
end

# particle-particle (pp) symmetries
#----------------------------------------------------------------------------------------------#
"""
    function sK1pp(w::NTuple{1, MatsubaraFrequency}, ::Type{Q})::Tuple{NTuple{1, MatsubaraFrequency}, Operation{Q}}

pp-channel K1 class symmetry: K1pp(iw) = K1pp*(-iw)
"""
function sK1pp(
    w :: NTuple{1, MatsubaraFrequency},
    :: Type{Q},
    ) :: Tuple{NTuple{1, MatsubaraFrequency}, Operation{Q}} where {Q <: Number}

    return (-w[1],), Operation{Q}(sgn = false, con = true)
end

"""
    function sK2pp1(w::NTuple{2, MatsubaraFrequency}, ::Type{Q})::Tuple{NTuple{2, MatsubaraFrequency}, Operation{Q}}

pp-channel K2 class symmetry (1): K2pp(iw1, iw2) = K2pp*(-iw1, -iw2)
"""
function sK2pp1(
    w :: NTuple{2, MatsubaraFrequency},
    :: Type{Q},
    ) :: Tuple{NTuple{2, MatsubaraFrequency}, Operation{Q}} where {Q <: Number}

    return (-w[1], -w[2]), Operation{Q}(sgn = false, con = true)
end

"""
    function sK2pp2(w::NTuple{2, MatsubaraFrequency}, ::Type{Q})::Tuple{NTuple{2, MatsubaraFrequency}, Operation{Q}}

pp-channel K2 class symmetry (2): K2pp(iw1, iw2) = K2pp(iw1, iw1 - iw2)
"""
function sK2pp2(
    w :: NTuple{2, MatsubaraFrequency},
    :: Type{Q},
    ) :: Tuple{NTuple{2, MatsubaraFrequency}, Operation{Q}} where {Q <: Number}

    return (w[1], w[1] - w[2]), Operation{Q}()
end

"""
    function sK3pp1(w::NTuple{3, MatsubaraFrequency}, ::Type{Q})::Tuple{NTuple{3, MatsubaraFrequency}, Operation{Q}}

pp-channel K3 class symmetry (1): K3pp(iw1, iw2, iw3) = K3pp*(-iw1, -iw2, -iw3)
"""
function sK3pp1(
    w :: NTuple{3, MatsubaraFrequency},
    :: Type{Q},
    ) :: Tuple{NTuple{3, MatsubaraFrequency}, Operation{Q}} where {Q <: Number}

    return (-w[1], -w[2], -w[3]), Operation{Q}(sgn = false, con = true)
end

"""
    function sK3pp2(w::NTuple{3, MatsubaraFrequency}, ::Type{Q})::Tuple{NTuple{3, MatsubaraFrequency}, Operation{Q}}

pp-channel K3 class symmetry (2): K3pp(iw1, iw2, iw3) = K3pp(iw1, iw3, iw2)
"""
function sK3pp2(
    w :: NTuple{3, MatsubaraFrequency},
    :: Type{Q},
    ) :: Tuple{NTuple{3, MatsubaraFrequency}, Operation{Q}} where {Q <: Number}

    return (w[1], w[3], w[2]), Operation{Q}()
end

"""
    function sK3pp3(w::NTuple{3, MatsubaraFrequency}, ::Type{Q})::Tuple{NTuple{3, MatsubaraFrequency}, Operation{Q}}

pp-channel K3 class symmetry (3): K3pp(iw1, iw2, iw3) = K3pp(iw1, iw1 - iw2, iw1 - iw3)
"""
function sK3pp3(
    w :: NTuple{3, MatsubaraFrequency},
    :: Type{Q},
    ) :: Tuple{NTuple{3, MatsubaraFrequency}, Operation{Q}} where {Q <: Number}

    return (w[1], w[1] - w[2], w[1] - w[3]), Operation{Q}()
end

# particle-hole symmetries
#----------------------------------------------------------------------------------------------#
# many identical to pp symmetries
"""
    function sK1ph(w::NTuple{1, MatsubaraFrequency}, ::Type{Q})::Tuple{NTuple{1, MatsubaraFrequency}, Operation{Q}}

ph-channel K1 class symmetry: K1ph(iw) = K1ph*(-iw)
"""
function sK1ph(
    w :: NTuple{1, MatsubaraFrequency},
    :: Type{Q},
    ) :: Tuple{NTuple{1, MatsubaraFrequency}, Operation{Q}} where {Q <: Number}

    return sK1pp(w, Q)
end

"""
    function sK2ph1(w::NTuple{2, MatsubaraFrequency}, ::Type{Q})::Tuple{NTuple{2, MatsubaraFrequency}, Operation{Q}}

ph-channel K2 class symmetry (1): K2ph(iw1, iw2) = K2ph*(-iw1, -iw2)
"""
function sK2ph1(
    w :: NTuple{2, MatsubaraFrequency},
    :: Type{Q},
    ) :: Tuple{NTuple{2, MatsubaraFrequency}, Operation{Q}} where {Q <: Number}

    return sK2pp1(w, Q)
end

"""
    function sK2ph2(w::NTuple{2, MatsubaraFrequency}, ::Type{Q})::Tuple{NTuple{2, MatsubaraFrequency}, Operation{Q}}

ph-channel K2 class symmetry (2): K2pp(iw1, iw2) = K2pp(-iw1, iw1 + iw2)
"""
function sK2ph2(
    w :: NTuple{2, MatsubaraFrequency},
    :: Type{Q},
    ) :: Tuple{NTuple{2, MatsubaraFrequency}, Operation{Q}} where {Q <: Number}

    return (-w[1], w[1] + w[2]), Operation{Q}()
end

"""
    function sK3ph1(w::NTuple{3, MatsubaraFrequency}, ::Type{Q})::Tuple{NTuple{3, MatsubaraFrequency}, Operation{Q}}

ph-channel K3 class symmetry (1): K3php(iw1, iw2, iw3) = K3ph*(-iw1, -iw2, -iw3)
"""
function sK3ph1(
    w :: NTuple{3, MatsubaraFrequency},
    :: Type{Q},
    ) :: Tuple{NTuple{3, MatsubaraFrequency}, Operation{Q}} where {Q <: Number}

    return sK3pp1(w, Q)
end

"""
    function sK3ph2(w::NTuple{3, MatsubaraFrequency}, ::Type{Q})::Tuple{NTuple{3, MatsubaraFrequency}, Operation{Q}}

ph-channel K3 class symmetry (2): K3ph(iw1, iw2, iw3) = K3ph(iw1, iw3, iw2)
"""
function sK3ph2(
    w :: NTuple{3, MatsubaraFrequency},
    :: Type{Q},
    ) :: Tuple{NTuple{3, MatsubaraFrequency}, Operation{Q}} where {Q <: Number}

    return sK3pp2(w, Q)
end

"""
    function sK3ph3(w::NTuple{3, MatsubaraFrequency}, ::Type{Q})::Tuple{NTuple{3, MatsubaraFrequency}, Operation{Q}}

ph-channel K3 class symmetry (3): K3ph(iw1, iw2, iw3) = K3ph(-iw1, iw1 + iw2, iw1 + iw3)
"""
function sK3ph3(
    w :: NTuple{3, MatsubaraFrequency},
    :: Type{Q},
    ) :: Tuple{NTuple{3, MatsubaraFrequency}, Operation{Q}} where {Q <: Number}

    return (-w[1], w[1] + w[2], w[1] + w[3]), Operation{Q}()
end
