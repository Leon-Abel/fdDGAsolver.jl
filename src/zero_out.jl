# =============================================================================
# Symmetry groups that keep only complete symmetry classes (switched off).
#
# A symmetry group (MatsubaraFunctions.SymmetryGroup) sorts the points of a mesh
# function into classes of symmetry-related points and computes one value per
# class. A symmetry can map a point inside the frequency box to a point outside
# it. The idea here was to drop such incomplete classes and set their entries to
# zero. The scheme is switched off: my_SymmetryGroup returns the plain symmetry
# group and my_symmetrize! changes nothing. The solvers still build their
# symmetry groups through my_SymmetryGroup, so the file is kept.
# =============================================================================

"""
    class_inbounds(class, symmetries, f) -> Bool

Return `true` if no symmetry in `symmetries` maps a point of `class` outside the
mesh of `f`. Only used by the switched-off filter in
[`my_SymmetryGroup`](@ref).
"""
function class_inbounds(
    class      :: Vector{Tuple{Int, Operation}},
    symmetries :: Vector{Symmetry{DD}},
    f          :: MeshFunction{DD},
    )          :: Bool where {DD}
    # False if any symmetry in symmetries map any element in class outside the bounds of f.

    for (idx, op) in class
        w = value.(to_meshes(f, idx))

        for S in symmetries
            wp, _ = S(w)
            if ! MatsubaraFunctions._all_inbounds(f, wp...)
                return false
            end
        end
    end

    return true
end

"""
    my_SymmetryGroup(symmetries, f) -> SymmetryGroup

Symmetry group of the mesh function `f` for the given `symmetries` (e.g. those
in symmetries.jl). It was meant to drop every class with a point that a
symmetry maps outside the box (see [`class_inbounds`](@ref)). That filter is
switched off, so this is the plain `SymmetryGroup(symmetries, f)`.
"""
function my_SymmetryGroup(
    symmetries :: Vector{Symmetry{DD}},
    f          :: MeshFunction{DD}
    ) where {DD}
    # SymmetryGroup with elements that strictly preserves the symmetry.
    # Remove a class if any element maps outside the box by any symmetry.

    SG = SymmetryGroup(symmetries, f)

    # JML (Jae-Mo Lihm): We do not use the modified symmetrization scheme

    # filter!(class -> class_inbounds(class, symmetries, f), SG.classes)
    SG
end

"""
    my_symmetrize!(f, SG) -> err

Meant to symmetrize `f` with `SG` and set the entries outside `SG`'s classes to
zero. The body is switched off, so `f` is not changed and the returned error
is 0.
"""
function my_symmetrize!(
    f  :: MeshFunction{DD, Q},
    SG :: SymmetryGroup{DD, Q}
    ) where {DD, Q <: Number}

    # Symmetrize MeshFunction f, set elements not in SG.classes to zero.

    err = zero(real(Q))

    # JML (Jae-Mo Lihm): We do not use the modified symmetrization scheme

    # # Store the representative value of f for each class
    # ref_list = [f[class[1][1]] for class in SG.classes]
    # set!(f, 0)

    # for (class, ref) in zip(SG.classes, ref_list)
    #     for (idx, op) in class
    #         new_val = op(ref)
    #         new_err = abs(f[idx] - new_val)

    #         err = max(err, new_err)
    #         f[idx] = new_val
    #     end
    # end

    return err
end
