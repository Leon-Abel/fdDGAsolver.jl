# BSEa_K1.jl
#
# Implements the Bethe-Salpeter equation (BSE) for the K1 vertex class of the
# parquet decomposition. K1 vertices depend only on the bosonic transfer frequency ω (Ω in code)
# and are independent of the two fermionic frequencies ν, ν'. As a consequence,
# their left- and right-insertion contributions to the BSE always vanish and only
# the one-loop (1ℓ) and central part remain.
# 
# The integration frequency is called ω in the code. 

"""
    BSE_K1!(K1, F0, F, FL, Π0, Π, SG, sign, ::Type{Ch}, ::Type{Sp}, is_mfRG; mode)

Solves the Bethe-Salpeter equation 

    `F_012‾r ( Π~_r + Π_r I~_r π_r) f_012r`

for the K1 vertex in-place. 
    
Compared to the notation in the paper, the reference and target functions have been interchanged.
"""
function BSE_K1!(
    K1   :: MF_K1{Q},
    F0   :: Union{RefVertex{Q}, AbstractVertex{Q}},
    F    :: AbstractVertex{Q},
    FL   :: AbstractVertex{Q},
    Π0   :: MF_Π{Q},
    Π    :: MF_Π{Q},
    SG   :: SymmetryGroup{1, Q},
    sign :: Int,
         :: Type{Ch},
         :: Type{Sp},
    is_mfRG :: Union{Val{true}, Val{false}} = Val(false),
    ;
    mode :: Symbol = :serial
    ) where {Q, Ch <: ChannelTag, Sp <: SpinTag}

    # model the diagram
    @inline function diagram(wtpl)

        Ω       = wtpl[1]
        val     = zero(Q)
        Π0slice = view(Π0, Ω, :)
        Πslice  = view(Π, Ω, :)

        for i in eachindex(Π0slice)
            ω = value(meshes(Π0, Val(2))[i])

            if is_mfRG == Val(true)
                # vertices
                F0l = F0(Ω, νInf, ω, aCh, pSp)
                FLr = FL(Ω, _crossing(Ω, ω, Ch), νInf, aCh, pSp)

                # central part
                val += F0l * Π0slice[i] * FLr
            else
                # vertices
                # Fl = F_012‾r = F_r(ω, ν → ∞, ν')
                Fl  = F( Ω, νInf, ω, Ch, Sp)
                # F0r = f_012r = f_r(ω, ν, ν' → ∞)
                F0r = F0(Ω, _crossing(Ω, ω, Ch), νInf, Ch, Sp)
                # FLr = Φ~^(L)r = I~_r π_r f_012r = I~_r π_r f_r(ω, ν, ν' → ∞)
                FLr = FL(Ω, _crossing(Ω, ω, Ch), νInf, Ch, Sp)

                # 1ℓ and central part
                val += Fl * ((Πslice[i] - Π0slice[i]) * F0r + Πslice[i] * FLr)
            end
        end

        return temperature(F) * val * sign
    end

    # compute K1
    SG(K1, InitFunction{1, Q}(diagram); mode)

    return nothing
end


"""
    BSE_K1_new!(K1, F0, F, Π0, Π, SG, sign, ::Type{Ch}, ::Type{Sp}, is_mfRG; mode)

Alternative BSE for K1 that right-inserts the bare vertex `U` extracted from `F`.

Accumulates `(F · Π - F0 · Π0) · U` over the internal fermionic frequency loop,
expressing the dressed-minus-bare difference on the left and closing with the
bare interaction on the right.
"""
function BSE_K1_new!(
    K1   :: MF_K1{Q},
    F0   :: Union{RefVertex{Q}, AbstractVertex{Q}},
    F    :: AbstractVertex{Q},
    Π0   :: MF_Π{Q},
    Π    :: MF_Π{Q},
    SG   :: SymmetryGroup{1, Q},
    sign :: Int,
         :: Type{Ch},
         :: Type{Sp},
    is_mfRG :: Union{Val{true}, Val{false}} = Val(false),
    ;
    mode :: Symbol = :serial
    ) where {Q, Ch <: ChannelTag, Sp <: SpinTag}

    U = bare_vertex(F, Sp)

    # model the diagram
    @inline function diagram(wtpl)

        Ω       = wtpl[1]
        val     = zero(Q)
        Π0slice = view(Π0, Ω, :)
        Πslice  = view(Π,  Ω, :)

        for i in eachindex(Π0slice)
            ω = value(meshes(Π0, Val(2))[i])

            # vertices
            Fl  = F( Ω, νInf, ω, Ch, Sp)
            F0l = F0(Ω, νInf, ω, Ch, Sp)

            # 1ℓ and central part
            val += (Fl * Πslice[i] - F0l * Π0slice[i]) * U
        end

        return temperature(F) * val * sign
    end

    # compute K1
    SG(K1, InitFunction{1, Q}(diagram); mode)

    return nothing
end
