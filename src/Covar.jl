module Covar

export @covar_system, ⟂, val, var, err, numvals, CorrelatedVar, CovariantVar, DerivedVar,
       CovariantSystem

using StaticArrays

"""
Represents a system of covariant variables

    vals:   Values of the covariant variables
    covar:  The covariance matrix
"""
struct CovariantSystem{T<:AbstractFloat,AV<:AbstractVector{T},AM<:AbstractMatrix{T}}
    vals::AV
    covar::AM
    
    function CovariantSystem(vals::AV, covar::AM) where
        {T<:AbstractFloat,AV<:AbstractVector{T},AM<:AbstractMatrix{T}}
        @assert(size(covar) == length.((vals, vals)),
                "CovariantSystem has to satisfy: vals ∈ ℝ^n, covar ∈ ℝ^(n×n) where n ∈ ℕ")
        return new{T,AV,AM}(vals, covar)
    end
end

numvals(x::CovariantSystem) = length(x.vals)

function Base.convert(::Type{CovariantSystem{_,AV,AM}}, cs) where {_,AV,AM}
    return CovariantSystem(convert(AV, cs.vals), convert(AM, cs.covar))
end

function Base.convert(::Type{CovariantSystem{T,AV,AM}},
                      cs::CovariantSystem{T,AV,AM}) where {T,AV,AM}
    return cs
end

function Base.promote_rule(::Type{CovariantSystem{T1,AV1,AM1}}, ::Type{CovariantSystem{T2,AV2,AM2}}) where
    {T1,AV1,AM1,T2,AV2,AM2}
    return CovariantSystem{promote_type(T1, T2),promote_type(AV1, AV2),promote_type(AM1, AM2)}
end


abstract type CorrelatedVar <: Number end

"""
Represents a variable of a CovariantSystem

    index:  Index of the variable in the CovariantSystem
    system: CovariantSystem of the variable
"""
struct CovariantVar{T<:AbstractFloat,AV<:AbstractVector{T},AM<:AbstractMatrix{T}} <: CorrelatedVar
    index::Int
    system::CovariantSystem{T,AV,AM}

    function CovariantVar(index::Int, system::CovariantSystem{T,AV,AM}) where
        {T<:AbstractFloat,AV<:AbstractVector{T},AM<:AbstractMatrix{T}}
        @assert index ≤ length(system.vals) "Index of CovariantVar not in range"
        return new{T,AV,AM}(index, system)
    end
end

system(x::CovariantVar) = x.system
val(x::CovariantVar) = system(x).vals[x.index]
var(x::CovariantVar) = system(x).covar[x.index, x.index]

⟂(x::CovariantVar, y::CovariantVar) = x.system !== y.system

"""
Creates a CovariantSystem and returns an Array containing the CovariantVars

See also: `CovariantSystem`, `CovariantVar`
"""
macro covar_system(vals, covar)
    return quote
        begin
            cs = CovariantSystem($(esc(vals)), $(esc(covar)))
            CovariantVar.(keys(cs.vals), Ref(cs))
        end
    end
end

function Base.show(io::IO, x::CovariantVar)
    println(io, "x_$(x.index) = $(val(x)) ± $(err(x))")
    print(io, "  Vₘ = $(x.system.covar)")
end


include("gradients.jl")

"""
Represents a variable derived from one or multiple CovariantSystems

    val:     Value of the Variable
    grads:   Gradients with respect to the corresponding CovariantSystems
    systems: The underlying CovariantSystems
"""
struct DerivedVar{T<:AbstractFloat,G<:Gradients} <: CorrelatedVar
    val::T
    grads::G
end

val(x::DerivedVar) = x.val

function var(x::DerivedVar{T}) where T
    var = zero(T)
    for (system,grad) in x.grads
        var += grad' * (system.covar * grad)
    end
    return var
end

err(x::CorrelatedVar) = √var(x)

function Base.show(io::IO, x::DerivedVar)
    print(io, "f({xᵢ}) = $(val(x)) ± $(err(x))")
end



function onehot(::Type{AV}, val, i::Int, len::Int) where
    {T<:AbstractFloat,AV<:AbstractVector{T}}
    a = zeros(T, len)
    @inbounds a[i] = val
    return a
end

function onehot(SV::Type{SVector{N,T}}, val, i::Int, len::Int) where {N,T<:AbstractFloat}
    a = zeros(MVector{N,T})
    @inbounds a[i] = val
    return SVector(a)
end

function onehot(SV::Type{SVector{1,T}}, val, i::Int, len::Int) where {T<:AbstractFloat}
    return SV(val)
end

function DerivedVar(x::CovariantVar{T,AV}) where {T,AV}
    return DerivedVar(val(x), Gradients(system(x), onehot(AV, one(T), x.index, numvals(system(x)))))
end

include("rules.jl")

end
