module Covar

export @covar_system, ⟂, val, var, err, numvals, CorrelatedVar, CovariantVar, DerivedVar,
       CovariantSystem

"""
Represents a system of covariant variables

    vals:   Values of the covariant variables
    covar:  The covariance matrix
"""
struct CovariantSystem{T<:AbstractFloat}
    vals::AbstractVector{T}
    covar::AbstractMatrix{T}
    
    function CovariantSystem(vals::AbstractVector{T}, covar::AbstractMatrix{T}) where T<:AbstractFloat
        @assert(size(covar) == length.((vals, vals)),
                "CovariantSystem has to satisfy: vals ∈ ℝ^n, covar ∈ ℝ^(n×n) where n ∈ ℕ")
        return new{T}(vals, covar)
    end
end

numvals(x::CovariantSystem) = length(x.vals)

abstract type CorrelatedVar <: Number end

"""
Represents a variable of a CovariantSystem

    index:  Index of the variable in the CovariantSystem
    system: CovariantSystem of the variable
"""
struct CovariantVar{T<:AbstractFloat} <: CorrelatedVar
    index::Int
    system::CovariantSystem{T}

    function CovariantVar(index::Int, system::CovariantSystem{T}) where T<:AbstractFloat
        @assert index ≤ length(system.vals) "Index of CovariantVar not in range"
        return new{T}(index, system)
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


"""
Represents a variable derived from one or multiple CovariantSystems

    val:     Value of the Variable
    grads:   Gradients with respect to the corresponding CovariantSystems
    systems: The underlying CovariantSystems
"""
struct DerivedVar{T<:AbstractFloat} <: CorrelatedVar
    val::T
    grads::Vector{Vector{T}}
    systems::Vector{CovariantSystem}

    function DerivedVar(val::T, grads::Vector{Vector{T}},
                        systems::Vector{CovariantSystem}) where T<:AbstractFloat
        @assert length(grads) == length(systems) "grads and systems have to be the same length"
        return new{T}(val, grads, systems)
    end

    function DerivedVar(val::T, grads::Vector{Vector{T}},
                        systems::Vector{CovariantSystem{U}}) where {T<:AbstractFloat, U<:AbstractFloat}
        @assert length(grads) == length(systems) "grads and systems have to be the same length"
        return new{T}(val, grads, systems)
    end
end

val(x::DerivedVar) = x.val

function var(x::DerivedVar{T}) where T
    var = zero(T)
    for i in length(x.systems)
        var += x.grads[i]' * (x.systems[i].covar * x.grads[i])
    end
    return var
end

err(x::CorrelatedVar) = √var(x)

function Base.show(io::IO, x::DerivedVar)
    println(io, "f({xᵢ}) = $(val(x)) ± $(err(x))")
    println(io, "  {∇⃗f} = $(x.grads)")
    print(io, "  {Vₘ} = $([ i.covar for i in x.systems ])")
end



function onehot(val::T, i::Int, len::Int...) where T <: AbstractFloat
    a = zeros(T, len...)
    @inbounds a[i] = val
    return a
end

#onehot(val, i::Int, len::Int...) = onehot(float(val), i, len...)

function DerivedVar(x::CovariantVar{T}) where T
    return DerivedVar(val(x),
                      [onehot(one(T), x.index, numvals(system(x)))],
                      [system(x)])
end

include("rules.jl")

end # module
