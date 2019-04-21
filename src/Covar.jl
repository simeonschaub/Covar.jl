module Covar

export @covar_system,  val, var, err, numvals, CorrelatedVar, CovariantVar, DerivedVar,
       CovariantSystem

const tag_counters = UInt64[1]
newtag!() = @inbounds tag_counters[Base.Threads.threadid()] += 1

function __init__()
    nthr = Base.Threads.nthreads()
    resize!(tag_counters, nthr)[:] = range(UInt64(1), step=typemax(UInt64)÷nthr, length=nthr)
end

"""
Represents a system of covariant variables

    vals:   Values of the covariant variables
    covar:  The covariance matrix
"""
struct CovariantSystem{Tval,Tvals<:AbstractVector{Tval},Tcovar<:AbstractMatrix}
    vals::Tvals
    covar::Tcovar
    tag::UInt64
    
    function CovariantSystem(vals::Tvals, covar::Tcovar) where
        {Tval,Tvals<:AbstractVector{Tval},Tcovar<:AbstractMatrix}
        size(covar) == length.((vals, vals)) ||
            "CovariantSystem has to satisfy: vals ∈ ℝ^n, covar ∈ ℝ^(n×n) where n ∈ ℕ"
        return new{Tval,Tvals,Tcovar}(vals, covar, newtag!())
    end
end

abstract type Measurement{T} <: Number end
abstract type CorrelatedVar{T} <: Measurement{T} end

struct IndependentVar{Tval,Terr} <: Measurement{Tval}
    val::Tval
    err::Terr
    tag::UInt64

    function IndependentVar(val::Tval, err::Terr) where {Tval,Terr}
        return new{Tval,Terr}(val, err, newtag!())
    end
end

"""
Represents a variable of a CovariantSystem

    index:  Index of the variable in the CovariantSystem
    system: CovariantSystem of the variable
"""
struct CovariantVar{Tval,Tvals<:AbstractVector{Tval},Tcovar<:AbstractMatrix} <: CorrelatedVar{Tval}
    index::Int
    system::CovariantSystem{Tval,Tvals,Tcovar}

    function CovariantVar(index::Int, system::CovariantSystem{Tval,Tvals,Tcovar}) where
        {Tval,Tvals<:AbstractVector{Tval},Tcovar<:AbstractMatrix}
        index ≤ length(system.vals) || "Index of CovariantVar not in range"
        return new{Tval,Tvals,Tcovar}(index, system)
    end
end

val(x::CovariantVar) = system(x).vals[x.index]
var(x::CovariantVar) = system(x).covar[x.index, x.index]

"""
Creates a CovariantSystem and returns an Array containing the CovariantVars

See also: `CovariantSystem`, `CovariantVar`
"""
macro covar_system(vals, covar)
    return quote
        cs = CovariantSystem($(esc(vals)), $(esc(covar)))
        CovariantVar.(keys(cs.vals), Ref(cs))
    end
end

function Base.show(io::IO, x::CovariantVar)
    println(io, "x_$(x.index) = $(val(x)) ± $(err(x))")
    print(io, "  Vₘ = $(x.system.covar)")
end


#include("gradients.jl")

"""
Represents a variable derived from one or multiple CovariantSystems

    val:     Value of the Variable
    grads:   Gradients with respect to the corresponding CovariantSystems
    systems: The underlying CovariantSystems
"""
struct DerivedVar{isholomorphic,Tval,Targs<:Tuple,Trules<:Tuple} <: CorrelatedVar{Tval}
    val::Tval
    args::Targs
    rules::Trules
end

@generated function isholomorphic(::Targs, ::Trules) where {Targs,Trules}
    return !(any(T -> T<:DerivedVar{false}, Targs.parameters) ||
             any(T -> T<:WirtingerRule, Trules.parameters))
end

function DerivedVar(val::Tval, args::Targs, rules::Trules) where
    {Tval,Targs<:Tuple,Trules<:Tuple}

    return DerivedVar{isholomporphic(Targs,Trules),Tval,Targs,Trules}(val, args, rules)
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

include("accumulator.jl")

#include("rules.jl")

end
