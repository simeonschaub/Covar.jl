struct Accumulator{T} <: AbstractDict{Union{CovariantVar,IndependentVar},T}
    covar_systems::IdDict{CovariantSystem,Dict{Int,T}}
    independent_vars::IdDict{IndependentVar,T}
end

function Accumulator(covar_systems::IdDict{<:CovariantSystem,Dict{Int,T}},
                     independent_vars::IdDict{<:IndependentVar,T}) where T
    return Accumulator(IdDict{CovariantSystem,Dict{Int,T}}(covar_systems),
                       IdDict{IndependentVar,T}(independent_vars))
end

function Base.iterate(a::Accumulator, state=(:covar_systems,))
    iter = iterate(getproperty(a, first(state)), Base.tail(state)...)
    if iter == nothing
        first(state) == :covar_systems && return iterate(a, (:independent_vars,))
        return nothing
    end
    return first(iter), (first(state), Base.tail(iter)...)
end

Base.length(a::Accumulator) = length(a.covar_systems) + length(a.independent_vars)

function materialize!(a::Accumulator{T}, key::CovariantVar, rule, args...) where T
    d = get(a.covar_systems, key.system, Dict{Int,T})
    d[key.index] = materialize(get(d, key.index, Zero()), rule, args...)
end

function materialize!(a::Accumulator{T}, key::IndependentVar, rule, args...) where T
    d = a.independent_vars
    d[key] = materialize(get(d, key.index, Zero()), rule, args...)
end
