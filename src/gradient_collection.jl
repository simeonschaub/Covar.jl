struct GradientCollection{T<:Tuple} <: AbstractDict{CovariantSystem, AbstractVector}
    dicts::T
    GradientCollection() = new{Tuple{}}(())
    GradientCollection(dicts::T) where T = new{T}(dicts)
end

GradientCollection(system, grad) = GradientCollection((Gradients(system, grad),))

function GradientCollection(system1::T, grad1, system2::T, grad2) where T<:CovariantSystem
    if system1 === system2
        return GradientCollection(system1, grad1 .+ grad2)
    else
        return GradientCollection((Gradients(Gradients(system1, grad1), system2, grad2),))
    end
end

function GradientCollection(system1::T, grad1, system2::U, grad2) where
    {T<:CovariantSystem, U<:CovariantSystem}
    return GradientCollection((Gradients(system1, grad1), Gradients(system2, grad2)))
end


Base.length(t::GradientCollection) = count(x->true, t)

Base.iterate(grads::T, t::Tuple{}) where T<:GradientCollection = nothing

function Base.iterate(grads::T, t=grads.dicts) where T<:GradientCollection
    next = iterate(first(t))
    if next === nothing
        return iterate(grads, Base.tail(t))
    else
        return next[1], (next[2], Base.tail(t)...)
    end
end

function linear_combine(df1, df2, gc1::T, gc2::U) where
    {T<:GradientCollection, U<:GradientCollection}
    return mergeo((df1, df2), gc1.dicts, gc2.dicts) |> GradientCollection
end

mul(c, g::T) where T<:GradientCollection = GradientCollection(mul.(c, g.dicts))



__merge((df1, df2), unused_ys, x_i) = mul(df1, x_i), unused_ys
function __merge(df, unused_ys, x_i::Gradients{T}, y_j::Gradients{T}, y...) where T
    return linear_combine(df..., x_i, y_j), (unused_ys..., y...)
end
__merge(df, unused_ys, x_i, y_j, y...) = __merge(df, (unused_ys..., y_j), x_i, y...)

_merge((df1, df2), results, remaining_ys) = (results..., mul.(df2, remaining_ys)...)
function _merge(df, results, remaining_ys, x_i, x...)
    _result, _remaining_ys = __merge(df, (), x_i, remaining_ys...)
    return _merge(df, (results..., _result), _remaining_ys, x...)
end

merge(df::Tuple{T,U}, x, y) where {T,U} = _merge(df, (), y, x...)



@inline _mergeo(f, results, remaining_ys) = (results, remaining_ys)
@inline function _mergeo(f, results, remaining_ys, x_i, x...)
    _result, _remaining_ys = __merge(f, (), x_i, remaining_ys...)
    return _mergeo(f, (results..., _result), _remaining_ys, x...)
end

@inline function mergeo((df1, df2), x::T, y) where T
    let (res::T, rem) = _mergeo((df1, df2), (), y, x...)
        return (res..., mul.(df2, rem)...)
    end
end
