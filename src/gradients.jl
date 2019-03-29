mutable struct Gradients{K,V} <: AbstractDict{K,V}
    parent::Gradients{K,V}
    key::K
    val::V

    Gradients{K,V}() where {K,V} = new()
    Gradients(key::K, val::V) where {K,V} = new{K,V}(Gradients{K,V}(), key, val)
    function Gradients(parent::Gradients{K,V}, key::K, val::V) where {K,V}
        return new{K,V}(parent, key, val)
    end
end

function Gradients(parent::Gradients{K1,V1}, key::K2, val::V2) where {K1,V1,K2,V2}
    K = promote_type(K1, K2)
    V = promote_type(V1, V2)
    #println("$K2 => $K: $(convert(K, key));  $V2 => $V: $(convert(V, val))")
    #println("Gradients{$K,$V}: $(convert(Gradients{K,V}, parent))")
    return Gradients(convert(Gradients{K,V}, parent), convert(K, key), convert(V, val))
end

Base.length(t::Gradients) = count(x->true, t)

function Base.iterate(grads::Gradients{K,V}, t=grads) where {K,V}
    isdefined(t, :parent) || return nothing
    return Pair{K,V}(t.key, t.val), t.parent
end

similar(::Gradients{K,V}) where {K,V} = Gradients{K,V}()

function Base.map(f, g::Gradients; keyfunc=identity)
    if isdefined(g, :parent)
        return Gradients(map(f, g.parent; keyfunc=keyfunc), keyfunc(g.key), f(g.val))
    else
        return g
    end
end

function mul(c, g::Gradients)
    return map(dg -> c .* dg, g)
end

function Base.convert(::Type{Gradients{K,V}}, g::Gradients{_K,_V}) where {K,V,_K,_V}
    @info "`convert` is called!"
    if !isdefined(g, :parent)
        return Gradients{K,V}()
    end
    return map(val -> convert(V, val), g; keyfunc = key -> convert(K, key))
end

function Base.convert(::Type{Gradients{K,V}}, g::Gradients{K,V}) where {K,V}
    return g
end

Base.promote_rule(::Type{Gradients{K1,V1}}, ::Type{Gradients{K2,V2}}) where {K1,V1,K2,V2} =
    Gradients{promote_type(K1, K2),promote_type(V1, V2)}


function linear_combine(∂f_∂g1, ∂f_∂g2, g1::Gradients, g2::Gradients)
    if !isdefined(g2, :parent)
        return mul(∂f_∂g1, g1)
    end
    return _linear_combine!(∂f_∂g1, g1, mul(∂f_∂g2, g2))
end

#modifies g2
function _linear_combine!(∂f_∂g1::T, g1::G, g2::G) where {T,U,G<:Gradients{<:CovariantSystem{U},<:AbstractVector{T}}}
    if !isdefined(g2, :parent)
        return mul(∂f_∂g1, g1)
    end
    if isdefined(g1, :parent)
        if g2.key === g1.key
            return Gradients(_linear_combine!(∂f_∂g1, g1.parent, g2.parent),
                             g1.key, ∂f_∂g1 .* g1.val .+ g2.val)
        end
        g = g2
        while isdefined(g.parent, :parent)
            if g.parent.key === g1.key
                newval = ∂f_∂g1 .* g1.val .+ g.parent.val
                g.parent = g.parent.parent
                return Gradients(_linear_combine!(∂f_∂g1, g1.parent, g2),
                                 g1.key, newval)
            end
            g = g.parent
        end
        return Gradients(_linear_combine!(∂f_∂g1, g1.parent, g2),
                             g1.key, ∂f_∂g1 .* g1.val)

    else
        return g2
    end
end

function _linear_combine!(∂f_∂g1::T0, g1::G1, g2::G2) where
    {T0,T1,U1,G1<:Gradients{<:CovariantSystem{U1},<:AbstractVector{T1}},
     T2,U2,G2<:Gradients{<:CovariantSystem{U2},<:AbstractVector{T2}}}

    T = promote_type(T0, T1, T2)
    return _linear_combine!(convert(T, ∂f_∂g1), promote(g1, g2)...)
end
