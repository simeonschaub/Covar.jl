mutable struct Gradients{CS<:CovariantSystem,V} <: AbstractDict{CS,V}
    parent::Gradients{CS,V}
    key::CS
    val::V

    Gradients{CS,V}() where {CS<:CovariantSystem,V} = new()
    Gradients(key::CS, val::V) where {CS<:CovariantSystem,V} = new{CS,V}(Gradients{CS,V}(), key, val)
    function Gradients(parent::Gradients{CS,V}, key::CS, val::V) where {CS<:CovariantSystem,V}
        return new{CS,V}(parent, key, val)
    end
end

@generated function Gradients(parent::Gradients{CS,W}, key::CS, val::V) where {CS<:CovariantSystem,V,W}
    valtype = promote_type(V, W)
    newval = convert(valtype, val)
    newparent = convert(Gradients{CS,valtype}, parent)
    return Gradients(newparent, key, newval)
end

#function Gradients(parent::Gradients{<:CovariantSystem,W}, key::CS, val::V) where {CS<:CovariantSystem,V,W}
#    return Gradients(parent, key::CovariantSystem, val)
#end

Base.length(t::Gradients) = count(x->true, t)

function Base.iterate(grads::Gradients{CS,V}, t=grads) where {CS<:CovariantSystem,V}
    isdefined(t, :parent) || return nothing
    return Pair{CS,V}(t.key, t.val), t.parent
end

similar(::Gradients{CS,V}) where {CS,V} = Gradients{CS,V}()

function Base.convert(GT::Type{Gradients{CS,T}}, g::Gradients{CS,U}) where {CS,T,U}
    if isdefined(g, :parent)
        return Gradients(convert(GT, g.parent), g.key, convert(T, g.val))
    else
        return g
    end
end

@inline Base.convert(::Type{Gradients{CS,T}}, g::Gradients{CS,T}) where {CS,T} = g

function promote_rule(::Type{Gradients{CS,T}}, ::Type{Gradients{CS,U}}) where {CS,T,U}
    return Gradients{CS,promote_type(T,U)}
end


function mul(c, g::Gradients)
    if isdefined(g, :parent)
        return Gradients(mul(c, g.parent), g.key, c .* g.val)
    else
        return g
    end
end

function linear_combine(∂f_∂g1, ∂f_∂g2, g1::Gradients, g2::Gradients)
    if !isdefined(g2, :parent)
        return mul(∂f_∂g1, g1)
    end
    return linear_combine!(∂f_∂g1, g1, mul(∂f_∂g2, g2))
end

#modifies g2
function linear_combine!(∂f_∂g1, g1::Gradients, g2::Gradients)
    if !isdefined(g2, :parent)
        return mul(∂f_∂g1, g1)
    end
    if isdefined(g1, :parent)
        if g2.key === g1.key
            return Gradients(linear_combine!(∂f_∂g1, g1.parent, g2.parent),
                             g1.key, ∂f_∂g1 .* g1.val .+ g2.val)
        end
        g = g2
        while isdefined(g.parent, :parent)
            if g.parent.key === g1.key
                newval = ∂f_∂g1 .* g1.val .+ g.parent.val
                g.parent = g.parent.parent
                return Gradients(linear_combine!(∂f_∂g1, g1.parent, g2),
                                 g1.key, newval)
            end
            g = g.parent
        end
        return Gradients(linear_combine!(∂f_∂g1, g1.parent, g2),
                             g1.key, ∂f_∂g1 .* g1.val)

    else
        return g2
    end
end
