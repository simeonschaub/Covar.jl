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

Base.length(t::Gradients) = count(x->true, t)

function Base.iterate(grads::Gradients{K,V}, t=grads) where {K,V}
    isdefined(t, :parent) || return nothing
    return Pair{K,V}(t.key, t.val), t.parent
end

similar(::Gradients{K,V}) where {K,V} = Gradients{K,V}()

function mul(c, g::Gradients)
    if isdefined(g, :parent)
        return Gradients(mul(c, g.parent), g.key, c .* g.val)
    else
        return g#Gradients(g.key, c .* g.val)
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
        return g2# |> similar#Gradients(g2, g1.key, ∂f_∂g1 .* g1.val)
    end
end
