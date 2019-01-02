import DiffRules: diffrule, diffrules

function wherein(x::T, a::AbstractArray{T}) where T
    for i in keys(a)
        if a[i] === x
            return i
        end
    end
    return nothing
end

#function merge_systems(newval, x::DerivedVar{T,<:Vector}, y::DerivedVar{U}, dg...) where {T,U}
#    
#    grads = x.grads .* dg[1]
#    from_y = trues(length(y.systems))
#
#    for i in keys(x.systems)
#        j = wherein(x.systems[i], y.systems)
#        if j ≠ nothing
#            @. grads[i] += y.grads[j] * dg[2]
#            from_y[j] = false
#        end
#    end
#
#    return DerivedVar(newval,
#                      vcat(grads, y.grads[from_y] .* dg[2]),
#                      vcat(x.systems, y.systems[from_y]))
#end
#
#function merge_systems(newval, x::DerivedVar{T}, y::DerivedVar{U}, dg...) where {T,U}
#    
#    grads = x.grads .* dg[1]
#    from_y = trues(length(y.systems))
#
#    for i in keys(x.systems)
#        j = wherein(x.systems[i], y.systems)
#        if j ≠ nothing
#            grads[i] += y.grads[j] * dg[2]
#            from_y[j] = false
#        end
#    end
#
#    return DerivedVar(newval,
#                      vcat(grads, y.grads[from_y] .* dg[2]),
#                      vcat(x.systems, y.systems[from_y]))
#end
#
#function merge_systems(newval, x::DerivedVar{T,<:SVector}, y::DerivedVar{U,<:SVector},
#                       dg...) where {T,U}
#    
#    grads = MVector.(x.grads .* dg[1])
#    from_y = trues(length(y.systems))
#
#    for i in keys(x.systems)
#        j = wherein(x.systems[i], y.systems)
#        if j ≠ nothing
#            @. grads[i] += y.grads[j] * dg[2]
#            from_y[j] = false
#        end
#    end
#
#    return DerivedVar(newval,
#                      vcat(SVector.(grads), y.grads[from_y] .* dg[2]),
#                      vcat(x.systems, y.systems[from_y]))
#end

@inline function merge_systems(newval, x::DerivedVar{T}, y::DerivedVar{U}, dg...) where {T,U}
    return DerivedVar(newval, linear_combine(dg..., x.grads, y.grads))
end


for i ∈ diffrules()

    pkg, f, arity = i

    # TODO: Rules for functions outside Base
    if pkg ≠ :Base continue end

    df(expr...) = diffrule(pkg, f, expr...)

    # Unitary functions
    if arity == 1
        quote
            function $pkg.$f(x::DerivedVar)
                return DerivedVar($f(val(x)), mul($(df(:(val(x)))), x.grads))
            end

            function $pkg.$f(x::CovariantVar{T,AV}) where {T,AV}
                return DerivedVar($f(val(x)), Gradients(x.system, onehot(AV, $(df(:(val(x)))),
                                                                         x.index, length(system(x).vals))))
            end
        end |> eval

    # Binary functions
    elseif arity == 2
        # Add explicit definitions for Integers to avoid ambiguities with Base
        for numtype in (:Real, :Integer)
            quote
                function $pkg.$f(x::DerivedVar, y::$numtype)
                    return DerivedVar($f(val(x), y),
                                      mul($(df(:(val(x)), :y)[1]), x.grads))
                end

                function $pkg.$f(x::$numtype, y::DerivedVar)
                    return DerivedVar($f(x, val(y)),
                                      mul($(df(:x, :(val(y)))[2]), y.grads))
                end

                $pkg.$f(x::CovariantVar, y::$numtype) = $f(DerivedVar(x), y)
                $pkg.$f(x::$numtype, y::CovariantVar) = $f(x, DerivedVar(y))
            end |> eval
        end

        quote
            # More of a hack, some ranges don't work otherwise
            $pkg.$f(x::CorrelatedVar, y::Base.TwicePrecision) = $f(x, convert(AbstractFloat, y))
            $pkg.$f(x::Base.TwicePrecision, y::CorrelatedVar) = $f(convert(AbstractFloat, x), y)

            function $pkg.$f(x::DerivedVar{T}, y::DerivedVar{U}) where {T,U}
                return merge_systems($f(val(x), val(y)), x, y,
                                     $(df(:(val(x)), :(val(y)))[1]),
                                     $(df(:(val(x)), :(val(y)))[2]))
                #return DerivedVar($f(val(x), val(y)), grads, systems)
            end

            $pkg.$f(x::CovariantVar, y::DerivedVar) = $f(DerivedVar(x), y)
            $pkg.$f(x::DerivedVar, y::CovariantVar) = $f(x, DerivedVar(y))

            function $pkg.$f(x::CovariantVar{T,AV1}, y::CovariantVar{U,AV2}) where {T,AV1,U,AV2}
                if system(x) === system(y)
                    return DerivedVar($f(val(x), val(y)),
                                      Derivatives(
                                        onehot(AV1, $(df(:(val(x)), :(val(y)))[1]),
                                               x.index, length(system(x).vals)) .+
                                        onehot(AV2, $(df(:(val(x)), :(val(y)))[2]),
                                               y.index, length(system(y).vals)),
                                        system(x)))
                else
                    return DerivedVar($f(val(x), val(y)), Derivatives(
                                    Derivatives(
                                        onehot(AV1, $(df(:(val(x)), :(val(y)))[1]),
                                               x.index, length(system(x).vals)),
                                        system(x)),
                                    onehot(AV2, $(df(:(val(x)), :(val(y)))[2]),
                                           y.index, length(system(y).vals)),
                                    system(y)))
                end
            end
        end |> eval
    else
        @warn "$i not handled"
    end
end

Base.sincos(x::CorrelatedVar) = (sin(x), cos(x))
