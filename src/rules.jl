import DiffRules: diffrule, diffrules

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
                return DerivedVar($f(val(x)), GradientCollection(x.system, onehot(AV, $(df(:(val(x)))),
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
                return DerivedVar($f(val(x), val(y)),
                                  GradientCollection(system(x),
                                                     onehot(AV1, $(df(:(val(x)), :(val(y)))[1]),
                                                            x.index, length(system(x).vals)),
                                                     system(y),
                                                     onehot(AV2, $(df(:(val(x)), :(val(y)))[2]),
                                                            x.index, length(system(y).vals))
                                                    ))
            end
        end |> eval
    else
        @warn "$i not handled"
    end
end

Base.sincos(x::CorrelatedVar) = (sin(x), cos(x))
