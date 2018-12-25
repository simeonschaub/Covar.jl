import DiffRules: diffrule, diffrules

function wherein(x::T, a::AbstractArray{T}) where T
    for i in keys(a)
        if a[i] === x
            return i
        end
    end
    return nothing
end

function merge_systems(x::DerivedVar{T}, y::DerivedVar{U}, dg...) where {T,U}
    
    grads = x.grads .* dg[1]
    from_y = trues(length(y.systems))

    for i in keys(x.systems)
        j = wherein(x.systems[i], y.systems)
        if j ≠ nothing
            @. grads[i] += y.grads[j] * dg[2]
            from_y[j] = false
        end
    end

    return vcat(grads, [ grad .* dg[2] for grad in y.grads[from_y] ]),
           vcat(x.systems, y.systems[from_y])
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
                return DerivedVar($f(val(x)), [ $(df(:(val(x)))) .* grad for grad in x.grads ], x.systems)
            end

            function $pkg.$f(x::CovariantVar{T}) where T
                return DerivedVar($f(val(x)),
                                  [onehot(convert(T, $(df(:(val(x))))),
                                          x.index, length(system(x).vals))],
                                  [x.system])
            end
        end |> eval

    # Binary functions
    elseif arity == 2
        # Add explicit definitions for Integers to avoid ambiguities with Base
        for numtype in (:Real, :Integer)
            quote
                function $pkg.$f(x::DerivedVar, y::$numtype)
                    return DerivedVar($f(val(x), y),
                                      [ $(df(:(val(x)), :y)[1]) .* grad for grad in x.grads ],
                                      x.systems)
                end

                function $pkg.$f(x::$numtype, y::DerivedVar)
                    return DerivedVar($f(x, val(y)),
                                      [ $(df(:x, :(val(y)))[2]) .* grad for grad in y.grads ],
                                      y.systems)
                end

                $pkg.$f(x::CovariantVar, y::$numtype) = $f(DerivedVar(x), y)
                $pkg.$f(x::$numtype, y::CovariantVar) = $f(x, DerivedVar(y))
            end |> eval
        end

        quote
            function $pkg.$f(x::DerivedVar{T}, y::DerivedVar{U}) where {T,U}
                grads, systems = merge_systems(x, y,
                                               $(df(:(val(x)), :(val(y)))[1]),
                                               $(df(:(val(x)), :(val(y)))[2]))
                return DerivedVar($f(val(x), val(y)), grads, systems)
            end

            $pkg.$f(x::CovariantVar, y::DerivedVar) = $f(DerivedVar(x), y)
            $pkg.$f(x::DerivedVar, y::CovariantVar) = $f(x, DerivedVar(y))

            function $pkg.$f(x::CovariantVar{T}, y::CovariantVar{U}) where {T,U}
                if system(x) === system(y)
                    return DerivedVar($f(val(x), val(y)),
                                      [ onehot(convert(T, $(df(:(val(x)), :(val(y)))[1])),
                                               x.index, length(system(x).vals)) .+
                                        onehot(convert(T, $(df(:(val(x)), :(val(y)))[2])),
                                               y.index, length(system(y).vals)) ],
                                      [system(x)])
                else
                    V = promote_type(T, U)
                    return DerivedVar($f(val(x), val(y)),
                                      [ onehot(convert(V, $(df(:(val(x)), :(val(y)))[1])),
                                               x.index, length(system(x).vals)),
                                        onehot(convert(V, $(df(:(val(x)), :(val(y)))[2])),
                                               y.index, length(system(y).vals)) ],
                                      system.([x, y]))
                end
            end
        end |> eval
    else
        @warn "$i not handled"
    end
end

Base.sincos(x::CorrelatedVar) = (sin(x), cos(x))
