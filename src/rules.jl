import DiffRules: diffrule, diffrules

for i ∈ diffrules()

    pkg, f, arity = i

    # TODO: Rules for functions outside Base
    if pkg ≠ :Base continue end

    df(expr...) = diffrule(pkg, f, expr...)

    # Unitary functions
    if arity == 1
        quote
        end |> eval

    # Binary functions
    elseif arity == 2
        quote
        end |> eval
    else
        @warn "$i not handled"
    end
end
