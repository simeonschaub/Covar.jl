# Covar.jl

## Introduction
`Covar.jl` allows for the representation of correlated variables with uncertainties,
which are described by their
[covariance matrices](https://en.wikipedia.org/wiki/Covariance_matrix).
It allows for easy propagation of these uncertainties, thanks to `Julia`'s convenient
method overloading, even for different covariant systems.

## Usage
The easiest way to construct a covariant system is with the `@covar_system` macro.
The following creates a `CovariantSystem` with values `[2., 3.]` and covariance
matrix `[.3 .1; .1 .4]` and assignes `a` and `b` their corresponding `CovariantVar`:
```julia
julia> a, b = @covar_system [2., 3.] [.3 .1; .1 .4]
2-element Array{CovariantVar{Float64},1}:
 x_1 = 2.0 ± 0.5477225575051661
  Vₘ = [0.3 0.1; 0.1 0.4]
 x_2 = 3.0 ± 0.6324555320336759
  Vₘ = [0.3 0.1; 0.1 0.4]
```

All basic operations on these variables are supported:
```julia
julia> a^2 + 2b + 1
f({xᵢ}) = 11.0 ± 2.8284271247461903
  {∇⃗f} = Array{Float64,1}[[4.0, 2.0]]
  {Vₘ} = Array{Float64,2}[[0.3 0.1; 0.1 0.4]]
```

Operations on `CovariantVars` automatically generate a `DerivedVar`, which contains
the corresponding value, the gradients with respect to their systems and the
`CovariantSystems` themselves. The gradients propagate according to the chain rule:

<img src="https://latex.codecogs.com/svg.latex?\operatorname{grad}&space;f&space;=&space;\frac{\partial&space;f}{\partial&space;g}&space;\cdot&space;\frac{\partial&space;g}{\partial&space;x_i}&space;\hat&space;e_i" title="\operatorname{grad} f = \frac{\partial f}{\partial g} \cdot \frac{\partial g}{\partial x_i} \hat e_i" />

The error is then calculated with the following formula:

<img src="https://latex.codecogs.com/svg.latex?\sigma_f&space;=&space;\sqrt{\operatorname{grad}&space;f\,^T&space;\cdot&space;V_m&space;\cdot&space;\operatorname{grad}&space;f}" title="\sigma_f = \sqrt{\operatorname{grad} f\,^T \cdot V_m \cdot \operatorname{grad} f}" />

`Covar.jl` also supports operations between multiple different `CovariantSystems`. The
variables in one system are then treated as independent from the ones from other systems:

```julia
julia> c, d, e = @covar_system [5., 2., 3.] [1.2 .3 .1; .3 .5 .2; .1 .2 .6]
3-element Array{CovariantVar{Float64},1}:
 x_1 = 5.0 ± 1.0954451150103321
  Vₘ = [1.2 0.3 0.1; 0.3 0.5 0.2; 0.1 0.2 0.6]
 x_2 = 2.0 ± 0.7071067811865476
  Vₘ = [1.2 0.3 0.1; 0.3 0.5 0.2; 0.1 0.2 0.6]
 x_3 = 3.0 ± 0.7745966692414834
  Vₘ = [1.2 0.3 0.1; 0.3 0.5 0.2; 0.1 0.2 0.6]

julia> a * c - sin(d + b) * e
f({xᵢ}) = 12.876772823989416 ± 2.1793998471976033
  {∇⃗f} = Array{Float64,1}[[5.0, -0.850987], [2.0, -0.850987, 0.958924]]
  {Vₘ} = Array{Float64,2}[[0.3 0.1; 0.1 0.4], [1.2 0.3 0.1; 0.3 0.5 0.2; 0.1 0.2 0.6]]
```

The following also works like you would expect:
```julia
julia> sin(a * c)^2 + cos(c * a)^2
f({xᵢ}) = 1.0 ± 0.0
  {∇⃗f} = Array{Float64,1}[[0.0, 0.0], [0.0, 0.0, 0.0]]
  {Vₘ} = Array{Float64,2}[[0.3 0.1; 0.1 0.4], [1.2 0.3 0.1; 0.3 0.5 0.2; 0.1 0.2 0.6]]
```

## TODO
* Write Tests
* Add support for SpecialFunctions and NaNMath via `Require.jl`
* Interoperability with `Measurements.jl`?
