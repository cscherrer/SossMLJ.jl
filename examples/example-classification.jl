using RDatasets
iris = dataset("datasets", "iris")

using Distributions
using Soss




"""
    smax!(r::AbstractArray, x::AbstractArray)

Overwrite `r` with the `smax` (or _normalized exponential_) transformation of `x`

That is, `r` is overwritten with `exp.(x)`, normalized to sum to 1.

See the [Wikipedia entry](https://en.wikipedia.org/wiki/smax_function)
"""
function smax!(r::AbstractArray, x::AbstractArray)
    n = length(x)
    length(r) == n || throw(DimensionMismatch("Inconsistent array lengths."))
    u = maximum(x)
    s = 0.
    @inbounds for i = 1:n
        s += (r[i] = exp(x[i] - u))
    end
    invs = inv(s)
    @inbounds for i = 1:n
        r[i] *= invs
    end
    r
end

"""
    smax(x::AbstractArray{<:Real})

Return the [`smax transformation`](https://en.wikipedia.org/wiki/smax_function) applied to `x`
"""
smax!(x::AbstractArray{<:AbstractFloat}) = smax!(x, x)
smax(x::AbstractArray{<:Real}) = smax!(similar(x), x)




m = @model X,num_levels begin
    k = size(X,2)
    β ~ Normal() |> iid(k,num_levels)
    p = smax.(eachrow(X*β))
    
    species ~ For(p) do pj
            Categorical(pj; check_args=false)
        end
end

X = randn(10,4);

rand(m(X=X,num_levels=3)) |> pairs

using SossMLJ

using MLJModelInterface: matrix

mdl = SossMLJModel(m;
    hyperparams = (num_levels=3,), 
    transform   = tbl -> (X=matrix(tbl[[:SepalWidth, :SepalLength, :PetalWidth, :PetalLength]]),),
    infer       = dynamicHMC,
    response    = :species
)


using MLJ
mach = machine(mdl, iris, getproperty.(iris.Species, :level))

fit!(mach)

jt = SossMLJ.predict_joint(mach, iris)

rand(jt)

predict_particles(jt, iris)

predict_mean(mach, X)
