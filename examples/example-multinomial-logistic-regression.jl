

using Distributions
using Soss
using MLJBase
using SossMLJ



using RDatasets
iris = dataset("datasets", "iris")

function softmax!(r::AbstractArray, x::AbstractArray)
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

softmax!(x::AbstractArray{<:AbstractFloat}) = softmax!(x, x)
softmax(x::AbstractArray{<:Real}) = softmax!(similar(x), x)

Distributions.logpdf(d::UnivariateFinite, y::CategoricalValue)  = log(pdf(d,y))

###################################


m = @model X,pool begin
    n = size(X,1)
    k = size(X,2)
    num_levels = length(pool.levels)
    β ~ Normal() |> iid(k,num_levels)

    η = X * β
    μ = mapslices(softmax, η; dims=2)

    ydists = UnivariateFinite(pool.levels, μ; pool=pool)

    y ~ For(j -> ydists[j], n)
end;

mdl = SossMLJModel(m;
    hyperparams = (pool=iris.Species.pool,),
    transform   = tbl -> (X=MLJBase.matrix(tbl[[:SepalWidth, :SepalLength, :PetalWidth, :PetalLength]]),),
    infer       = dynamicHMC,
    response    = :y
)


using MLJ
mach = machine(mdl, iris, iris.Species)

fit!(mach)

jt = predict_joint(mach, iris)

rand(jt)

# predict_particles(jt, iris)

# predict_mean(mach, X)
