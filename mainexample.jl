using MLJ
using MLJModelInterface
using Soss
using SossMLJ

const MMI = MLJModelInterface

m = @model X,α,σ begin
    k = size(X,2)
    β ~ Normal(0,α) |> iid(k)
    yhat = X * β
    y ~ For(eachindex(yhat)) do j
        Normal(yhat[j], σ)
    end
end

X = (a=randn(10), b=randn(10), c=randn(10))

params = (α=2.0, σ=1.0)

mdl = SossMLJ.SossMLJModel(params, X -> (X=MMI.matrix(X),), m, dynamicHMC)

args = merge(mdl.transform(X), params)

truth = rand(m(args))
y = truth.y

mach = machine(mdl, X, y)
fit!(mach)

predictor_marginal = MLJ.predict(mach, X)
rand.(predictor_marginal)

# predictor_joint = MLJ.predict_joint(mach, X)
predictor_joint = SossMLJ.predict_joint(mach, X)
rand(predictor_joint)

logpdf(predictor_joint, rand(predictor_joint))
