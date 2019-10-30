module SossMLJ

import MLJBase
import MLJBase: @mlj_model, metadata_pkg, metadata_model
import Soss
import Soss: @model, predictive, Univariate, Continuous, dynamicHMC
using Parameters
using LinearAlgebra
using Distributions

@with_kw_noshow mutable struct BayesianRidgeRegressor <: MLJBase.Probabilistic
    fit_intercept::Bool = true
end

struct BayesianRidgeRegressorPred <: Distributions.Sampleable{Univariate,Continuous}
    pred::Soss.Model
    Xrow::AbstractMatrix
    θ::Vector{NamedTuple}
#    b::Union{Nothing,NamedTuple}
end

function _model(brr::BayesianRidgeRegressor)
    if brr.fit_intercept
        mdl = @model X begin
            θ ~ Normal() |> iid(size(X, 2))
            yhat = X*θ
            y ~ For(length(yhat)) do j
                Normal(yhat[j],1)
            end
        end
    else
        mdl = @model X begin
            θ ~ Normal() |> iid(size(X, 2))
            yhat = X*θ
            y ~ For(eachindex(yhat)) do j
                Normal(yhat[j],1)
            end
        end
    end
    return mdl
end

function MLJBase.fit(brr::BayesianRidgeRegressor, verb::Int, X, y)
    # construct the model
    Xm = MLJBase.matrix(X)
    mdl = _model(brr)
    # fit the model
    res = dynamicHMC(mdl(X=Xm), (y=y,))
    cache = nothing
    report = NamedTuple{}()
    ((mdl, res), cache, report)
end

function MLJBase.predict(::BayesianRidgeRegressor, (mdl, res), Xnew)
    pred = predictive(mdl, setdiff(variables(mdl), [:X,  :y])...)
    map(eachrow(Xnew)) do x
        BayesianRidgeRegressorPred(pred, x', res)
    end
end

function Base.rand(p::BayesianRidgeRegressorPred)
    rand(p.pred(X=p.Xrow, Σ=1)).y
end

##########

X = randn(500, 10);
θ = randn(10);
y = X * θ + randn(500);

brr = BayesianRidgeRegressor(; fit_intercept=false)

Xt = MLJBase.table(X)

fr, = MLJBase.fit(brr, 1, Xt, y)

end # module
