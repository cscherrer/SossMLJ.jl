__precompile__(false)

module SossMLJ

using MLJModelInterface
const MMI = MLJModelInterface

using Soss
import Soss: Model, @model, predictive, Univariate, Continuous, dynamicHMC
import Distributions
import Base
import Tables

const Dists = Distributions




# Just a placeholder
m0 = @model X,α,σ begin
    k = size(X,2)
    β ~ Normal(0,α) |> iid(k)
    yhat = X * β
    y ~ For(eachindex(yhat)) do j
        Normal(yhat[j], σ)
    end
end

# TODO: Add an way to specify which variable is observed, and which is predicted
# e.g. even if we observe `y`, we might sometimes prefer to predict yhat

# @mlj_model mutable struct SossMLJModel <: MMI.JointProbabilistic
#     model :: Model = m0
# end


export SossMLJModel

# TODO: Should the user instead specify a `prior` and `likelihood`?

# mutable struct SossMLJModel{H,T,M,I} <: MLJModelInterface.JointProbabilistic
mutable struct SossMLJModel{H,T,M,I} <: MLJModelInterface.Probabilistic
    hyperparams :: H
    transform :: T
    model :: M
    infer :: I
end

function SossMLJModel(; model = m0, infer=dynamicHMC)
    smm = SossMLJModel(hyperparams, transform, model, infer)
    message = MLJModelInterface.clean!(smm)
    return smm
end


function MLJModelInterface.clean!(smm::SossMLJModel)
    warning = ""
    return warning
end

struct MixedVariate <: Dists.VariateForm end
struct MixedSupport <: Dists.ValueSupport end

struct SossMLJPredictor{M} <: Distributions.Sampleable{MixedVariate, MixedSupport}
    model :: M
    post 
    pred
    args
end


function Base.rand(sp::SossMLJPredictor{M}) where {M}
    pars = rand(sp.post)
    args = merge(sp.args, pars)
    return rand(sp.pred(args)).y[1]
end

function MMI.fit(sm::SossMLJModel, verbosity::Int, X, y, w=nothing)
    # construct the model
    args = merge(sm.transform(X), sm.hyperparams)

    jd = sm.model(args)

    post = sm.infer(jd, (y=y,))

    # TODO: Allow w to be included

    cache = nothing
    report = NamedTuple{}()

    newargs = setdiff(sampled(jd.model),(:y,))
    pred = predictive(jd.model, newargs...)
    ((model=sm, post=post), cache, report)
end

function MMI.predict(sm::SossMLJModel, fitresult, Xnew)
    m = sm.model
    post = fitresult.post
    pred = predictive(m, keys(post[1])...)

    map(Tables.rowtable(Xnew)) do xrow
        args = merge(sm.transform([xrow]), sm.hyperparams)
        SossMLJPredictor(sm, post, pred, args)
    end
end

# function MMI.predict_joint(sm::SossMLJModel, fitresult, Xnew)
function predict_joint(sm::SossMLJModel, fitresult, Xnew)
    m = sm.model
    post = fitresult.post
    pred = predictive(m, keys(post[1])...)
    args = merge(sm.transform(Xnew), sm.hyperparams)
    return SossMLJPredictor(sm, post, pred, args)
end

end # module
