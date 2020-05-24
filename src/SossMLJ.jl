module SossMLJ

using MLJModelInterface
const MMI = MLJModelInterface

using Soss
import Soss: Model, @model, predictive, Univariate, Continuous, dynamicHMC
using Distributions
import Base

# Just a placeholder
m0 = @model X begin
    β ~ Normal()
    yhat = X * β
    y ~ For(size(yhat)) do j
        Normal(yhat[j], 1)
    end
end

# TODO: Add an way to specify which variable is observed, and which is predicted
# e.g. even if we observe `y`, we might sometimes prefer to predict yhat

# @mlj_model mutable struct SossMLJModel <: MMI.Probabilistic
#     model :: Model = m0
# end


export SossMLJModel

mutable struct SossMLJModel{A,B,M} <: MLJModelInterface.Probabilistic
    model::Model{A,B,M}
end

function SossMLJModel(; model = m0)
    model = SossMLJModel(model)
    message = MLJModelInterface.clean!(model)
    return model
end


function MLJModelInterface.clean!(model::SossMLJModel)
    warning = ""
    return warning
end

struct SossPredictor{A,B,M} <: Distributions.Sampleable{Univariate,Continuous}
    model :: Model{A,B,M}
    post
    xrow
end


function Base.rand(sp::SossPredictor{A,B,M}) where {A,B,M}
    par = rand(sp.post)
    X = reshape(sp.xrow, (1,:))
    args = merge(par, (X=X,))
    return rand(sp.model(args)).y[1]
end

function MMI.fit(sm::SossMLJModel, verbosity::Int, X, y, w=nothing)
    # construct the model
    
    jd = sm.model(X=X)

    # TODO: Specify inference method in SossMLJModel
    post = dynamicHMC(jd, (y=y,))

    # TODO: Allow w to be included

    cache = nothing
    report = NamedTuple{}()

    newargs = setdiff(sampled(jd.model),(:y,))
    m = predictive(jd.model, newargs...)
    ((model=m, post=post), cache, report)
end

function MMI.predict(sm::SossMLJModel, fitresult, Xnew)
    m = fitresult.model
    post = fitresult.post
    
    map(eachrow(Xnew)) do x
        SossPredictor(m, post, x)
    end
end

end # module
