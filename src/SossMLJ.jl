module SossMLJ

using MLJModelInterface
const MMI = MLJModelInterface

using Soss
import Soss: Model, @model, predictive, Univariate, Continuous, dynamicHMC
using Distributions

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

@mlj_model mutable struct SossModel <: MMI.Probabilistic
    model :: Model = m0
end

struct SossPredictor <: Distributions.Sampleable{Univariate,Continuous}
    model
    post
    xrow
end

function rand(sp::SossPredictor)
    par = rand(sp.post)
    X = reshape(sp.xrow, (1,:))
    args = merge(par, (X=X,))
    rand(sp.model(args)).y[1]
end

function MMI.fit(sm::SossModel, verbosity::Int, X, y, w=nothing)
    # construct the model
    
    jd = sm.model(X=X)

    # TODO: Specify inference method in SossModel
    post = dynamicHMC(jd, (y=y,))

    # TODO: Allow w to be included

    cache = nothing
    report = NamedTuple{}()

    newargs = setdiff(sampled(jd.model),(:y,))
    m = predictive(jd.model, newargs...)
    ((model=m, post=post), cache, report)
end

function MMI.predict(sm::SossModel, fitresult, Xnew)
    m = fitresult.model
    post = fitresult.post
    
    map(eachrow(Xnew)) do x
        SossPredictor(m, post, x)
    end
end

end # module
