__precompile__(false) # TODO: enable precompilation for this package

module SossMLJ

import Distributions
import Soss
import Soss: Model, @model, predictive, Univariate, Continuous, dynamicHMC
import MLJBase # TODO: remove the dependency on MLJBase.jl
import MLJModelInterface
import MonteCarloMeasurements
import Tables
const MMI = MLJModelInterface

export SossMLJModel
export predict_particles

# Just a placeholder
# m0 = @model X,α,σ begin
#     k = size(X,2)
#     β ~ Normal(0,α) |> iid(k)
#     yhat = X * β
#     y ~ For(eachindex(yhat)) do j
#         Normal(yhat[j], σ)
#     end
# end

# TODO: Add an way to specify which variable is observed, and which is predicted
# e.g. even if we observe `y`, we might sometimes prefer to predict yhat

# @mlj_model mutable struct SossMLJModel <: MMI.JointProbabilistic
#     model :: Model = m0
# end

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

struct MixedVariate <: Distributions.VariateForm end
struct MixedSupport <: Distributions.ValueSupport end

struct SossMLJPredictor{M} <: Distributions.Distribution{MixedVariate, MixedSupport}
    model::M
    post
    pred
    args
end

function Base.rand(sp::SossMLJPredictor{M}) where {M}
    pars = rand(sp.post)
    args = merge(sp.args, pars)
    return rand(sp.pred(args)).y
end

function Distributions.logpdf(sp::SossMLJPredictor{M}, x) where {M}
    # Get all the distribution mixture components
    dists = Base.Generator(sp.post) do pars
        args = merge(sp.args, pars)
        sp.pred(args)
    end
    # Evaluate logpdf(d,x) on each component d
    logvals = Base.Generator((d -> logpdf(d, (y=x,))) ∘ dists.f, dists.iter)
    n = length(sp.post)
    return logsumexp(logvals) - log(n)
end

function MMI.fit(sm::SossMLJModel, verbosity::Int, X, y, w=nothing)
    # construct the model
    args = merge(sm.transform(X), sm.hyperparams)

    jd = sm.model(args)

    post = sm.infer(jd, (y=y,))

    # TODO: Allow w to be included

    cache = nothing
    report = NamedTuple{}()

    newargs = setdiff(Soss.sampled(jd.model),(:y,))
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

function predict_particles(predictor::SossMLJPredictor, Xnew)
    args = predictor.args
    pars = MonteCarloMeasurements.particles(predictor.post)
    pred = predictor.pred
    transform = predictor.model.transform
    dist = pred(merge(args, transform(Xnew), pars))
    return particles(dist)
end

#### BEGIN code to make predict_joint work on machines. Remove this once it is merged in MMI upstream.

const OPERATIONS = (:predict_joint,)

for operation in OPERATIONS

    if operation != :inverse_transform

        ex = quote
            # 0. operations on machs, given empty data:
            function $(operation)(mach::MLJBase.Machine; rows=:)
                # Base.depwarn("`$($operation)(mach)` and "*
                #              "`$($operation)(mach, rows=...)` are "*
                #              "deprecated. Data or nodes "*
                #              "should be explictly specified, "*
                #              "as in `$($operation)(mach, X)`. ",
                #              Base.Core.Typeof($operation).name.mt.name)
                if isempty(mach.args) # deserialized machine with no data
                    throw(ArgumentError("Calling $($operation) on a "*
                                        "deserialized machine with no data "*
                                        "bound to it. "))
                end
                return ($operation)(mach, mach.args[1](rows=rows))
            end
        end
        eval(ex)

    end
end

_symbol(f) = Base.Core.Typeof(f).name.mt.name

for operation in OPERATIONS

    ex = quote
        # 1. operations on machines, given *concrete* data:
        function $operation(mach::MLJBase.Machine, Xraw)
            if mach.state > 0
                return $(operation)(mach.model, mach.fitresult,
                                    Xraw)
            else
                error("$mach has not been trained.")
            end
        end

        function $operation(mach::MLJBase.Machine{<:MMI.Static}, Xraw, Xraw_more...)
            isdefined(mach, :fitresult) || (mach.fitresult = nothing)
            return $(operation)(mach.model, mach.fitresult,
                                    Xraw, Xraw_more...)
        end

        # 2. operations on machines, given *dynamic* data (nodes):
        $operation(mach::MLJBase.Machine, X::MLJBase.AbstractNode) =
            node($(operation), mach, X)

        $operation(mach::MLJBase.Machine{<:MMI.Static}, X::MLJBase.AbstractNode, Xmore::MLJBase.AbstractNode...) =
            node($(operation), mach, X, Xmore...)
    end
    eval(ex)
end

## SURROGATE AND COMPOSITE MODELS

for operation in [:predict_joint,]
    ex = quote
        $operation(model::Union{MLJBase.Composite,MLJBase.Surrogate}, fitresult,X) =
            fitresult.$operation(X)
    end
    eval(ex)
end

#### END code to make predict_joint work on machines. Remove this once it is merged in MMI upstream.

end # module
