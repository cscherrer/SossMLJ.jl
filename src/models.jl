import MLJBase
import MLJModelInterface
import Soss
import Statistics
import NamedTupleTools

function MLJModelInterface.fit(sm::SossMLJModel, verbosity::Integer, X, y, w = nothing)
    # construct the model
    args = merge(sm.transform(X), sm.hyperparams)

    jd = sm.model(args)

    y_namedtuple = NamedTupleTools.namedtuple(sm.response_fit)(tuple(y))

    post = sm.infer(jd, y_namedtuple) # this is the slow part: doing inference with e.g. MCMC

    # TODO: Allow w to be included

    cache = nothing
    report = NamedTuple{}()

    newargs = setdiff(Soss.sampled(jd.model),(:y,))
    pred = Soss.predictive(jd.model, newargs...)
    fitresult = (model=sm, post=post)
    return (fitresult, cache, report)
end

function MLJModelInterface.clean!(smm::SossMLJModel)
    warning = ""
    return warning
end
