import MLJModelInterface
import Statistics
import NamedTupleTools

const MMI = MLJModelInterface

function MMI.fit(sm::SossMLJModel, verbosity::Int, X, y, w=nothing)
    # construct the model
    args = merge(sm.transform(X), sm.hyperparams)

    jd = sm.model(args)

    y_namedtuple = NamedTupleTools.namedtuple(sm.response)(tuple(y))
    post = sm.infer(jd, y_namedtuple)

    # TODO: Allow w to be included

    cache = nothing
    report = NamedTuple{}()

    newargs = setdiff(Soss.sampled(jd.model),(:y,))
    pred = Soss.predictive(jd.model, newargs...)
    ((model=sm, post=post), cache, report)
end

function MMI.clean!(smm::SossMLJModel)
    warning = ""
    return warning
end

function MMI.predict(sm::SossMLJModel, fitresult, Xnew)
    m = sm.model
    post = fitresult.post
    pred = Soss.predictive(m, keys(post[1])...)

    map(Tables.rowtable(Xnew)) do xrow
        args = merge(sm.transform([xrow]), sm.hyperparams)
        SossMLJPredictor(sm, post, pred, args)
    end
end

function MMI.predict_joint(sm::SossMLJModel, fitresult, Xnew)
    m = sm.model
    post = fitresult.post
    pred = Soss.predictive(m, keys(post[1])...)
    args = merge(sm.transform(Xnew), sm.hyperparams)
    return SossMLJPredictor(sm, post, pred, args)
end

function MMI.predict_mean(sm::SossMLJModel,
                          fitresult,
                          Xnew;
                          response = sm.response)
    predictor_joint = MMI.predict_joint(sm, fitresult, Xnew)
    return Statistics.mean(getproperty(predict_particles(predictor_joint, Xnew), response))
end
