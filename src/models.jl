import MLJModelInterface
import Statistics
import NamedTupleTools: namedtuple

function MLJModelInterface.fit(sm::SossMLJModel, verbosity::Int, X, y, w=nothing)
    # construct the model
    args = merge(sm.transform(X), sm.hyperparams)

    jd = sm.model(args)

    y_namedtuple = namedtuple(sm.response)(y)
    post = sm.infer(jd, y_namedtuple)

    # TODO: Allow w to be included

    cache = nothing
    report = NamedTuple{}()

    newargs = setdiff(Soss.sampled(jd.model),(:y,))
    pred = Soss.predictive(jd.model, newargs...)
    ((model=sm, post=post), cache, report)
end

function MLJModelInterface.clean!(smm::SossMLJModel)
    warning = ""
    return warning
end

function MLJModelInterface.predict(sm::SossMLJModel, fitresult, Xnew)
    m = sm.model
    post = fitresult.post
    pred = Soss.predictive(m, keys(post[1])...)

    map(Tables.rowtable(Xnew)) do xrow
        args = merge(sm.transform([xrow]), sm.hyperparams)
        SossMLJPredictor(sm, post, pred, args)
    end
end

function MLJModelInterface.predict_joint(sm::SossMLJModel, fitresult, Xnew)
    m = sm.model
    post = fitresult.post
    pred = Soss.predictive(m, keys(post[1])...)
    args = merge(sm.transform(Xnew), sm.hyperparams)
    return SossMLJPredictor(sm, post, pred, args)
end

function MLJModelInterface.predict_mean(sm::SossMLJModel, fitresult, Xnew;
                          response = sm.response)
    predictor_joint = MLJModelInterface.predict_joint(sm, fitresult, Xnew)
    return Statistics.mean(getproperty(predict_particles(predictor_joint, Xnew), response))
end
