import MLJModelInterface
const MMI = MLJModelInterface

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

function MLJModelInterface.clean!(smm::SossMLJModel)
    warning = ""
    return warning
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
