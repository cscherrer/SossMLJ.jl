import MLJModelInterface
import Soss

function _predict_all_particles(predictor::SossMLJPredictor, Xnew)
    args = predictor.args
    pars = Soss.particles(predictor.post)
    pred = predictor.pred
    transform = predictor.model.transform
    dist = pred(merge(args, transform(Xnew), pars))
    return Soss.particles(dist)
end

function _predict_all_particles(sm::SossMLJModel, fitresult, Xnew)
    predictor_joint = MLJModelInterface.predict_joint(sm, fitresult, Xnew)
    return predict_particles(predictor_joint, Xnew)
end

function predict_particles(response::Symbol)
    function _predict_particles(varargs...; kwargs...)
        if :response in keys(kwargs)
            throw(ArgumentError("you cannot provide the `response` keyword argument to this method"))
        end
        return predict_particles(varargs...; response = response, kwargs...)
    end
    return _predict_particles
end

function predict_particles(original_predictor::SossMLJPredictor,
                           Xnew;
                           response = original_predictor.model.response_predict)
    original_args = original_predictor.args
    original_sm  = original_predictor.model
    original_post = original_predictor.post
    original_model = original_sm.model
    original_transform = original_sm.transform
    original_pars = Soss.particles(original_post)
    if response in keys(original_pars)
        return original_pars
    end
    new_model = Soss.before(
        original_model,
        response;
        inclusive = true,
        strict = false,
    )
    new_pred = Soss.predictive(
        new_model,
        keys(original_post[1])...,
    )
    new_dist = new_pred(merge(original_args, original_transform(Xnew), original_pars))
    return Soss.particles(new_dist)
end

function predict_particles(sm::SossMLJModel,
                           fitresult,
                           Xnew;
                           response = sm.response_predict)
    predictor_joint = MLJModelInterface.predict_joint(sm, fitresult, Xnew)
    result_particles = getproperty(
        predict_particles(predictor_joint, Xnew; response = response),
        response,
    )
    return result_particles
end
