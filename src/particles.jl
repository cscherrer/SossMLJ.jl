import MLJModelInterface
import Soss

const MMI = MLJModelInterface

function predict_particles(predictor::SossMLJPredictor, Xnew)
    args = predictor.args
    pars = Soss.particles(predictor.post)
    pred = predictor.pred
    transform = predictor.model.transform
    dist = pred(merge(args, transform(Xnew), pars))
    return Soss.particles(dist)
end

function predict_particles(sm::SossMLJModel,
                           fitresult,
                           Xnew;
                           response = sm.response)
    predictor_joint = MMI.predict_joint(sm, fitresult, Xnew)
    return getproperty(predict_particles(predictor_joint, Xnew), response)
end
