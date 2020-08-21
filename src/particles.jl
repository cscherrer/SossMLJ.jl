import Soss

function predict_particles(predictor::SossMLJPredictor, Xnew)
    args = predictor.args
    pars = Soss.particles(predictor.post)
    pred = predictor.pred
    transform = predictor.model.transform
    dist = pred(merge(args, transform(Xnew), pars))
    return Soss.particles(dist)
end

function predict_particles(sm::SossMLJModel, fitresult, Xnew;
                           variable = sm.response)
    predictor_joint = predict_joint(sm, fitresult, Xnew)
    return getproperty(predict_particles(predictor_joint, Xnew), variable)
end
