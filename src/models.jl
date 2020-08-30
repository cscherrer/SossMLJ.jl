import MLJBase
import MLJModelInterface
import Soss
import Statistics
import NamedTupleTools

function MLJModelInterface.fit(sm::SossMLJModel, verbosity::Integer, X, y, w = nothing)
    # construct the model
    args = merge(sm.transform(X), sm.hyperparams)

    jd = sm.model(args)

    y_namedtuple = NamedTupleTools.namedtuple(sm.response)(tuple(y))

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

function MLJModelInterface.predict(sm::SossMLJModel{<:SossMLJPredictor}, fitresult, Xnew)
    m = sm.model
    post = fitresult.post
    pred = Soss.predictive(m, keys(post[1])...)

    return map(Tables.rowtable(Xnew)) do xrow
        args = merge(sm.transform([xrow]), sm.hyperparams)
        return SossMLJPredictor(sm, post, pred, args)
    end
end

# This method MUST return a `UnivariateFiniteVector`
# Currently, this is a hacky and incorrect implementation.
# TODO: Implement this method correctly.
function MLJModelInterface.predict(sm::SossMLJModel{<:MLJBase.UnivariateFinite}, fitresult, Xnew; response = sm.response)
    return _predict_marginals_incorrectly(sm, fitresult, Xnew; response = sm.response)
end

# TODO: remove this hacky and incorrect method.
# Here's how it works:
# 1. Construct the joint distribution
# 2. Draw five thousand samples from the joint distribution
# 3. Use these five thousand samples to empirically estimate the probabilities of each class
# 4. Return a `UnivariateFiniteVector` using these probabilities
function _predict_marginals_incorrectly(sm::SossMLJModel{<:MLJBase.UnivariateFinite}, fitresult, Xnew; response = sm.response)
    predictor_joint = MLJModelInterface.predict_joint(sm, fitresult, Xnew)
    num_samples = 5_000
    samples = hcat([rand(predictor_joint; response = response) for sample = 1:num_samples]...)
    pool = sm.hyperparams.pool
    levels = pool.levels
    num_levels = length(levels)
    num_rows = size(Xnew, 1)
    probs = Matrix{Float64}(undef, num_rows, num_levels)
    for i = 1:num_rows
        for j = 1:num_levels
            level = levels[j]
            level_prob = sum(samples[i, :] .== level)/num_samples
            probs[i, j] = level_prob
        end
    end
    support = levels
    marginal_distributions = MLJBase.UnivariateFinite(support, probs; pool = pool)
    return marginal_distributions
end

function MLJModelInterface.predict_joint(sm::SossMLJModel, fitresult, Xnew)
    m = sm.model
    post = fitresult.post
    pred = Soss.predictive(m, keys(post[1])...)
    args = merge(sm.transform(Xnew), sm.hyperparams)
    return SossMLJPredictor(sm, post, pred, args)
end

function MLJModelInterface.predict_mean(sm::SossMLJModel{<:SossMLJPredictor}, fitresult, Xnew; response = sm.response)
    predictor_joint = MLJModelInterface.predict_joint(sm, fitresult, Xnew)
    return Statistics.mean(getproperty(predict_particles(predictor_joint, Xnew), response))
end

function MLJModelInterface.predict_mode(sm::SossMLJModel{<:MLJBase.UnivariateFinite}, fitresult, Xnew; response = sm.response)
    marginal_distributions = MLJModelInterface.predict(sm, fitresult, Xnew; response = sm.response) # `marginal_distributions` is of type `UnivariateFiniteVector`
    modes = [MLJBase.mode(marginal_dist) for marginal_dist in marginal_distributions]
    return modes
end
