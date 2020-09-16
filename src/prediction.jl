import MLJBase
import MLJModelInterface
import Soss
import Statistics
import NamedTupleTools

function MLJModelInterface.predict_joint(sm::SossMLJModel,
                                         fitresult,
                                         Xnew)
    m = sm.model
    post = fitresult.post
    pred = Soss.predictive(m, keys(post[1])...)
    args = merge(sm.transform(Xnew), sm.hyperparams)
    return SossMLJPredictor(sm, post, pred, args)
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
    result = _predict_marginals_incorrectly(
        sm,
        fitresult,
        Xnew;
        response = sm.response_predict,
    )
    return result
end

# TODO: remove this hacky and incorrect method.
# Here's how it works:
# 1. Construct the joint distribution
# 2. Draw five thousand samples from the joint distribution
# 3. Use these five thousand samples to empirically estimate the probabilities of each class
# 4. Return a `UnivariateFiniteVector` using these probabilities
function _predict_marginals_incorrectly(sm::SossMLJModel{<:MLJBase.UnivariateFinite},
                                        fitresult,
                                        Xnew;
                                        response = sm.response_predict)
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

function MLJModelInterface.predict_mean(sm::SossMLJModel,
                                        fitresult,
                                        Xnew;
                                        response = sm.response_predict)
    pars = predict_particles(sm, fitresult, Xnew; response = response)
    return Statistics.mean.(pars)
end

function MLJModelInterface.predict_mode(sm::SossMLJModel{<:MLJBase.UnivariateFinite},
                                        fitresult, Xnew;
                                        response = sm.response_predict)
    marginal_distributions = MLJModelInterface.predict(sm, fitresult, Xnew; response = sm.response) # `marginal_distributions` is of type `UnivariateFiniteVector`
    modes = [MLJBase.mode(marginal_dist) for marginal_dist in marginal_distributions]
    return modes
end
