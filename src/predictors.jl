import Distributions
import MonteCarloMeasurements

function Base.rand(sp::SossMLJPredictor{M};
                   variable = sp.model.response) where {M}
    pars = rand(sp.post)
    args = merge(sp.args, pars)
    return getproperty(rand(sp.pred(args)), variable)
end

function Distributions.logpdf(sp::SossMLJPredictor{M}, x) where {M}
    # Get all the distribution mixture components
    dists = Base.Generator(sp.post) do pars
        args = merge(sp.args, pars)
        sp.pred(args)
    end
    # Evaluate logpdf(d,x) on each component d
    logvals = Base.Generator((d -> Distributions.logpdf(d, (y=x,))) ∘ dists.f, dists.iter)
    n = length(sp.post)
    return Distributions.logsumexp(logvals) - log(n)
end
