import Distributions
import MonteCarloMeasurements
import NamedTupleTools

function Base.rand(sp::SossMLJPredictor{M};
                   response = sp.model.response) where {M}
    pars = rand(sp.post)
    args = merge(sp.args, pars)
    return getproperty(rand(sp.pred(args)), response)
end

function Distributions.logpdf(sp::SossMLJPredictor{M}, y;
        response = sp.model.response) where {M}
    # Get all the distribution mixture components
    dists = Base.Generator(sp.post) do pars
        args = merge(sp.args, pars)
        sp.pred(args)
    end

    y_namedtuple = NamedTupleTools.namedtuple(response)(tuple(y))
    # Evaluate logpdf(d,x) on each component d
    logvals = Base.Generator((d -> Distributions.logpdf(d, y_namedtuple)) ∘ dists.f, dists.iter)
    n = length(sp.post)
    return Distributions.logsumexp(logvals) - log(n)
end
