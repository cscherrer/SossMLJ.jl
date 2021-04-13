import Distributions
import MonteCarloMeasurements
import NamedTupleTools

function Base.rand(sp::SossMLJPredictor{M};
                   response = sp.model.response) where M
    pars = rand(sp.post)
    args = merge(sp.args, pars)
    return getproperty(rand(sp.pred(args)), response)
end
