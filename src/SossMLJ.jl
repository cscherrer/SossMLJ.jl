__precompile__(false) # TODO: enable precompilation for this package

module SossMLJ

import Distributions
import MLJBase
import MLJModelInterface
import MonteCarloMeasurements
import Soss
import Statistics
import Tables

const MMI = MLJModelInterface

export SossMLJModel
export predict_particles
export rms_distribution
export rms_expected
export rms_median

# TODO: remove the `logpdf` definition once this is merged into MLJBase upstream

Distributions.logpdf(d::MLJBase.UnivariateFinite, y::MLJBase.CategoricalValue)  = log(pdf(d,y))

include("types.jl")

include("loss-functions.jl")
include("machine-operations.jl")
include("models.jl")
include("particles.jl")
include("predictors.jl")

end # end module SossMLJ
