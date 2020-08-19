__precompile__(false) # TODO: enable precompilation for this package

module SossMLJ

import Distributions
import Soss
import Soss: Model, @model, predictive, Univariate, Continuous, dynamicHMC
import MLJBase # TODO: remove the dependency on MLJBase.jl
import MLJModelInterface
import MonteCarloMeasurements
import Tables
const MMI = MLJModelInterface

export SossMLJModel
export predict_particles

include("types.jl")

include("mljbase.jl") # TODO: remove the dependency on MLJBase.jl
include("model.jl")
include("predictor.jl")

end # end module SossMLJ
