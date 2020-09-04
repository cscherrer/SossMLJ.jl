module SossMLJ

import Distributions
import MLJBase
import MLJModelInterface
import MonteCarloMeasurements
import Reexport
import Soss
import Statistics
import Tables

# Because we re-export Soss, we need to follow a few rules:
# 1. In `Project.toml`, the `[compat]` entry for `Soss` should only allow a
#    single breaking release of Soss at any given time.
# 2. Whenever we update the `[compat]` entry for `Soss` to a new breaking
#    release of Soss, we will need to make a breaking release of `SossMLJ`.
Reexport.@reexport using Soss

export SossMLJModel
export predict_particles
export rms_distribution
export rms_expected
export rms_median

include("types.jl")

include("loss-functions.jl")
include("machine-operations.jl")
include("models.jl")
include("particles.jl")
include("predictors.jl")

end # end module SossMLJ
