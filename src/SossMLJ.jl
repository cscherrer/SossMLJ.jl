module SossMLJ

import CategoricalArrays
import Distributions
import MLJBase
import MLJModelInterface
import MonteCarloMeasurements
import Soss
import Statistics
import Tables

# exports:
export SossMLJModel
export predict_particles

# loss function exports:
export BrierScoreDistribution, BrierScoreExpected, BrierScoreMedian
export RMSDistribution, RMSExpected, RMSMedian
export brier_score_distribution, brier_score_expected, brier_score_median
export rms_distribution, rms_expected, rms_median

include("types.jl")
include("loss-functions/types.jl")

include("categorical-arrays.jl")
include("check-rows.jl")
include("distributions.jl")
include("machine-operations.jl")
include("models.jl")
include("particles.jl")
include("prediction.jl")
include("rand.jl")

include("loss-functions/brier-score.jl")
include("loss-functions/rms.jl")

end # end module SossMLJ
