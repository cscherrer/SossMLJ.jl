import MLJBase
import MonteCarloMeasurements
import Statistics

const rms_distribution = RMSDistribution()
const rms_expected = RMSExpected()
const rms_median = RMSMedian()

function (::RMSDistribution)(ŷ::MonteCarloMeasurements.MvParticles, y::Vector{<:Real})
    return MLJBase.rms(ŷ, y)
end

function (::RMSExpected)(ŷ::MonteCarloMeasurements.MvParticles, y::Vector{<:Real})
    return Statistics.mean(rms_distribution(ŷ, y))
end

function (::RMSMedian)(ŷ::MonteCarloMeasurements.MvParticles, y::Vector{<:Real})
    return Statistics.median(rms_distribution(ŷ, y))
    # return Statistics.median.(rms_distribution(ŷ, y))
end
