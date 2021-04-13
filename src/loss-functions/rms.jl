import MLJBase
import MonteCarloMeasurements
import Statistics

const rms_distribution = RMSDistribution()
const rms_expected = RMSExpected()
const rms_median = RMSMedian()

function (::RMSDistribution)(μ̂::MonteCarloMeasurements.MvParticles,
                             y::AbstractVector{<:Real})
    return MLJBase.rms(μ̂, y)
end

function (::RMSExpected)(μ̂::MonteCarloMeasurements.MvParticles,
                         y::AbstractVector{<:Real})
    return Statistics.mean.(rms_distribution(μ̂, y))
end

function (::RMSMedian)(μ̂::MonteCarloMeasurements.MvParticles,
                       y::AbstractVector{<:Real})
    return Statistics.median.(rms_distribution(μ̂, y))
end

MLJBase.orientation(::Type{<:RMSDistribution}) = :loss
MLJBase.orientation(::Type{<:RMSExpected}) = :loss
MLJBase.orientation(::Type{<:RMSMedian}) = :loss

MLJBase.reports_each_observation(::Type{<:RMSDistribution}) = false
MLJBase.reports_each_observation(::Type{<:RMSExpected}) = false
MLJBase.reports_each_observation(::Type{<:RMSMedian}) = false
