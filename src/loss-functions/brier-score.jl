import CategoricalArrays
import MLJBase
import MonteCarloMeasurements
import Statistics

const brier_score_distribution = BrierScoreDistribution()
const brier_score_expected = BrierScoreExpected()
const brier_score_median = BrierScoreMedian()

function (::BrierScoreDistribution)(μ̂::ParticleMatrix,
                                    y::CategoricalArrays.AbstractCategoricalVector)
    check_rows(μ̂, y)
    offset = 1 .+ vec(sum(μ̂.^2, dims=2))
    unweighted = 2 .* _categorical_pdf(μ̂, y) .- offset
    return unweighted
end

function (::BrierScoreExpected)(μ̂::ParticleMatrix,
                                y::CategoricalArrays.AbstractCategoricalVector)
    return Statistics.mean.(brier_score_distribution(μ̂, y))
end

function (::BrierScoreMedian)(μ̂::ParticleMatrix,
                              y::CategoricalArrays.AbstractCategoricalVector)
    return Statistics.median.(brier_score_distribution(μ̂, y))
end

MLJBase.orientation(::Type{<:BrierScoreDistribution}) = :score
MLJBase.orientation(::Type{<:BrierScoreExpected}) = :score
MLJBase.orientation(::Type{<:BrierScoreMedian}) = :score

MLJBase.reports_each_observation(::Type{<:BrierScoreDistribution}) = true
MLJBase.reports_each_observation(::Type{<:BrierScoreExpected}) = true
MLJBase.reports_each_observation(::Type{<:BrierScoreMedian}) = true
