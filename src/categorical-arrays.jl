import CategoricalArrays
import MonteCarloMeasurements

@inline function _categorical_pdf(μ̂::ParticleMatrix,
                                  y::CategoricalArrays.AbstractCategoricalVector)
    num_rows = length(y)
    pool = CategoricalArrays.pool(y)
    support = pool.levels
    result = [μ̂[i, findfirst(isequal(y[i]), support)] for i in 1:num_rows]
    return result
end
