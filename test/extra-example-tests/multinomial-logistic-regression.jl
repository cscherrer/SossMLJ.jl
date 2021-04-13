import CategoricalArrays
import MLJBase
import MonteCarloMeasurements
import SossMLJ
import Statistics

samples = hcat([rand(predictor_joint) for sample = 1:1_000]...)
modes = [mode(samples[i, :]) for i = 1:size(samples, 1)]
overall_accuracy = Statistics.mean(modes .== iris[!, label_column])
@test overall_accuracy > 0.9 # make sure the overall accuracy is greater than 90%
classes = sort(unique(iris[!, label_column]))
@test length(classes) == 3
@test classes == ["setosa", "versicolor", "virginica"]
setosa_rows = findall(iris[!, label_column] .== "setosa")
versicolor_rows = findall(iris[!, label_column] .== "versicolor")
virginica_rows = findall(iris[!, label_column] .== "virginica")
@test length(setosa_rows) == 50
@test length(versicolor_rows) == 50
@test length(virginica_rows) == 50
@test length(iris[setosa_rows, label_column]) == 50
@test length(iris[versicolor_rows, label_column]) == 50
@test length(iris[virginica_rows, label_column]) == 50
setosa_accuracy  = Statistics.mean(modes[setosa_rows] .== iris[setosa_rows, label_column])
versicolor_accuracy  = Statistics.mean(modes[versicolor_rows] .== iris[versicolor_rows, label_column])
virginica_accuracy  = Statistics.mean(modes[virginica_rows] .== iris[virginica_rows, label_column])
@test setosa_accuracy > 0.9 # make sure the accuracy on the "setosa" class is greater than 90%
@test versicolor_accuracy > 0.9 # make sure the accuracy on the "setosa" class is greater than 90%
@test virginica_accuracy > 0.9 # make sure the accuracy on the "setosa" class is greater than 90%

predictor_marginal = MLJBase.predict(mach, iris[1:size(iris, 1), feature_columns])
@test predictor_marginal isa MLJBase.UnivariateFiniteVector
@test size(predictor_marginal) == (size(iris, 1),)

# Now, we make sure that the cross-validated accuracy is greater than 90%
nfolds = 4
evaluation_results = evaluate!(mach, resampling=CV(; nfolds = nfolds, shuffle = true), measure=MLJBase.accuracy, operation=MLJBase.predict_mode)
@test evaluation_results.measurement isa Vector
@test length(evaluation_results.measurement) == 1
@test evaluation_results.measurement[1] > 0.9 # make sure that the overall cross-validated accuracy is greater than 90%
@test evaluation_results.per_fold isa Vector
@test length(evaluation_results.per_fold) == 1
@test evaluation_results.per_fold[1] isa Vector
@test length(evaluation_results.per_fold[1]) == nfolds # make sure that the accuracy is greater than 80% in each fold
@test all(evaluation_results.per_fold[1] .> 0.8)

# Now, we test the correctness of our Brier score particle implementation by
# comparing it to the non-particle implementation in MLJBase.
let
    y = iris[!, :Species]
    μ̂ = SossMLJ.predict_particles(:μ)(mach, iris[!, feature_columns])
    pool = CategoricalArrays.pool(y)
    μ̂_dists = MLJBase.UnivariateFinite(pool.levels, Statistics.mean.(μ̂); pool=pool)
    mljbase_brier_score = MLJBase.BrierScore()(μ̂_dists, y)
    μ̂_mean = [MonteCarloMeasurements.Particles([Statistics.mean(element)]) for element in μ̂]
    sossmlj_brier_score_μ̂_mean = SossMLJ.BrierScoreDistribution()(μ̂_mean, y)
    @test all(mljbase_brier_score .≈ only.(getproperty.(sossmlj_brier_score_μ̂_mean, :particles)))
    @test all(mljbase_brier_score .≈ Statistics.mean.(sossmlj_brier_score_μ̂_mean))
    sossmlj_brier_score = SossMLJ.BrierScoreDistribution()(μ̂, y)
    @test all(abs.(mljbase_brier_score .- Statistics.mean.(sossmlj_brier_score)) .< 0.1)
end
