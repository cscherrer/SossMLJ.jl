samples = hcat([rand(predictor_joint) for sample = 1:1_000]...)
modes = [mode(samples[i, :]) for i = 1:size(samples, 1)]
overall_accuracy = Statistics.mean(modes .== iris[!, label_column])
@test overall_accuracy >= 0.9 # make sure the overall accuracy is at least 90%
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
@test setosa_accuracy > 0.9 # make sure the accuracy on the "setosa" class is at least 90%
@test versicolor_accuracy > 0.9 # make sure the accuracy on the "setosa" class is at least 90%
@test virginica_accuracy > 0.9 # make sure the accuracy on the "setosa" class is at least 90%
