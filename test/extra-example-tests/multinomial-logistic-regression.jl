samples = hcat([rand(predictor_joint) for sample = 1:1_000]...)
modes = [mode(samples[i, :]) for i = 1:size(samples, 1)]
overall_accuracy = Statistics.mean(modes .== iris[!, label_column])
@test overall_accuracy >= 0.9 # make sure the overall accuracy is at least 90%
