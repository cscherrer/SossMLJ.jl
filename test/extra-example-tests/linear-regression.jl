import MLJBase
import SossMLJ
import Statistics

MLJBase.evaluate!(mach, resampling=MLJBase.CV(; shuffle = true), measure=MLJBase.rms, operation=MLJBase.predict_mean)
particles = predict_particles(mach, X; response = :y)
@test rms_expected(particles, truth.y) == Statistics.mean(rms_distribution(particles, truth.y))

mach_1 = MLJBase.machine(model, X, truth.y)
@test_throws ErrorException predict_particles(mach_1, X; response = :y)
