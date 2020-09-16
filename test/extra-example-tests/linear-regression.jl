import MLJBase
import SossMLJ
import Statistics

MLJBase.evaluate!(mach, resampling=MLJBase.CV(; shuffle = true), measure=MLJBase.rms, operation=MLJBase.predict_mean)
particles = predict_particles(mach, X; response = :y)
@test rms_expected(particles, truth.y) == Statistics.mean(rms_distribution(particles, truth.y))

for response in [:β, :σ, :η, :μ]
    a = getproperty(SossMLJ._predict_all_particles(mach, X), response)
    b = predict_particles(mach, X; response = response)
    @test a == b
end

for response in [:y]
    a = getproperty(SossMLJ._predict_all_particles(mach, X), response)
    b = predict_particles(mach, X; response = response)
    @test all(.≈(Statistics.mean.(a), Statistics.mean.(b); atol = 0.001))
end

mach_1 = MLJBase.machine(model, X, truth.y)
@test_throws ErrorException predict_particles(mach_1, X; response = :y)
@test_throws ErrorException SossMLJ._predict_all_particles(mach_1, X)
