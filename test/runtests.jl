using SossMLJ
using Test

import Documenter
import MLJBase
import Soss

include("examples-list.jl")

@testset "SossMLJ.jl" begin
    @testset "Unit tests" begin
        @testset "types.jl" begin
            @testset "Constructors for the `SossMLJModel` type" begin
                let
                    m = Soss.@model begin
                        y ~ Normal()
                        return y
                    end
                    hyperparams = (;)
                    transform = () -> ()
                    infer = Soss.dynamicHMC
                    model = SossMLJModel(m;
                                         transform = transform,
                                         infer = infer,
                                         hyperparams = hyperparams)
                    @test model.hyperparams == hyperparams
                    @test model.transform == transform
                    @test model.model == m
                    @test model.infer == infer
                    @test model.response isa Symbol
                    @test model.response == :y
                end
                let
                    m = Soss.@model begin
                        y ~ Normal()
                        return y
                    end
                    hyperparams = (;)
                    transform = () -> ()
                    infer = Soss.dynamicHMC
                    model = SossMLJModel(m;
                                         transform = transform,
                                         infer = infer,
                                         hyperparams = hyperparams,
                                         response = :y)
                    @test model.hyperparams == hyperparams
                    @test model.transform == transform
                    @test model.model == m
                    @test model.infer == infer
                    @test model.response isa Symbol
                    @test model.response == :y
                end
            end
        end
    end

    @testset "Examples" begin
        for example in EXAMPLES
            @testset "Example: $(example[1])" begin
                example_file = joinpath(EXAMPLESROOT, "example-$(example[2]).jl")
                include(example_file)
            end

            @testset "Additional tests for linear regression example" begin
                include(joinpath(EXAMPLESROOT, "example-linear-regression.jl"))
                MLJBase.evaluate!(mach, resampling=MLJBase.CV(; shuffle = true), measure=MLJBase.rms, operation=MLJBase.predict_mean)
            end
        end
    end

    @testset "Doctests" begin
        Documenter.doctest(SossMLJ)
    end
end
