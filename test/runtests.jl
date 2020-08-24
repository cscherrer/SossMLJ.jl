using SossMLJ
using Test

import Documenter
import Literate
import Soss

include("examples-list.jl")

@testset "SossMLJ.jl" begin
    @testset "Unit tests" begin
        @testset "types.jl" begin
            @testset "Constructors for SossMLJModel" begin
                m = Soss.@model begin
                    y ~ Normal()
                    return y
                end
                hyperparams = (;)
                transform = () -> ()
                infer = Soss.dynamicHMC
                model = SossMLJModel(m, transform, infer, hyperparams, :y)
                @test model.hyperparams == hyperparams
                @test model.transform == transform
                @test model.model == m
                @test model.infer == infer
                @test model.response isa Symbol
                @test model.response == :y
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
                MLJ.evaluate!(mach, resampling=MLJ.CV(; shuffle = true), measure=MLJ.rms, operation=MLJ.predict_mean)
            end
        end
    end

    @testset "Doctests" begin
        Documenter.doctest(SossMLJ)
    end
end
