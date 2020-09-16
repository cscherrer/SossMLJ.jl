using SossMLJ
using Test

import DataFrames
import Documenter
import MLJBase
import MLJModelInterface
import MonteCarloMeasurements
import Soss

include("examples-list.jl")

@testset "SossMLJ.jl" begin
    @testset "Unit tests" begin
        @testset "types.jl" begin
            @testset "default transform" begin
                df = DataFrames.DataFrame()
                df[!, :a] = [1.0, 2.0, 3.0]
                df[!, :b] = [4.0, 5.0, 6.0]
                @test SossMLJ.default_transform(df) == (X = [1.0 4.0; 2.0 5.0; 3.0 6.0],)
                @test SossMLJ.default_transform(df).X isa Matrix{Float64}
            end
            @testset "Constructors for the `SossMLJModel` type" begin
                @testset "default response" begin
                    m = Soss.@model begin
                        y ~ Normal()
                        return y
                    end
                    hyperparams = (;)
                    transform = () -> ()
                    infer = Soss.dynamicHMC
                    model = SossMLJModel(;
                        model = m,
                        hyperparams = hyperparams,
                        infer = infer,
                        transform = transform,
                    )
                    @test model.hyperparams == hyperparams
                    @test model.transform == transform
                    @test model.model == m
                    @test model.infer == infer
                    @test model.response isa Symbol
                    @test model.response == :y
                    @test MLJModelInterface.clean!(model) == ""
                end
                @testset "specify response" begin
                    m = Soss.@model begin
                        y ~ Normal()
                        return y
                    end
                    hyperparams = (;)
                    transform = () -> ()
                    infer = Soss.dynamicHMC
                    model = SossMLJModel(;
                        model = m,
                        hyperparams = hyperparams,
                        infer = infer,
                        response = :y,
                        transform = transform,
                    )
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
            @testset "Run example: $(example[1])" begin
                example_file = joinpath(EXAMPLESROOT, "example-$(example[2]).jl")
                extra_example_tests = joinpath(TESTROOT, "extra-example-tests", "$(example[2]).jl")
                @info("Running $(example_file)")
                include(example_file)
                if isfile(extra_example_tests)
                    @info("Running $(extra_example_tests)")
                    include(extra_example_tests)
                end
            end
        end
    end

    @testset "Doctests" begin
        Documenter.doctest(SossMLJ)
    end
end
