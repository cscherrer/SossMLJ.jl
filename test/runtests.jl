using SossMLJ
using Test

import Documenter
import Literate

include("examples-list.jl")

@testset "SossMLJ.jl" begin
    @testset "Unit tests" begin
    end

    @testset "Examples" begin
        for example in EXAMPLES
            @testset "Example: $(example[1])" begin
                example_file = joinpath(EXAMPLESROOT, "example-$(example[2]).jl")
                include(example_file)
            end
        end
    end

    @testset "Doctests" begin
        Documenter.doctest(SossMLJ)
    end
end
