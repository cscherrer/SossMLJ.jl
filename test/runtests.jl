using SossMLJ
using Test

import Documenter

@testset "SossMLJ.jl" begin
    @testset "Unit tests" begin
    end

    @testset "Examples" begin
        @testset "mainexample.jl" begin
            mainexample = joinpath(dirname(dirname(@__FILE__)), "mainexample.jl")
            include(mainexample)
        end
    end

    @testset "Doctests" begin
        Documenter.doctest(SossMLJ)
    end
end
