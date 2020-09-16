import Distributions
import MLJBase
import MLJModelInterface
import MonteCarloMeasurements

struct MixedVariate <: Distributions.VariateForm end
struct MixedSupport <: Distributions.ValueSupport end

abstract type AbstractSossMLJPredictor{M} <: Distributions.Distribution{MixedVariate, MixedSupport}
end

struct SossMLJPredictor{M,PO,PR,A} <: AbstractSossMLJPredictor{M}
    model::M
    post::PO
    pred::PR
    args::A
end

abstract type AbstractSossMLJModel{P, M} <: MLJModelInterface.JointProbabilistic
end

mutable struct SossMLJModel{P, M, H, I, R, RF, RP, T} <: AbstractSossMLJModel{P, M}
    # the first type parameter P is for the predictor_type
    # then we do the model, which is of type M
    model::M
    # after the model, we put the rest of the fields in alphabetical order for simplicity
    hyperparams::H
    infer::I
    response::R
    response_fit::RF
    response_predict::RP
    transform::T
end

@inline function default_transform(tbl)
    return (X = MLJModelInterface.matrix(tbl),)
end

function SossMLJModel(;
                      predictor::Type{P} = SossMLJPredictor,
                      model::M,
                      hyperparams::H = NamedTuple(),
                      infer::I = Soss.dynamicHMC,
                      response::R = :y,
                      response_fit::RF = response,
                      response_predict::RP = response,
                      transform::T = default_transform) where P where M where H where I where R where RF where RP where T
    result = SossMLJModel{P, M, H, I, R, RF, RP, T}(
        model,
        hyperparams,
        infer,
        response,
        response_fit,
        response_predict,
        transform,
    )
    return result
end

const ParticleMatrix = AbstractMatrix{<:MonteCarloMeasurements.AbstractParticles}
