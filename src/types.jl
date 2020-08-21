import Distributions
import MLJBase
import MLJModelInterface

mutable struct SossMLJModel{H,T,M,I,R} <: MLJModelInterface.JointProbabilistic
    hyperparams::H
    transform::T
    model::M
    infer::I
    response::R
end

function SossMLJModel(hyperparams, transform, model, infer)
    response = :y
    return SossMLJModel(hyperparams, transform, model, infer, response)
end

struct MixedVariate <: Distributions.VariateForm end
struct MixedSupport <: Distributions.ValueSupport end

struct SossMLJPredictor{M,PO,PR,A} <: Distributions.Distribution{MixedVariate, MixedSupport}
    model::M
    post::PO
    pred::PR
    args::A
end

struct RMSDistribution <: MLJBase.Measure end
struct RMSExpected <: MLJBase.Measure end
struct RMSMedian <: MLJBase.Measure end
