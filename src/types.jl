import Distributions
import MLJBase
import MLJModelInterface

const MMI = MLJModelInterface

mutable struct SossMLJModel{M,T,I,H,R} <: MMI.JointProbabilistic
    model::M
    transform::T
    infer::I
    hyperparams::H
    response::R
end

function SossMLJModel(m::Soss.Model;
                      transform = tbl -> (X = MMI.matrix(tbl),),
                      infer = Soss.dynamicHMC,
                      hyperparams = NamedTuple(),
                      response = :y)
    return SossMLJModel(m,transform, infer, hyperparams, response)
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
