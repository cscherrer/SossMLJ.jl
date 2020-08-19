import Distributions
import MLJModelInterface
const MMI = MLJModelInterface

mutable struct SossMLJModel{H,T,M,I} <: MMI.JointProbabilistic
    hyperparams :: H
    transform :: T
    model :: M
    infer :: I
end

struct MixedVariate <: Distributions.VariateForm end
struct MixedSupport <: Distributions.ValueSupport end

struct SossMLJPredictor{M} <: Distributions.Distribution{MixedVariate, MixedSupport}
    model::M
    post
    pred
    args
end
