import MLJBase

struct BrierScoreDistribution <: MLJBase.Measure end
struct BrierScoreExpected <: MLJBase.Measure end
struct BrierScoreMedian <: MLJBase.Measure end

struct RMSDistribution <: MLJBase.Measure end
struct RMSExpected <: MLJBase.Measure end
struct RMSMedian <: MLJBase.Measure end
