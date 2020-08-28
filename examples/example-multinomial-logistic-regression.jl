# # Multinomial logistic regression

# Import the necessary packages:

using DataFrames
using Distributions
using MLJBase
using NNlib
using RDatasets
using Soss
using SossMLJ
using Statistics

# In this example, we fit a Bayesian multinomial logistic regression model
# with the canonical link function.

# Suppose that we are given a matrix of features `X` and a column vector of
# labels `y`. `X` has `n` rows and `p` columns. `y` has `n` elements. We assume
# that our observation vector `y` is a realization of a random variable `Y`.
# We define `μ` (mu) as the expected value of `Y`, i.e. `μ := E[Y]`. Our model
# comprises three components:
#
# 1. The probability distribution of `Y`. We assume that each `Yᵢ` follows a multinomial distribution with `k` categories, mean `μᵢ`, and one trial.
# 2. The systematic component, which consists of linear predictor `η` (eta), which we define as `η := Xβ`, where `β` is the column vector of `p` coefficients.
# 3. The link function `g`, which provides the following relationship: `g(E[Y]) = g(μ) = η = Xβ`. It follows that `μ = g⁻¹(η)`, where `g⁻¹` denotes the inverse of `g`. In multinomial logistic regression, the canonical link function is the generalized logit function. The inverse of the generalized logit function is the softmax function. Therefore, when using the canonical link function, `μ = g⁻¹(η) = softmax(η)`.

# A multinomial distribution with one trial is equivalent to the categorical distribution.
# Therefore, the following two statements are equivalent:
# - `Yᵢ` follows a multinomial distribution with `k` categories, mean `μᵢ`, and one trial.
# - `Yᵢ` follows a categorical distribution with `k` categories and mean `μᵢ`.
#
# Observe that the logistic regression model is a special case of the
# multinomial logistic regression model where `k = 2`.
#
# In this model, the parameters that we want to estimate are the coefficients `β`.
# We need to select prior distributions for these parameters. For each `βᵢ`
# we choose a normal distribution with zero mean and unit variance. Here, `βᵢ`
# denotes the `i`th component of `β`.

# We define this model using the Soss probabilistic programming library:

m = @model X,pool begin
    n = size(X,1) # number of observations
    p = size(X,2) # number of features
    k = length(pool.levels) # number of classes
    β ~ Normal(0.0, 1.0) |> iid(p, k) # coefficients
    η = X * β # linear predictor
    μ = NNlib.softmax(η; dims=2) # μ = g⁻¹(η) = softmax(η)
    y_dists = UnivariateFinite(pool.levels, μ; pool=pool) # `UnivariateFinite` is mathematically equivalent to `Categorical`
    y ~ For(j -> y_dists[j], n) # `Yᵢ ~ UnivariateFinite(mean=μᵢ, categories=k)`, which is mathematically equivalent to `Yᵢ ~ Categorical(mean=μᵢ, categories=k)`
end;

# Import the *Iris* flower data set:

iris = dataset("datasets", "iris");

# Define our feature columns:

feature_columns = [
    :PetalLength,
    :PetalWidth,
    :SepalLength,
    :SepalWidth,
]

# Define our label column:

label_column = :Species

# Convert the Soss model into a `SossMLJModel`:

model = SossMLJModel(m;
    hyperparams = (pool=iris.Species.pool,),
    transform   = tbl -> (X=MLJBase.matrix(tbl[1:size(tbl, 1), feature_columns]),),
    infer       = dynamicHMC,
    response    = :y,
);

# Create an MLJ machine for fitting our model:

mach = MLJBase.machine(model, iris[1:size(iris, 1), feature_columns], iris[1:size(iris, 1), :Species])

# Fit the machine. This may take several minutes.

MLJBase.fit!(mach)

# Construct the joint posterior:

predictor_joint = MLJBase.predict_joint(mach, iris[1:size(iris, 1), feature_columns])
typeof(predictor_joint)

# Draw a sample from the posterior:

single_sample = rand(predictor_joint)
