# # Bayesian linear regression

# Import the necessary packages:

using Distributions
using MLJ
using MLJBase
using MLJModelInterface
using Soss
using SossMLJ
using Statistics

# Use the Soss probabilistic programming language
# to define a Bayesian linear regression model:

m = @model X,s,a begin
    σ ~ HalfNormal(a)
    k = size(X,2)
    β ~ Normal(0,s) |> iid(k)
    yhat = X * β
    y ~ For(eachindex(yhat)) do j
        Normal(yhat[j], σ)
    end
end

# Generate some synthetic features:

num_rows = 100
X = (x1=randn(num_rows), x2=randn(num_rows), x3=randn(num_rows))

# Define the hyperparameters of our prior distributions:

hyperparameters = (s=2.0, a=1.0)

# Convert the Soss model into a `SossMLJModel`:

model = SossMLJModel(hyperparameters, X -> (X=MLJModelInterface.matrix(X),), m, dynamicHMC)

# Generate some synthetic labels:

args = merge(model.transform(X), hyperparameters)
truth = rand(m(args))
y = truth.y

# Create an MLJ machine for fitting our model:

mach = machine(model, X, y)

# Fit the model:

fit!(mach)

# Construct the posterior distribution and the joint posterior predictive distribution:

##predictor_joint = MLJ.predict_joint(mach, X)
predictor_joint = SossMLJ.predict_joint(mach, X)
typeof(predictor_joint)

# Compare the posterior distribution of `β` to the true coefficients `β`:

truth.β - predict_particles(predictor_joint, X).β

# Compare the posterior distribution of `σ` to the true dispersion parameter `σ`:

truth.σ - predict_particles(predictor_joint, X).σ

# Compare the joint posterior predictive distribution to the true labels:

truth.yhat - predict_particles(predictor_joint, X).yhat

# Draw a single sample from the joint posterior predictive distribution:

single_sample = rand(predictor_joint)

# Evaluate the logpdf of the joint posterior predictive distribution at this sample:

logpdf(predictor_joint, single_sample)

# Construct each of the marginal posterior predictive distributions:

predictor_marginal = MLJ.predict(mach, X)
typeof(predictor_marginal)

# `predictor_marginal` has one element for each row in `X`

size(predictor_marginal)

# Draw a single sample from each of the marginal posterior predictive distributions:

only.(rand.(predictor_marginal))
