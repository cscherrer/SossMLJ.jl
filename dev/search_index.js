var documenterSearchIndex = {"docs":
[{"location":"api/#API","page":"API","title":"API","text":"","category":"section"},{"location":"api/","page":"API","title":"API","text":"","category":"page"},{"location":"api/","page":"API","title":"API","text":"Modules = [SossMLJ]","category":"page"},{"location":"example-bayesian-linear-regression/","page":"Bayesian linear regression","title":"Bayesian linear regression","text":"EditURL = \"https://github.com/cscherrer/SossMLJ.jl/blob/master/examples/example-bayesian-linear-regression.jl\"","category":"page"},{"location":"example-bayesian-linear-regression/#Bayesian-linear-regression","page":"Bayesian linear regression","title":"Bayesian linear regression","text":"","category":"section"},{"location":"example-bayesian-linear-regression/","page":"Bayesian linear regression","title":"Bayesian linear regression","text":"Import the necessary packages:","category":"page"},{"location":"example-bayesian-linear-regression/","page":"Bayesian linear regression","title":"Bayesian linear regression","text":"using Distributions\nusing MLJ\nusing MLJBase\nusing MLJModelInterface\nusing Soss\nusing SossMLJ\nusing Statistics","category":"page"},{"location":"example-bayesian-linear-regression/","page":"Bayesian linear regression","title":"Bayesian linear regression","text":"Use the Soss probabilistic programming language to define a Bayesian linear regression model:","category":"page"},{"location":"example-bayesian-linear-regression/","page":"Bayesian linear regression","title":"Bayesian linear regression","text":"m = @model X,α,σ begin\n    k = size(X,2)\n    β ~ Normal(0,α) |> iid(k)\n    yhat = X * β\n    y ~ For(eachindex(yhat)) do j\n        Normal(yhat[j], σ)\n    end\nend","category":"page"},{"location":"example-bayesian-linear-regression/","page":"Bayesian linear regression","title":"Bayesian linear regression","text":"Generate some synthetic features:","category":"page"},{"location":"example-bayesian-linear-regression/","page":"Bayesian linear regression","title":"Bayesian linear regression","text":"num_rows = 100\nX = (a=randn(num_rows), b=randn(num_rows), c=randn(num_rows))","category":"page"},{"location":"example-bayesian-linear-regression/","page":"Bayesian linear regression","title":"Bayesian linear regression","text":"Define the hyperparameters of our prior distributions:","category":"page"},{"location":"example-bayesian-linear-regression/","page":"Bayesian linear regression","title":"Bayesian linear regression","text":"hyperparameters = (α=2.0, σ=1.0)","category":"page"},{"location":"example-bayesian-linear-regression/","page":"Bayesian linear regression","title":"Bayesian linear regression","text":"Convert the Soss model into a SossMLJModel:","category":"page"},{"location":"example-bayesian-linear-regression/","page":"Bayesian linear regression","title":"Bayesian linear regression","text":"model = SossMLJModel(hyperparameters, X -> (X=MLJModelInterface.matrix(X),), m, dynamicHMC)","category":"page"},{"location":"example-bayesian-linear-regression/","page":"Bayesian linear regression","title":"Bayesian linear regression","text":"Generate some synthetic labels:","category":"page"},{"location":"example-bayesian-linear-regression/","page":"Bayesian linear regression","title":"Bayesian linear regression","text":"args = merge(model.transform(X), hyperparameters)\ntruth = rand(m(args))\ny = truth.y","category":"page"},{"location":"example-bayesian-linear-regression/","page":"Bayesian linear regression","title":"Bayesian linear regression","text":"Create an MLJ machine for fitting our model:","category":"page"},{"location":"example-bayesian-linear-regression/","page":"Bayesian linear regression","title":"Bayesian linear regression","text":"mach = machine(model, X, y)","category":"page"},{"location":"example-bayesian-linear-regression/","page":"Bayesian linear regression","title":"Bayesian linear regression","text":"Fit the model:","category":"page"},{"location":"example-bayesian-linear-regression/","page":"Bayesian linear regression","title":"Bayesian linear regression","text":"fit!(mach)","category":"page"},{"location":"example-bayesian-linear-regression/","page":"Bayesian linear regression","title":"Bayesian linear regression","text":"Construct the joint posterior distribution and the joint posterior predictive distribution:","category":"page"},{"location":"example-bayesian-linear-regression/","page":"Bayesian linear regression","title":"Bayesian linear regression","text":"##predictor_joint = MLJ.predict_joint(mach, X)\npredictor_joint = SossMLJ.predict_joint(mach, X)\ntypeof(predictor_joint)","category":"page"},{"location":"example-bayesian-linear-regression/","page":"Bayesian linear regression","title":"Bayesian linear regression","text":"Compare the joint posterior distribution to the true parameter values:","category":"page"},{"location":"example-bayesian-linear-regression/","page":"Bayesian linear regression","title":"Bayesian linear regression","text":"truth.β - predict_particles(predictor_joint, X).β","category":"page"},{"location":"example-bayesian-linear-regression/","page":"Bayesian linear regression","title":"Bayesian linear regression","text":"Compare the joint posterior predictive distribution to the true labels:","category":"page"},{"location":"example-bayesian-linear-regression/","page":"Bayesian linear regression","title":"Bayesian linear regression","text":"truth.yhat - predict_particles(predictor_joint, X).yhat","category":"page"},{"location":"example-bayesian-linear-regression/","page":"Bayesian linear regression","title":"Bayesian linear regression","text":"Draw a single sample from the joint posterior predictive distribution:","category":"page"},{"location":"example-bayesian-linear-regression/","page":"Bayesian linear regression","title":"Bayesian linear regression","text":"s = rand(predictor_joint)","category":"page"},{"location":"example-bayesian-linear-regression/","page":"Bayesian linear regression","title":"Bayesian linear regression","text":"Evaluate the logpdf of the joint posterior predictive distribution at our sample:","category":"page"},{"location":"example-bayesian-linear-regression/","page":"Bayesian linear regression","title":"Bayesian linear regression","text":"logpdf(predictor_joint, s)","category":"page"},{"location":"example-bayesian-linear-regression/","page":"Bayesian linear regression","title":"Bayesian linear regression","text":"Construct each of the marginal posterior predictive distributions:","category":"page"},{"location":"example-bayesian-linear-regression/","page":"Bayesian linear regression","title":"Bayesian linear regression","text":"predictor_marginal = MLJ.predict(mach, X)\ntypeof(predictor_marginal)","category":"page"},{"location":"example-bayesian-linear-regression/","page":"Bayesian linear regression","title":"Bayesian linear regression","text":"predictor_marginal has one element for each row in X","category":"page"},{"location":"example-bayesian-linear-regression/","page":"Bayesian linear regression","title":"Bayesian linear regression","text":"size(predictor_marginal)","category":"page"},{"location":"example-bayesian-linear-regression/","page":"Bayesian linear regression","title":"Bayesian linear regression","text":"Draw a single sample from each of the marginal posterior predictive distributions:","category":"page"},{"location":"example-bayesian-linear-regression/","page":"Bayesian linear regression","title":"Bayesian linear regression","text":"only.(rand.(predictor_marginal))","category":"page"},{"location":"example-bayesian-linear-regression/","page":"Bayesian linear regression","title":"Bayesian linear regression","text":"","category":"page"},{"location":"example-bayesian-linear-regression/","page":"Bayesian linear regression","title":"Bayesian linear regression","text":"This page was generated using Literate.jl.","category":"page"},{"location":"#SossMLJ.jl","page":"Home","title":"SossMLJ.jl","text":"","category":"section"},{"location":"","page":"Home","title":"Home","text":"SossMLJ integrates the Soss probabilistic programming library into the MLJ machine learning framework.","category":"page"}]
}
