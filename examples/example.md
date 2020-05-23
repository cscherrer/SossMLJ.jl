Loading some things

````julia
import MLJModelInterface: matrix
using MLJ
using SossMLJ
using DataFrames
using Soss
````





Now we make a Soss model

````julia
m = @model X begin
    β ~ Cauchy() |> iid(2)
    yhat = X * β
    y ~ For(length(yhat)) do j
        Normal(yhat[j], 1)
    end
end
````





next we wrap this in a `SossModel` (note no space), which connects us to MLJ.

````julia
sm = SossModel(model=m)
````


````
SossModel(
    model = @model X begin
        β ~ Cauchy() |> iid(2)
        yhat = X * β
        y ~ For(length(yhat)) do j
                Normal(yhat[j], 1)
            end
    end
) @ 6…74
````





Let's get some fake data to try it out.

````julia
X = rand(Cauchy(),(10,2))

truth = rand(m(X=X))
````





This gives us the parameter value

````julia
truth.β
````


````
2-element Array{Float64,1}:
 -1.0093507723803745
  0.26905501971718504
````





and response vector

````julia
truth.y
````


````
10-element Array{Float64,1}:
 -0.008925867470963439
 -1.17041904226459
 -3.8772252420759603
 -5.542658774416509
 -2.3563333581263595
 -1.0434536997791046
 -1.6949874762720023
  6.867722415072275
 -0.08036640723198274
  0.2690492402379255
````





Now we create and fit a machine:

````julia
mach = machine(sm, X, truth.y)
fit!(mach)
````





And now for prediciton! Note that this is a probabilistic model, so `MLJ.predict` gives us a vector of sampleable results:

````julia
pred = MLJ.predict(mach, X);
rand.(pred)
````


````
10-element Array{Float64,1}:
 -0.5823962805383203
 -1.1542594285703425
 -2.943776540184552
 -5.802540120444263
 -1.1939557093059376
 -1.5626238910765875
 -1.5075390211787794
  8.505279016002367
 -1.3359159166489252
  0.7515948639373163
````





And here's another sample:

````julia
rand.(pred)
````


````
10-element Array{Float64,1}:
  0.4545035944560679
 -2.2704545332964097
 -2.454234692816206
 -6.112258677392331
 -2.3740500951156926
 -2.8632497901149625
 -2.2871388130056527
  6.391811461427609
 -2.593058134863292
  0.6388242551678939
````


