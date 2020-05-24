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
sm = SossMLJModel(model=m)
````


````
SossMLJModel(
    model = @model X begin
        β ~ Cauchy() |> iid(2)
        yhat = X * β
        y ~ For(length(yhat)) do j
                Normal(yhat[j], 1)
            end
    end
) @ 8…40
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
 -1.052842470618594
 -0.32313815196765355
````





and response vector

````julia
truth.y
````


````
10-element Array{Float64,1}:
   1.2194835627473744
  -2.7412005432140303
  12.002611085834218
 -19.114830122007955
   1.93443283873503
   6.2560828342743635
  -2.089943963691071
  -2.09189964854799
  -0.040910192346357066
  -4.082931394487771
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
  -0.5488955582061885
  -1.7413085973070443
  12.575938782453635
 -21.056442299630447
  -0.22472449435620967
   6.185157479019461
   1.5513817518802753
  -1.6457066581985291
  -0.4375283227299232
  -3.3484128144034386
````





And here's another sample:

````julia
rand.(pred)
````


````
10-element Array{Float64,1}:
   2.24061225928648
   0.2288893786733307
  14.098576466576073
 -18.67517658825167
   1.5762551439764534
   5.767273453937659
  -0.6325337875785283
  -2.614635269810888
   1.0779920279126034
  -2.53387758478215
````


