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


````
@model X begin
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
) @ 1…29
````





Let's get some fake data to try it out.

````julia
X = rand(Cauchy(),(10,2))

truth = rand(m(X=X));

truth.β

truth.y
````


````
10-element Array{Float64,1}:
   -0.9882288099848306
   -0.1645079059069885
    1.6999215586425038
  -20.369301116899035
    1.1958221986820015
  -87.83189075161529
 -179.79529663598848
  -25.607316543021206
    4.886516042158956
   -2.473920435563424
````





Now we create and fit a machine:

````julia
mach = machine(sm, X, truth.y)
fit!(mach)
````





And now for prediciton! Note that this is a probabilistic model, so `MLJ.predict` gives us a vector of sampleable results:

````julia
pred = MLJ.predict(mach, matrix(X));
rand.(pred)
````


````
10-element Array{Float64,1}:
   -2.4054428368859515
    1.0361387807574978
    0.06290836450112702
  -19.660800378442826
    2.3426686374129426
  -85.447141898946
 -180.69453322501323
  -26.93577536015601
    4.427303382174472
   -3.4887102744424534
````





And here's another sample:

````julia
rand.(pred)
````


````
10-element Array{Float64,1}:
   -3.0253286002259387
    0.3366748413835682
   -0.9789079378071528
  -21.038826107529996
    1.112538996959603
  -86.82757737272571
 -180.6248444095373
  -27.775349834015337
    5.29664907487426
   -1.7183563384729488
````


