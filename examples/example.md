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
) @ 1…95
````





Let's get some fake data to try it out.

````julia
X = rand(Cauchy(),(10,2))

truth = rand(m(X=X));
````


````
(X = [-1.0935808982919206 -0.9106575080882054; -0.03222890722365356 1.25492
02722317525; … ; 1.4389934137398537 -5.024190574012656; -1.940062690859753 
-0.23554645888602924],
 yhat = [10.660697191332511, 21.360162265804334, -220.24117831164722, 344.5
136209491732, -48.43145019803801, 84.28074636948237, -37.84475883680752, 9.
523349548789767, -116.19872874865423, 41.57166769870845],
 β = [-23.421537268319504, 16.41961817821605],
 y = [9.999316333225192, 21.699419981125132, -219.77585014016103, 345.47438
14713501, -48.658921879725185, 85.64130474440258, -38.06117786081072, 7.577
9726016304405, -116.92041156109336, 40.24904737235539],)
````





This gives us the parameter value

````julia
truth.β
````


````
2-element Array{Float64,1}:
 -23.421537268319504
  16.41961817821605
````





and response vector

````julia
truth.y
````


````
10-element Array{Float64,1}:
    9.999316333225192
   21.699419981125132
 -219.77585014016103
  345.4743814713501
  -48.658921879725185
   85.64130474440258
  -38.06117786081072
    7.5779726016304405
 -116.92041156109336
   40.24904737235539
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
    9.37795322391339
   22.291496744110958
 -221.65345185651594
  344.22750884526147
  -49.8163601065379
   84.6788200597252
  -38.10298148316002
    7.051576898024532
 -116.97633520180851
   42.46493918784828
````





And here's another sample:

````julia
rand.(pred)
````


````
10-element Array{Float64,1}:
   10.701634305903841
   22.245086752037817
 -221.20896519010356
  344.2295183796381
  -46.115831235773555
   86.09877562022572
  -37.830834958176304
    8.50036371719223
 -116.11158017683363
   41.51638676553269
````


