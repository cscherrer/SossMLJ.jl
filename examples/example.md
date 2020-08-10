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
,
    infer = Soss.dynamicHMC) @ 4…19
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
 0.7795558910334254
 1.3205090289240893
````





and response vector

````julia
truth.y
````


````
10-element Array{Float64,1}:
 -0.009656286618348886
  0.9379274012136625
 -1.0765641894263076
 -4.884093727697979
  0.44322739533204614
  3.3290688788206877
  9.804025505635714
 -0.6967895342106162
 -3.249175636559466
  0.3924770024728915
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
  0.6478518484624256
 -0.09595465176171525
 -2.4607605070347063
 -6.263126184478209
  1.0491468939935231
  2.0356523077028923
  9.115481356543857
 -1.3853144490084879
 -3.8689070474959615
  0.6793707061491905
````





And here's another sample:

````julia
rand.(pred)
````


````
10-element Array{Float64,1}:
 -0.1191636821334261
  0.5832262026481227
 -3.5828613320656912
 -6.423922375749305
 -1.4119001531626445
  2.3152820204955673
 10.719893825466206
  0.8946705351309647
 -4.306828782379904
 -1.4862297043387542
````


