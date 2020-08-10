import MLJModelInterface: matrix
using MLJ
using SossMLJ
using DataFrames
using Soss  

m = @model X begin
    β ~ Cauchy() |> iid(2)
    yhat = X * β
    y ~ For(length(yhat)) do j
        Normal(yhat[j], 1)
    end
end

sm = SossModel(model=m)

X = rand(Cauchy(),(10,2))

truth = rand(m(X=X));

truth.β

truth.y

mach = machine(sm, X, truth.y)
fit!(mach)

pred = MLJ.predict(mach, matrix(X));

rand.(pred)
