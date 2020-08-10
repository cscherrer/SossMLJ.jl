using Documenter, SossMLJ

makedocs(;
    modules=[SossMLJ],
    format=Documenter.HTML(),
    pages=[
        "Home" => "index.md",
    ],
    repo="https://github.com/cscherrer/SossMLJ.jl/blob/{commit}{path}#L{line}",
    sitename="SossMLJ.jl",
    authors="Chad Scherrer",
    assets=String[],
    strict=true,
)

deploydocs(;
    repo="github.com/cscherrer/SossMLJ.jl",
)
