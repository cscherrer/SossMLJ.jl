using SossMLJ
using Documenter

import Literate

include(joinpath(dirname(dirname(@__FILE__)), "test", "examples-list.jl"))

pages_before_examples = [
    "Home" => "index.md",
]
pages_examples = ["Examples" => ["$(example[1])" => "example-$(example[2]).md" for example in EXAMPLES]]
pages_after_examples = [
    "API" => "api.md",
]
pages = vcat(
    pages_before_examples,
    pages_examples,
    pages_after_examples,
)

# Use Literate.jl to generate Markdown files for each of the examples
for example in EXAMPLES
    input_file = joinpath(EXAMPLESROOT, "example-$(example[2]).jl")
    Literate.markdown(input_file, DOCSOURCE)
end

makedocs(;
    modules=[SossMLJ],
    format=Documenter.HTML(),
    pages=pages,
    repo="https://github.com/cscherrer/SossMLJ.jl/blob/{commit}{path}#L{line}",
    sitename="SossMLJ.jl",
    authors="Chad Scherrer, Thibaut Lienart, Dilum Aluthge, and contributors",
    strict=true,
)

deploydocs(;
    repo="github.com/cscherrer/SossMLJ.jl",
)
