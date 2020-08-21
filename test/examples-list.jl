const PACKAGE_ROOT = dirname(dirname(@__FILE__))
const DOCROOT = joinpath(PACKAGE_ROOT, "docs")
const DOCSOURCE = joinpath(DOCROOT, "src")
const EXAMPLESROOT = joinpath(PACKAGE_ROOT, "examples")
const EXAMPLES = [
    ("Linear regression", "linear-regression"),
    ]
