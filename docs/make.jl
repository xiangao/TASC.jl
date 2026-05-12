using Pkg

Pkg.develop(PackageSpec(path = joinpath(@__DIR__, "..")))
Pkg.instantiate()

using Documenter
using TASC

DocMeta.setdocmeta!(TASC, :DocTestSetup, :(using TASC); recursive = true)

makedocs(
    sitename = "TASC.jl",
    modules = [TASC],
    format = Documenter.HTML(
        edit_link = nothing,
        repolink = "https://github.com/xiangao/TASC.jl",
    ),
    pages = [
        "Home" => "index.md",
        "Vignettes" => [
            "Getting Started" => "vignettes/01_getting_started.md",
            "Multiple Treated Units" => "vignettes/02_multiple_treated_units.md",
            "Preprocessing and Baselines" => "vignettes/03_preprocessing_baselines.md",
        ],
        "Reference" => "reference.md",
    ],
    warnonly = true,
    checkdocs = :none,
    remotes = nothing,
)

deploydocs(
    repo = "github.com/xiangao/TASC.jl.git",
    devbranch = "main",
    push_preview = false,
)
