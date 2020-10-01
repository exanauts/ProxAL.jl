using Documenter, ProxAL

makedocs(
    sitename = "ProxAL.jl",
    format = Documenter.HTML(
        prettyurls = Base.get(ENV, "CI", nothing) == "true",
        mathengine = Documenter.KaTeX()
    ),
    strict = true,
    pages = [
        "Home" => "index.md",
    ]
)

