# This file pulls the dependencies using git branches for development

using Pkg
exapfspec = PackageSpec(url="https://github.com/exanauts/ExaPF.jl.git", rev="proxal")
exatronspec = PackageSpec(url="https://github.com/exanauts/ExaTron.jl.git", rev="ms/ka")
Pkg.add([exapfspec, exatronspec])
