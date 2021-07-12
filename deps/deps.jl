# This file pulls the dependencies using git branches for development

using Pkg
exapfspec = PackageSpec(url="https://github.com/exanauts/ExaPF.jl.git", rev="develop")
exatronspec = PackageSpec(url="https://github.com/exanauts/ExaTron.jl.git", rev="fp/proxal")
Pkg.add([exapfspec, exatronspec])
