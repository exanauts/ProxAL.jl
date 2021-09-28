# This file pulls the dependencies using git branches for development
using LibGit2
repo_url = "https://github.com/exanauts/ExaData.git"
if !isdir("ExaData")
	repo = LibGit2.clone(repo_url, "ExaData")
end

using Pkg
exapfspec = PackageSpec(url="https://github.com/exanauts/ExaPF.jl.git", rev="proxal")
exatronspec = PackageSpec(url="https://github.com/exanauts/ExaTron.jl.git", rev="fp/proxal")
Pkg.add([exapfspec, exatronspec])
