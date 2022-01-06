using Pkg
exapfspec = PackageSpec(url="https://github.com/exanauts/ExaPF.jl.git", rev="proxal")
exatronspec = PackageSpec(url="https://github.com/exanauts/ExaTron.jl.git", rev="release")
proxalspec = PackageSpec(path=pwd())
Pkg.add([exatronspec,proxalspec])
