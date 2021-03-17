using PackageCompiler
create_sysimage([:ExaPF, :Ipopt, :MadNLP, :ProxAL]; sysimage_path="proxal.so", precompile_execution_file="test/distributed64.jl")