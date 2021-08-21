using ActiveSetMethods
using PowerModels, JuMP, GLPK

PowerModels.silence()

build_acp(data_file::String) = instantiate_model(
    PowerModels.parse_file(data_file), 
    ACPPowerModel, 
    PowerModels.build_opf
)

function run_opf_ls(data_file::String, max_iter::Int = 100)
    pm = build_acp(data_file)
    result = optimize_model!(pm, optimizer = optimizer_with_attributes(
        ActiveSetMethods.Optimizer, 
        "OutputFlag" => 0,
        "external_optimizer" => GLPK.Optimizer,
        "max_iter" => max_iter,
        "algorithm" => "Line Search",
    ))
    return pm, result
end

function run_opf_tr(data_file::String, max_iter::Int = 100)
    pm = build_acp(data_file)
    result = optimize_model!(pm, optimizer = optimizer_with_attributes(
        ActiveSetMethods.Optimizer, 
        "OutputFlag" => 0,
        "external_optimizer" => GLPK.Optimizer,
        "max_iter" => max_iter,
        "algorithm" => "Trust Region",
    ))
    return pm, result
end
