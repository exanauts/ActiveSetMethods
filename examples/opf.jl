using ActiveSetMethods
using PowerModels, JuMP, GLPK

Options_["LP_solver"] = GLPK.Optimizer

function run_opf(data_file::String)
    network_data = PowerModels.parse_file(data_file)
    
    pm = instantiate_model(network_data, ACPPowerModel, PowerModels.build_opf)
    
    return optimize_model!(pm, optimizer = ActiveSetMethods.Optimizer)
end
