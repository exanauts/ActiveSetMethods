using ActiveSetMethods
using PowerModels, JuMP, GLPK

function run_opf(data_file::String)
    network_data = PowerModels.parse_file(data_file)
    
    pm = instantiate_model(network_data, ACRPowerModel, PowerModels.build_opf)
    
    return optimize_model!(pm, optimizer = optimizer_with_attributes(
        ActiveSetMethods.Optimizer, 
        "external_optimizer" => GLPK.Optimizer()
    ))
end
