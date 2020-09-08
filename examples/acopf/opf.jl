using ActiveSetMethods
using PowerModels, JuMP, GLPK

function run_opf(data_file::String, 
                 model = ACRPowerModel,
                 build_function = PowerModels.build_opf)
    network_data = PowerModels.parse_file(data_file)
    pm = instantiate_model(network_data, model, build_function)
    return optimize_model!(pm, optimizer = optimizer_with_attributes(
        ActiveSetMethods.Optimizer, 
        "external_optimizer" => GLPK.Optimizer()
    ))
end
