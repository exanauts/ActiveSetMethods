"""
Initialize variable values for ACPPowerModel by taking the mean of lower and upper bounds.
"""
function init_vars(pm::ACPPowerModel)
    init_branch_vars(pm)
    init_dc_vars(pm)
    init_gen_vars(pm)
    init_voltage_vars(pm)
end

"""
Initialize variable values for ACRPowerModel by taking the mean of lower and upper bounds.
"""
function init_vars(pm::ACRPowerModel)
    init_branch_vars(pm)
    init_dc_vars(pm)
    init_gen_vars(pm)
    init_voltage_vars(pm)
end

"""
Initialize variable values for IVRPowerModel by taking the mean of lower and upper bounds.
"""
function init_vars(pm::IVRPowerModel)
    # init_branch_vars(pm)
    # init_dc_vars(pm)
    # init_gen_vars(pm)
    # The above are expressions.

    # TODO: crg, cig, csr, csi, cr, ci, crdc, cidc
    init_voltage_vars(pm)
end

"""
Initialize variable values for ACPPowerModel from Ipopt solution.
"""
function init_vars_from_ipopt(pm::ACPPowerModel, pm2::ACPPowerModel)
    optimize_model!(pm2, optimizer = Ipopt.Optimizer)
    init_branch_vars(pm, pm2)
    init_dc_vars(pm, pm2)
    init_gen_vars(pm, pm2)
    init_voltage_vars(pm, pm2)
end

"""
Initialize variable values for ACRPowerModel from Ipopt solution.
"""
function init_vars_from_ipopt(pm::ACRPowerModel, pm2::ACRPowerModel)
    optimize_model!(pm2, optimizer = Ipopt.Optimizer)
    init_branch_vars(pm, pm2)
    init_dc_vars(pm, pm2)
    init_gen_vars(pm, pm2)
    init_voltage_vars(pm, pm2)
end

"""
Set initial variable value to JuMP, if the variable has both lower and upper bounds.
"""
function set_start_value(v::JuMP.VariableRef)
    if has_lower_bound(v) && has_upper_bound(v)
        JuMP.set_start_value(v, (upper_bound(v)+lower_bound(v))/2)
    end
end

"""
Initilize branch variable values
"""

function init_branch_vars(pm::AbstractPowerModel)
    for (l,i,j) in ref(pm,:arcs)
        set_start_value(var(pm,:p)[(l,i,j)])
        set_start_value(var(pm,:q)[(l,i,j)])
    end
end

function init_branch_vars(pm::AbstractPowerModel, pm_solved::AbstractPowerModel)
    for (l,i,j) in ref(pm,:arcs)
        JuMP.set_start_value(var(pm,:p)[(l,i,j)], JuMP.value(var(pm_solved,:p)[(l,i,j)]))
        JuMP.set_start_value(var(pm,:q)[(l,i,j)], JuMP.value(var(pm_solved,:q)[(l,i,j)]))
    end
end

"""
Initilize direct current branch variable values
"""

function init_dc_vars(pm::AbstractPowerModel)
    for arc in ref(pm,:arcs_dc)
        set_start_value(var(pm,:p_dc)[arc])
        set_start_value(var(pm,:q_dc)[arc])
    end
end

function init_dc_vars(pm::AbstractPowerModel, pm_solved::AbstractPowerModel)
    for arc in ref(pm,:arcs_dc)
        JuMP.set_start_value(var(pm,:p_dc)[arc], JuMP.value(var(pm_solved,:p_dc)[arc]))
        JuMP.set_start_value(var(pm,:q_dc)[arc], JuMP.value(var(pm_solved,:q_dc)[arc]))
    end
end

"""
Initilize generation variable values
"""

function init_gen_vars(pm::AbstractPowerModel)
    for (i,gen) in ref(pm,:gen)
        set_start_value(var(pm,:pg)[i])
        set_start_value(var(pm,:qg)[i])
    end
end

function init_gen_vars(pm::AbstractPowerModel, pm_solved::AbstractPowerModel)
    for (i,gen) in ref(pm,:gen)
        JuMP.set_start_value(var(pm,:pg)[i], JuMP.value(var(pm_solved,:pg)[i]))
        JuMP.set_start_value(var(pm,:qg)[i], JuMP.value(var(pm_solved,:qg)[i]))
    end
end

"""
Initilize voltage variable values
"""

function init_voltage_vars(pm::AbstractACPModel)
    for (i,bus) in ref(pm,:bus)
        set_start_value(var(pm,:va)[i])
        set_start_value(var(pm,:vm)[i])
    end
end

function init_voltage_vars(pm::AbstractACPModel, pm_solved::AbstractACPModel)
    for (i,bus) in ref(pm,:bus)
        JuMP.set_start_value(var(pm,:va)[i], JuMP.value(var(pm_solved,:va)[i]))
        JuMP.set_start_value(var(pm,:vm)[i], JuMP.value(var(pm_solved,:vm)[i]))
    end
end

function init_voltage_vars(pm::AbstractACRModel)
    for (i,bus) in ref(pm,:bus)
        set_start_value(var(pm,:vr)[i])
        set_start_value(var(pm,:vi)[i])
    end
end

function init_voltage_vars(pm::AbstractACRModel, pm_solved::AbstractACRModel)
    for (i,bus) in ref(pm,:bus)
        JuMP.set_start_value(var(pm,:vr)[i], JuMP.value(var(pm_solved,:vr)[i]))
        JuMP.set_start_value(var(pm,:vi)[i], JuMP.value(var(pm_solved,:vi)[i]))
    end
end
