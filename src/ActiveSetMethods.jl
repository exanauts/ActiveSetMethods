module ActiveSetMethods

using LinearAlgebra
using SparseArrays
import MathOptInterface

const MOI = MathOptInterface
const MOIU = MathOptInterface.Utilities

include("status.jl")
include("parameters.jl")
include("model.jl")

include("SLP/SLP.jl")

include("MOI_wrapper.jl")

end
