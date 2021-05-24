module ExploringGridapHybridization
using Gridap
using FillArrays
include("GridapOverloads.jl")
include("CellBoundary.jl")
include("DensifyInnerMostBlockLevel.jl")

export DensifyInnerMostBlockLevel
export CellBoundary
export get_cell_owner_normal_vector
export get_cell_normal_vector
export quadrature_evaluation_points_and_weights
export restrict_to_cell_boundary
export integrate_low_level

end # module