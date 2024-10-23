using Gridap
using Gridap.Geometry, Gridap.ReferenceFEs, Gridap.Arrays
using GridapHybrid
using GridapDistributed
using PartitionedArrays

using Gridap.CellData, Gridap.FESpaces

function GridapHybrid.Skeleton(model::GridapDistributed.DistributedDiscreteModel)
    cell_ids = get_cell_gids(model)
    trians = map(local_views(model),partition(cell_ids)) do model, indices
        Gridap.Geometry.TriangulationView(
            GridapHybrid.Skeleton(model),
            own_to_local(indices)
        )
    end
    return GridapDistributed.DistributedTriangulation(trians,model)
end

const SkeletonView{Dc,Dp} = Geometry.TriangulationView{Dc,Dp,<:GridapHybrid.SkeletonTriangulation}

function Geometry.is_change_possible(strian::GridapHybrid.SkeletonTriangulation,ttrian::SkeletonView)
    is_change_possible(strian,ttrian.parent)
end

function Geometry.is_change_possible(strian::SkeletonView,ttrian::GridapHybrid.SkeletonTriangulation)
    false
end

function CellData.change_domain(
    a::CellField,strian::GridapHybrid.SkeletonTriangulation,::ReferenceDomain,ttrian::SkeletonView,::ReferenceDomain
)
    @assert is_change_possible(strian,ttrian)
    b = change_domain(a,strian,ReferenceDomain(),ttrian.parent,ReferenceDomain())

    data = Geometry.restrict(CellData.get_data(b),ttrian.cell_to_parent_cell)
    CellData.similar_cell_field(b,data,ttrian,ReferenceDomain())
end

function FESpaces.get_cell_fe_data(
    fun, f, ttrian::SkeletonView
)
    data = FESpaces.get_cell_fe_data(fun,f,ttrian.parent)
    return Geometry.restrict(data,ttrian.cell_to_parent_cell)
end


np = (2,2)
ranks = with_debug() do distribute
    distribute(LinearIndices((prod(np),)))
end

# serial
domain = (0,1,0,1)
mesh_partition = (4,4)
model = CartesianDiscreteModel(ranks,np,domain, mesh_partition)
D  = num_cell_dims(model) 

Ω  = Triangulation(ReferenceFE{D}, model)
Γ  = Triangulation(ReferenceFE{D-1}, model)
∂K = GridapHybrid.Skeleton(model)

reffe = ReferenceFE(lagrangian,Float64,1)
V = TestFESpace(Ω  , reffe; conformity=:L2)
M = TestFESpace(Γ, reffe; conformity=:L2)  

d∂K    = Measure(∂K,4)

yh = get_fe_basis(V)
xh = get_trial_fe_basis(M)

c = ∫(xh⋅yh)d∂K

cell_dof_ids = map(get_cell_dof_ids,local_views(M),local_views(∂K))

∂K2 = GridapHybrid.Skeleton(model.models.items[1])
V1 = V.spaces.items[1]
M1 = M.spaces.items[1]
get_cell_dof_ids(M1,∂K2)

