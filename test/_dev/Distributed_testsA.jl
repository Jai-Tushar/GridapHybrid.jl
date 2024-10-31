using Gridap
using Gridap.Geometry, Gridap.ReferenceFEs, Gridap.Arrays, Gridap.CellData, Gridap.FESpaces
using GridapHybrid

using GridapDistributed
using PartitionedArrays

np = (2,2)
ranks = with_debug() do distribute
    distribute(LinearIndices((prod(np),)))
end

domain = (0,1,0,1)
mesh_partition = (4,4)
Dmodel = CartesianDiscreteModel(ranks,np,domain, mesh_partition)
D  = num_cell_dims(Dmodel) 

DΩ  = Triangulation(ReferenceFE{D}, Dmodel)
DΓ  = Triangulation(ReferenceFE{D-1}, Dmodel)
D∂K = GridapHybrid.Skeleton(Dmodel)

order = 1
dDΩ  = Measure(DΩ,2*order+1) 
dD∂K = Measure(D∂K,2*order+1)
reffe = ReferenceFE(lagrangian,Float64,order; space=:P)
DVK  = TestFESpace(DΩ, reffe; conformity=:L2)
DV∂K = TestFESpace(DΓ, reffe; conformity=:L2)  

yh = get_fe_basis(DVK)
xh = get_trial_fe_basis(DV∂K)

# const SkeletonView{Dc,Dp} = Gridap.Geometry.TriangulationView{Dc,Dp,<:GridapHybrid.SkeletonTriangulation}

# function Gridap.Geometry.is_change_possible(strian::GridapHybrid.SkeletonTriangulation,ttrian::SkeletonView)
#     is_change_possible(strian,ttrian.parent)
# end

# function Gridap.Geometry.is_change_possible(strian::SkeletonView,ttrian::Gridap.Geometry.Triangulation)
#     false
# end

# function Gridap.CellData.change_domain(
#     a::CellField,strian::GridapHybrid.SkeletonTriangulation,::ReferenceDomain,ttrian::SkeletonView,::ReferenceDomain
# )
#     @assert is_change_possible(strian,ttrian)
#     b = change_domain(a,strian,ReferenceDomain(),ttrian.parent,ReferenceDomain())

#     data = Gridap.Geometry.restrict(CellData.get_data(b),ttrian.cell_to_parent_cell)
#     Gridap.CellData.similar_cell_field(b,data,ttrian,ReferenceDomain())
# end

# function Gridap.FESpaces.get_cell_fe_data(
#     fun, f, ttrian::SkeletonView
# )
#     data = Gridap.FESpaces.get_cell_fe_data(fun,f,ttrian.parent)
#     return Gridap.Geometry.restrict(data,ttrian.cell_to_parent_cell)
# end


Dc = ∫(xh⋅yh)dD∂K
# cf = xh ⋅ yh
# cf_1 = local_views(cf).items[1]
# cm_1 = local_views(dD∂K).items[1]
# ct_1 = local_views(D∂K).items[1]

# @enter integrate(cf_1,cm_1)

# is_change_possible(cf_1.trian,ct_1)
# b = change_domain(cf,cf_1.trian,ReferenceDomain(),ct_1.parent,ReferenceDomain())
# data = Gridap.Geometry.restrict(CellData.get_data(b),ttrian.cell_to_parent_cell)
# b = Gridap.CellData.change_domain(cf_1,cf_1.trian,ReferenceDomain(),ct_1,ReferenceDomain())



# DVKids = partition(get_free_dof_ids(DVK))
# LtoO = map(local_to_owner, DVKids)
# LtoG = map(local_to_global, DVKids)

# DV∂Kids = partition(get_free_dof_ids(DV∂K))
# LtoO = map(local_to_owner, DV∂Kids)
# LtoG = map(local_to_global, DV∂Kids)


# DVK_ids  = map(get_cell_dof_ids,local_views(DVK), local_views(D∂K)) 
# DV∂K_ids = map(get_cell_dof_ids,local_views(DV∂K),local_views(D∂K))

DV = MultiFieldFESpace([DVK,DV∂K])
DV_ids = map(get_cell_dof_ids,local_views(DV), local_views(D∂K))

# function Gridap.Geometry.is_change_possible(strian::SkeletonView,ttrian::GridapHybrid.SkeletonTriangulation)
#     false
# end

# function Gridap.Geometry.is_change_possible(strian::BodyFittedTriangulation,ttrian::SkeletonView)
#     is_change_possible(strian,ttrian.parent)
# end


vh = get_fe_basis(DV);
uh = get_trial_fe_basis(DV);

(vt, vf) = vh;
(ut, uf) = uh;

op0_cf = vt * uf
op0 = ∫(op0_cf)dD∂K
# op0_cf_1 = local_views(op0_cf).items[1]
# dD∂K_1 = local_views(dD∂K).items[1]
# is_change_possible(get_triangulation(op0_cf_1), D∂K_1)
# Gridap.CellData.change_domain( op0_cf_1,get_triangulation(op0_cf_1),ReferenceDomain(),D∂K_1,ReferenceDomain())

op1_cf = vt ⋅ ut
op1_a  = ∫(op1_cf)dDΩ

f_2 = local_views(op1_cf).items[1]
m_2 = local_views(dDΩ).items[1]
# @enter integrate(f_2,m_2)

op2_cf = ∇(vt) ⋅ ∇(ut)
op2    = ∫( ∇(vt) ⋅ ∇(ut) )dDΩ  

# function Gridap.CellData.change_domain(
#     a::Gridap.MultiField.MultiFieldFEBasisComponent,
#     ttrian::SkeletonView,
#     tdomain::DomainStyle)
#     if get_triangulation(a) == ttrian # SkeletonView to SkeletonView, return Idem
#         return a
#     end
#     b = change_domain(a, ttrian.parent, tdomain)
#     b_sf = b.single_field
#     b_data = Gridap.Geometry.restrict(CellData.get_data(b_sf),ttrian.cell_to_parent_cell)
#     c_sf = Gridap.CellData.similar_cell_field(b_sf,b_data,ttrian,ReferenceDomain())
#     c_data = Gridap.Geometry.restrict(CellData.get_data(b),ttrian.cell_to_parent_cell)
#     return Gridap.MultiField.MultiFieldFEBasisComponent(c_data,c_sf,a.fieldid,a.nfields)
# end




op3_cf = vf * uf
op3 = ∫(op3_cf)dD∂K


# pts = get_cell_points(t_1)

# f_1 = local_views(op3_cf).items[1]
# m_1 = local_views(dD∂K).items[1]
# t_1 = local_views(D∂K).items[1]
# # @enter integrate(f_1, m_1)
# integrate(f_1,m_1)

# f_2 = local_views(op1_cf).items[1]
# @enter integrate(f_1, m_1)


