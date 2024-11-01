# module DarcyDistributedHDGConvgTests

using Gridap
using Gridap.Geometry, Gridap.ReferenceFEs, Gridap.Arrays, Gridap.CellData, Gridap.FESpaces
using GridapHybrid
using GridapDistributed
using PartitionedArrays

u(x) = VectorValue(1+x[1],1+x[2])
Gridap.divergence(::typeof(u)) = (x) -> 2
p(x) = -3.14
∇p(x) = VectorValue(0,0)
Gridap.∇(::typeof(p)) = ∇p
f(x) = u(x) + ∇p(x)
# Normal component of u(x) on Neumann boundary
function g(x)
  tol=1.0e-14
  if (abs(x[2])<tol)
    return -x[2] #-x[1]-x[2]
  elseif (abs(x[2]-1.0)<tol)
    return x[2] # x[1]+x[2]
  end
  Gridap.Helpers.@check false
end

order = 1
np = (2,2)
# function solve_darcy_hdg((n,n), np, order)

  mesh_partition = (4,4)
  domain = (0,1,0,1)
  ranks = with_debug() do distribute
    distribute(LinearIndices((prod(np),)))
  end
  model = CartesianDiscreteModel(ranks,np,domain, mesh_partition)
  D  = num_cell_dims(model) 
  Ω = Triangulation(ReferenceFE{D},model)
  Γ = Triangulation(ReferenceFE{D-1},model)
  ∂K = GridapHybrid.Skeleton(model)

  
  reffeᵤ = ReferenceFE(lagrangian,VectorValue{D,Float64},order;space=:P)
  reffeₚ = ReferenceFE(lagrangian,Float64,order-1;space=:P)
  # reffeₚ = ReferenceFE(lagrangian,Float64,order;space=:P)
  reffeₗ = ReferenceFE(lagrangian,Float64,order;space=:P)

  # Define test FESpaces
  V = TestFESpace(Ω  , reffeᵤ; conformity=:L2)
  Q = TestFESpace(Ω  , reffeₚ; conformity=:L2)
  M = TestFESpace(Γ, reffeₗ; conformity=:L2,dirichlet_tags=collect(5:8))   
  Y = MultiFieldFESpace([V,Q,M])

  U = TrialFESpace(V)
  P = TrialFESpace(Q)
  L = TrialFESpace(M,p)
  X = MultiFieldFESpace([U, P, L])

  τ = 1.0 # HDG stab parameter

  degree = 2*(order+1)
  dΩ     = Measure(Ω,degree)
  n = get_cell_normal_vector(∂K)
  n0 = get_cell_owner_normal_vector(∂K)
  d∂K    = Measure(∂K,degree)

  
  a((uh,ph,lh),(vh,qh,mh),dΩ,d∂K) = ∫( vh⋅uh - (∇⋅vh)*ph - ∇(qh)⋅uh )dΩ +
                          ∫((vh⋅n)*lh)d∂K +
                          #∫(qh*(uh⋅n+τ*(ph-lh)*n⋅no))*d∂K
                          ∫(qh*(uh⋅n))d∂K +
                          ∫(τ*qh*ph*(n⋅n0))d∂K -
                          ∫(τ*qh*lh*(n⋅n0))d∂K +
                          #∫(mh*(uh⋅n+τ*(ph-lh)*n⋅no))*d∂K
                          ∫(mh*(uh⋅n))d∂K +
                          ∫(τ*mh*ph*(n⋅n0))d∂K -
                          ∫(τ*mh*lh*(n⋅n0))d∂K                            

  l((vh,qh,mh),dΩ) = ∫( vh⋅f + qh*(∇⋅u))*dΩ

  weakform = (u,v)->(a(u,v,dΩ,d∂K),l(v,dΩ))

  op = HybridAffineFEOperator(weakform, X, Y, [1,2], [3])
  @enter sol = solve(op) 


  function Gridap.FESpaces.solve!(uh::GridapDistributed.DistributedMultiFieldCellField, solver::LinearFESolver, op::HybridAffineFEOperator, cache)
    # Solve linear system defined on the skeleton
      lh = solve(op.skeleton_op)
  
    # Invoke weak form of the hybridizable system
      u = get_trial_fe_basis(op.trial)
      v = get_fe_basis(op.test)
      biform, liform  = op.weakform(u, v)
  
    # Transform DomainContribution objects of the hybridizable system into a
    # suitable form for assembling the linear system defined on the skeleton
    # (i.e., the hybrid system) and pair LHS and RHS terms associated to SkeletonTriangulation

      matvec, _, _ = map(local_views(biform), local_views(liform)) do biformi, liformi
        obiformi, oliformi = GridapHybrid._merge_bulk_and_skeleton_contributions(biformi, liformi)
        Gridap.FESpaces._pair_contribution_when_possible(obiformi, oliformi)
      end |> tuple_of_arrays
      
  
      free_dof_values = GridapHybrid._compute_hybridizable_from_skeleton_free_dof_values(
                      lh,
                      op.trial,
                      op.test,
                      Gridap.FESpaces.get_trial(op.skeleton_op),
                      matvec,
                      op.bulk_fields,
                      op.skeleton_fields)
  
      cache = nothing
      FEFunction(op.trial, free_dof_values), cache
  end 

  function GridapHybrid._compute_hybridizable_from_skeleton_free_dof_values(skeleton_fe_function::GridapDistributed.DistributedCellField,
    trial_hybridizable::GridapDistributed.DistributedMultiFieldFESpace,
    test_hybridizable::GridapDistributed.DistributedMultiFieldFESpace,
    trial_skeleton::GridapDistributed.DistributedSingleFieldFESpace,
    matvec::AbstractArray,
    bulk_fields,
    skeleton_fields)

    values = map(local_views(skeleton_fe_function), 
               local_views(trial_hybridizable), 
               local_views(test_hybridizable), 
               local_views(trial_skeleton), 
               local_views(matvec)) do lh, uh, vh, trial_sk_op, matvec

      GridapHybrid._compute_hybridizable_from_skeleton_free_dof_values(lh, uh, vh, trial_sk_op, matvec, bulk_fields, skeleton_fields)
    end
    return values
  end




#########################
function my_func(op::HybridAffineFEOperator)
  # Solve linear system defined on the skeleton
    lh = solve(op.skeleton_op)

  # Invoke weak form of the hybridizable system
    u = get_trial_fe_basis(op.trial)
    v = get_fe_basis(op.test)
    biform, liform  = op.weakform(u, v)

  # Transform DomainContribution objects of the hybridizable system into a
  # suitable form for assembling the linear system defined on the skeleton
  # (i.e., the hybrid system)
  #   obiform, oliform = _merge_bulk_and_skeleton_contributions(biform, liform)

  # # Pair LHS and RHS terms associated to SkeletonTriangulation
  #   matvec, _, _ = Gridap.FESpaces._pair_contribution_when_possible(obiform, oliform)

    matvec, _, _ = map(local_views(biform), local_views(liform)) do biformi, liformi
      obiformi, oliformi = GridapHybrid._merge_bulk_and_skeleton_contributions(biformi, liformi)
      Gridap.FESpaces._pair_contribution_when_possible(obiformi, oliformi)
    end |> tuple_of_arrays
    return matvec 

    # free_dof_values = GridapHybrid._compute_hybridizable_from_skeleton_free_dof_values(
    #                 lh,
    #                 op.trial,
    #                 op.test,
    #                 Gridap.FESpaces.get_trial(op.skeleton_op),
    #                 matvec,
    #                 op.bulk_fields,
    #                 op.skeleton_fields)

    # cache = nothing
    # FEFunction(op.trial, free_dof_values), cache
end 

# lh = solve(op.skeleton_op)  
# matvec = my_func(op)



  # lh1 = local_views(lh).items[1];
  # trial1 = local_views(op.trial).items[1];
  # test1 = local_views(op.test).items[1];
  # sk_trial1 =  local_views(Gridap.FESpaces.get_trial(op.skeleton_op)).items[1]
  # matvec1 = local_views(matvec).items[1]
  



  # @enter free_dof_vals = GridapHybrid._compute_hybridizable_from_skeleton_free_dof_values(lh1, trial1, test1, 
  #                                           sk_trial1, matvec1, op.bulk_fields, op.skeleton_fields)
  # # @enter free_dof_vals =GridapHybrid._compute_hybridizable_from_skeleton_free_dof_values(lh, op.trial, op.test, 
  # #                                   Gridap.FESpaces.get_trial(op.skeleton_op), matvec, op.bulk_fields, op.skeleton_fields)

  bulk_fields = op.bulk_fields
  skeleton_fields = op.skeleton_fields
  # Solve linear system defined on the skeleton
  lh = solve(op.skeleton_op)
  # Invoke weak form of the hybridizable system
    uu = get_trial_fe_basis(op.trial);
    vv = get_fe_basis(op.test);
    biform, liform  = op.weakform(uu, vv)


vecdata = map(local_views(biform), local_views(liform), local_views(lh), local_views(op.trial), local_views(op.test)) do biform, liform, skeleton_fe_function, trial_hybridizable, test_hybridizable

  # Transform DomainContribution objects of the hybridizable system into a
  # suitable form for assembling the linear system defined on the skeleton
  # (i.e., the hybrid system)
  obiform, oliform = GridapHybrid._merge_bulk_and_skeleton_contributions(biform, liform)

  # Pair LHS and RHS terms associated to SkeletonTriangulation
  matvec, _, _ = Gridap.FESpaces._pair_contribution_when_possible(obiform, oliform)

  # Convert dof-wise dof values of lh into cell-wise dof values lhₖ
  Γ = first(keys(matvec.dict))
  Gridap.Helpers.@check isa(Γ, GridapHybrid.SkeletonTriangulation) || isa(Γ, GridapHybrid.SkeletonView)
  lhₖ = get_cell_dof_values(skeleton_fe_function, Γ)

  # Compute cell-wise dof values of bulk fields out of lhₖ
  t = matvec.dict[Γ]
  m = BackwardStaticCondensationMap(bulk_fields, skeleton_fields)
  uhphlhₖ = lazy_map(m, t, lhₖ)

  model = get_background_model(Γ)
  cell_wise_facets_parent = GridapHybrid._get_cell_wise_facets(model)
  # cell_wise_facets = Gridap.Geometry.restrict(cell_wise_facets_parent, Γ1.cell_to_parent_cell) #
  cells_around_facets_parent = GridapHybrid._get_cells_around_facets(model)
  # cells_around_facets = Gridap.Geometry.restrict(cells_around_facets, cell_wise_facets[1]) #

  nfields = length(bulk_fields) + length(skeleton_fields)
  m = Gridap.Fields.BlockMap(nfields, skeleton_fields)
  L = Gridap.FESpaces.get_fe_space(skeleton_fe_function)
  lhₑ = lazy_map(m,
               GridapHybrid.convert_cell_wise_dofs_array_to_facet_dofs_array(
               cells_around_facets,
               cell_wise_facets,
               lhₖ,
               get_cell_dof_ids(L))...)

  lhₑ_dofs = Gridap.FESpaces.get_cell_dof_ids(trial_hybridizable, get_triangulation(L))
  lhₑ_dofs = lazy_map(m, lhₑ_dofs.args[skeleton_fields]...)
           
  Ω = trial_hybridizable[first(bulk_fields)]
  Ω = get_triangulation(Ω)
           
  m = Gridap.Fields.BlockMap(length(bulk_fields), bulk_fields)
  uhph_dofs = get_cell_dof_ids(trial_hybridizable, Ω)
  
  # This last step is needed as get_cell_dof_ids(...) returns as many blocks
  # as fields in trial, regardless of the FEspaces defined on Ω or not
  uhph_dofs = lazy_map(m, uhph_dofs.args[bulk_fields]...)
           
  uhphₖ = lazy_map(RestrictArrayBlockMap(bulk_fields), uhphlhₖ)             
  vecdata = ([lhₑ,uhphₖ], [lhₑ_dofs,uhph_dofs])
  return vecdata
end

assem = SparseMatrixAssembler(op.trial, op.test)
free_dof_values = assemble_vector(assem, vecdata)


biform1 = local_views(biform).items[1]
liform1 = local_views(liform).items[1]
lh1 = local_views(lh).items[1]
trial1 = local_views(op.trial).items[1]
test1 = local_views(op.test).items[1]

obiform1, oliform1 = GridapHybrid._merge_bulk_and_skeleton_contributions(biform1, liform1)
matvec1, _, _ = Gridap.FESpaces._pair_contribution_when_possible(obiform1, oliform1)

Γ1 = first(keys(matvec1.dict))
Gridap.Helpers.@check isa(Γ1, GridapHybrid.SkeletonTriangulation) || isa(Γ1, GridapHybrid.SkeletonView)
# lhₖ1 = get_cell_dof_values(lh1, Γ1.parent)
# datalhₖ1 = Gridap.Geometry.restrict(lhₖ1,Γ1.cell_to_parent_cell)
# compare = datalhₖ1 == get_cell_dof_values(lh1, Γ1)
lhₖ1 = get_cell_dof_values(lh1, Γ1) # No ghosts

t1 = matvec1.dict[Γ1]
m1 = BackwardStaticCondensationMap(op.bulk_fields, op.skeleton_fields)
uhphlhₖ1 = lazy_map(m1, t1, lhₖ1) #No ghosts 

model1 = get_background_model(Γ1)
cell_wise_facets = GridapHybrid._get_cell_wise_facets(model1) # with ghosts 
cell_wise_facets_1 = Table( Gridap.Geometry.restrict(cell_wise_facets, Γ1.cell_to_parent_cell) ) # without ghosts
cells_around_facets = GridapHybrid._get_cells_around_facets(model1) 

# Still has ghosts
cells_around_facets_1 = Table( Gridap.Geometry.restrict(cells_around_facets, unique(vcat(cell_wise_facets_1...))) )
manual = Table( [[1],[1,4],[1],[1,2],[2],[2,5],[2],[4],[4],[4,5],[5],[5]] )

nfields = length(bulk_fields) + length(skeleton_fields)
m = Gridap.Fields.BlockMap(nfields, skeleton_fields)
L = Gridap.FESpaces.get_fe_space(lh1)   # Ghosts
L_ids = get_cell_dof_ids(L) # Ghosts
L_ids_1 = Table( Gridap.Geometry.restrict(L_ids, unique(vcat(cell_wise_facets_1...))) ) # No ghosts

# have to restrict other three arguments, since lhₖ1 has no ghost info
manual_lhₑ = GridapHybrid.convert_cell_wise_dofs_array_to_facet_dofs_array(cells_around_facets,cell_wise_facets,lhₖ1,L_ids)

lhₑ = lazy_map(m,GridapHybrid.convert_cell_wise_dofs_array_to_facet_dofs_array(
                                                              cells_around_facets,cell_wise_facets,lhₖ1,get_cell_dof_ids(L))...)
datalhe = Gridap.Geometry.restrict(lhₑ,Γ1.cell_to_parent_cell)

lhₑ_dofs = Gridap.FESpaces.get_cell_dof_ids(trial1, get_triangulation(L))
lhₑ_dofs = lazy_map(m, lhₑ_dofs.args[skeleton_fields]...)
datalhe_dofs = Gridap.Geometry.restrict(lhₑ_dofs,Γ1.cell_to_parent_cell)
datalhe_dofs = Gridap.Geometry.restrict(lhₑ_dofs,cell_wise_facets_1[1])

Ω1 = trial1[first(bulk_fields)]
Ω1 = get_triangulation(Ω1)        

m = Gridap.Fields.BlockMap(length(bulk_fields), bulk_fields)
uhph_dofs = get_cell_dof_ids(trial1, Ω1)

uhph_dofs = lazy_map(m, uhph_dofs.args[bulk_fields]...)
uhphₖ = lazy_map(RestrictArrayBlockMap(bulk_fields), uhphlhₖ1)  # No ghosts
vecdata = ([datalhe,uhphₖ], [datalhe_dofs,uhph_dofs])



glue = GridapHybrid._generate_glue_among_facet_and_cell_wise_dofs_arrays(
   manual, cell_wise_facets_1, L_ids_1)