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

  mesh_partition = (2,2)
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
  M = TestFESpace(Γ, reffeₗ; conformity=:L2)   
  Y = MultiFieldFESpace([V,Q,M])

  U = TrialFESpace(V)
  P = TrialFESpace(Q)
  L = TrialFESpace(M)
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

  # @enter xh = solve(op)

  lh = solve(op.skeleton_op)


  # Cross checking skeleton_op
  A = op.skeleton_op.op.matrix
  b = op.skeleton_op.op.vector

  As = PartitionedArrays.to_trivial_partition(A)

  # uh = get_trial_fe_basis(op.trial);
  # vh = get_fe_basis(op.test);

  # biform, liform = weakform(uh, vh)

  # matvec, _, _ = map(local_views(biform), local_views(liform)) do biformi, liformi
  #   obiformi, oliformi = GridapHybrid._merge_bulk_and_skeleton_contributions(biformi, liformi)
  #   Gridap.FESpaces._pair_contribution_when_possible(biformi, oliformi)
  # end |> tuple_of_arrays;

  # free_dof_values = map(local_views(matvec), local_views(lh), local_views(op.test), local_views(op.trial), local_views(Gridap.FESpaces.get_trial(op.skeleton_op)))  do matvec, lh, vh, uh, trial_sk_op

  #   GridapHybrid._compute_hybridizable_from_skeleton_free_dof_values(lh, uh, vh, trial_sk_op, matvec, 
  #                                                                                    op.bulk_fields, op.skeleton_fields)

  # end

  # matvec_1 = local_views(matvec).items[1]; 
  # lh_1 = local_views(lh).items[1]; 
  # optest_1 = local_views(op.test).items[1]; 
  # optrial_1 = local_views(op.trial).items[1]; 
  # trial_sk_op_1 = local_views(Gridap.FESpaces.get_trial(op.skeleton_op)).items[1];

  # Γ = first(keys(matvec_1.dict))
  # Gridap.Helpers.@check isa(Γ, SkeletonTriangulation) || isa(Γ, GridapHybrid.SkeletonView)
  # lhₖ = get_cell_dof_values(lh_1, Γ)
  # t = matvec_1.dict[Γ]
  # m = BackwardStaticCondensationMap(op.bulk_fields, op.skeleton_fields)
  # @enter uhphlhₖ = lazy_map(m, t, lhₖ)