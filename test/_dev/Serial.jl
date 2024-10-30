using Test
using Gridap
using FillArrays
using Gridap.Geometry
using GridapHybrid

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

partition = (0,1,0,1)
cells = (2,2)
model = CartesianDiscreteModel(partition,cells)
order=1

    # Geometry
    D = num_cell_dims(model)
    Ω = Triangulation(ReferenceFE{D},model)
    Γ = Triangulation(ReferenceFE{D-1},model)
    ∂K = GridapHybrid.Skeleton(model)

    # Reference FEs
    order  = 1
    reffeᵤ = ReferenceFE(lagrangian,VectorValue{D,Float64},order;space=:P)
    reffeₚ = ReferenceFE(lagrangian,Float64,order-1;space=:P)
    reffeₗ = ReferenceFE(lagrangian,Float64,order;space=:P)

    # Define test FESpaces
    V = TestFESpace(Ω  , reffeᵤ; conformity=:L2)
    Q = TestFESpace(Ω  , reffeₚ; conformity=:L2)
    M = TestFESpace(Γ,
                    reffeₗ;
                    conformity=:L2,
                    dirichlet_tags=collect(5:8))
    Y = MultiFieldFESpace([V,Q,M])

    # Define trial FEspaces
    U = TrialFESpace(V)
    P = TrialFESpace(Q)
    L = TrialFESpace(M,p)
    X = MultiFieldFESpace([U, P, L])

    # FE formulation params
    τ = 1.0 # HDG stab parameter

    degree = 2*(order+1)
    dΩ     = Measure(Ω,degree)
    n      = get_cell_normal_vector(∂K)
    nₒ     = get_cell_owner_normal_vector(∂K)
    d∂K    = Measure(∂K,degree)

    yh = get_fe_basis(Y)
    xh = get_trial_fe_basis(X)

    (uh,ph,lh) = xh
    (vh,qh,mh) = yh

    a((uh,ph,lh),(vh,qh,mh)) = ∫( vh⋅uh - (∇⋅vh)*ph - ∇(qh)⋅uh )dΩ +
                              ∫((vh⋅n)*lh)d∂K +
                              #∫(qh*(uh⋅n+τ*(ph-lh)*n⋅no))*d∂K
                              ∫(qh*(uh⋅n))d∂K +
                              ∫(τ*qh*ph*(n⋅nₒ))d∂K -
                              ∫(τ*qh*lh*(n⋅nₒ))d∂K +
                              #∫(mh*(uh⋅n+τ*(ph-lh)*n⋅no))*d∂K
                              ∫(mh*(uh⋅n))d∂K +
                              ∫(τ*mh*ph*(n⋅nₒ))d∂K -
                              ∫(τ*mh*lh*(n⋅nₒ))d∂K
    l((vh,qh,mh)) = ∫( vh⋅f + qh*(∇⋅u))*dΩ

    op=HybridAffineFEOperator((u,v)->(a(u,v),l(v)), X, Y, [1,2], [3])
    # xh=solve(op)
    lh = solve(op.skeleton_op)

    uu = get_trial_fe_basis(op.trial)
    vv = get_fe_basis(op.test)
    biform, liform  = op.weakform(uu, vv)    

    obiform, oliform = GridapHybrid._merge_bulk_and_skeleton_contributions(biform, liform)
    matvec, _, _ = Gridap.FESpaces._pair_contribution_when_possible(obiform, oliform)
    Γ = first(keys(matvec.dict))
    Gridap.Helpers.@check isa(Γ, GridapHybrid.SkeletonTriangulation) || isa(Γ, GridapHybrid.SkeletonView)
    lhₖ = get_cell_dof_values(lh, Γ)
    t = matvec.dict[Γ]
    m = BackwardStaticCondensationMap(op.bulk_fields, op.skeleton_fields)
    @enter uhphlhₖ = lazy_map(m, t, lhₖ)
    
    
    
    
    
    
    
    
    
    
    uh,_=xh
    e = u -uh
    @test sqrt(sum(∫(e⋅e)dΩ)) < 1.0e-12

    residual((uh,ph,lh),(vh,qh,mh))=a((uh,ph,lh),(vh,qh,mh))-l((vh,qh,mh))
    op = FEOperator(residual, X, Y)

    nls = NLSolver(show_trace=true, method=:newton)
    solver = FESolver(nls)

    xh0 = FEFunction(X,zeros(num_free_dofs(X)))
    xh, = solve!(xh0,solver,op)

    uh,_=xh
    e = u -uh
    @test sqrt(sum(∫(e⋅e)dΩ)) < 1.0e-12

    op = HybridFEOperator(residual, X, Y, [1,2], [3])