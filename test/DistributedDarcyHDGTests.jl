module DistributedDarcyHDGTests

  using Test
  using Gridap
  using Gridap.Geometry, Gridap.ReferenceFEs, Gridap.Arrays, Gridap.CellData, Gridap.FESpaces
  using GridapHybrid
  using GridapDistributed
  using PartitionedArrays

  # 2D
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


  function solve_darcy_hdg(mesh_partition, np, ranks, order)

  
    domain = (0,1,0,1)

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
                          #∫(qh*(uh⋅n+τ*(ph-lh)*n⋅n0))*d∂K
                          ∫(qh*(uh⋅n))d∂K +
                          ∫(τ*qh*ph*(n⋅n0))d∂K -
                          ∫(τ*qh*lh*(n⋅n0))d∂K +
                          #∫(mh*(uh⋅n+τ*(ph-lh)*n⋅n0))*d∂K
                          ∫(mh*(uh⋅n))d∂K +
                          ∫(τ*mh*ph*(n⋅n0))d∂K -
                          ∫(τ*mh*lh*(n⋅n0))d∂K                            

    l((vh,qh,mh),dΩ) = ∫( vh⋅f + qh*(∇⋅u))*dΩ

    weakform = (u,v)->(a(u,v,dΩ,d∂K),l(v,dΩ))

    op = HybridAffineFEOperator(weakform, X, Y, [1,2], [3])

    lh = solve(op.skeleton_op)

    xh = solve(op)

    uh,_ = xh;

    e = u - uh

    @test sqrt(sum(∫(e⋅e)dΩ)) < 1.0e-12
  end

  order = 2
  np = (2,2)
  mesh_partition = (2,2)

  ranks = with_debug() do distribute
    distribute(LinearIndices((prod(np),)))
  end

  solve_darcy_hdg(mesh_partition, np, ranks, order)

end














