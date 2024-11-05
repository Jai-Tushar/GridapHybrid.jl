using Gridap
using Gridap.Geometry, Gridap.ReferenceFEs, Gridap.Arrays, Gridap.CellData, Gridap.FESpaces
using GridapHybrid
using GridapDistributed
using PartitionedArrays

function setup_reconstruction_operator(model, order, dΩ, d∂K)
  nK        = get_cell_normal_vector(d∂K.quad.trian)
  refferecᵤ = ReferenceFE(orthogonal_basis, Float64, order+1)
  reffe_c   = ReferenceFE(monomial_basis  , Float64, order+1; subspace=:OnlyConstant)

  Ω = dΩ.quad.trian

  VKR     = TestFESpace(Ω  , refferecᵤ; conformity=:L2)
  UKR     = TrialFESpace(VKR)
  VKR_C   = TestFESpace(Ω, reffe_c ; conformity=:L2, vector_type=Vector{Float64})
  UKR_C   = TrialFESpace(VKR_C)

  V = MultiFieldFESpace([VKR,VKR_C])
  U = MultiFieldFESpace([UKR,UKR_C])

  m( (u,u_c), (v,v_c)) = ∫(∇(v)⋅∇(u))dΩ + ∫(v_c*u)dΩ + ∫(v*u_c)dΩ
  n( (uK,u∂K), (v,v_c)) = ∫(∇(v)⋅∇(uK))dΩ + ∫(v_c*uK)dΩ + ∫((∇(v)⋅nK)*u∂K)d∂K - ∫((∇(v)⋅nK)*uK)d∂K 

  ReconstructionFEOperator((m,n), U, V)
end


function setup_projection_operator(UK_U∂K,VK_V∂K,R,dΩ,d∂K)
  m((uK,u∂K)  , (vK,v∂K)) = ∫(vK*uK)dΩ + ∫(v∂K*u∂K)d∂K
  function n(uK_u∂K, (vK,v∂K))
    urK_ur∂K = R(uK_u∂K)
    urK,ur∂K = urK_ur∂K
    uK,u∂K   = uK_u∂K
    # ∫(vK*(urK-uK))dΩ+∫(vK*ur∂K)dΩ -                 # bulk projection terms
    #    ∫(v∂K*urK)d∂K+∫(v∂K*u∂K)d∂K-∫(v∂K*ur∂K)d∂K   # skeleton projection terms
    ∫(vK*(urK-uK))dΩ+∫(vK*ur∂K)dΩ +                 # bulk projection terms
       ∫(v∂K*urK)d∂K-∫(v∂K*u∂K)d∂K+∫(v∂K*ur∂K)d∂K   # skeleton projection terms
  end
  function n(uK_u∂K::Gridap.MultiField.MultiFieldFEFunction, (vK,v∂K))
    uK,u∂K = uK_u∂K
    urK = R(uK_u∂K)
    ∫(vK*(urK-uK))dΩ +      # bulk projection terms
      ∫(v∂K*(urK-u∂K))d∂K   # skeleton projection terms
  end
  GridapHybrid.ProjectionFEOperator((m,n),UK_U∂K,VK_V∂K,R)
end



# Compute length of diagonals in the reference domain
function foo(m)
p0 = evaluate(m,Point(0.0,0.0))
p1 = evaluate(m,Point(1.0,1.0))
norm(p1-p0)
# To Do: incorporate the other diagonal    
#   p2 = evaluate(m,Point(1.0,0.0))
#   p3 = evaluate(m,Point(0.0,1.0))
#   maximum( norm(p1-p0), norm(p2-p3) )
end


  
  np = (2,2)
  # ranks = with_debug() do distribute
  #     distribute(LinearIndices((prod(np),)))
  # end

  domain = (0,1,0,1)
  mesh_partition = (2,2)
  model = CartesianDiscreteModel(domain, mesh_partition)

  D = num_cell_dims(model)
  Ω = Triangulation(ReferenceFE{D},model)
  Γ = Triangulation(ReferenceFE{D-1},model)
  ∂K= GridapHybrid.Skeleton(model)


  order = 0
  reffe = ReferenceFE(lagrangian, Float64, order; space=:P)
  VK    = TestFESpace(Ω  , reffe; conformity=:L2)
  V∂K   = TestFESpace(Γ  , reffe; conformity=:L2)
  UK = TrialFESpace(VK)
  U∂K= TrialFESpace(V∂K)


  degree = 2*(order+1)
  dΩ = Measure(Ω, degree)
  d∂K= Measure(∂K, degree)
  dΓ = Measure(Γ, degree)

  V = MultiFieldFESpace([VK,V∂K])
  U = MultiFieldFESpace([UK,U∂K])
  
  # v = get_fe_basis(V)
  # u = get_trial_fe_basis(U)

  # cmaps = collect1d(get_cell_map(Ω))
  # h_array = lazy_map(foo,cmaps)

  R = setup_reconstruction_operator(model, order, dΩ, d∂K)
  P = setup_projection_operator(U,V,R,dΩ,d∂K)

  xh = get_trial_fe_basis(U)
  yh = get_fe_basis(V)
  
  basis_style = GridapHybrid._get_basis_style(xh)

  LHSf, RHSf = GridapHybrid._evaluate_forms(P, xh)

  trial = GridapHybrid._to_trial_basis(xh)
  test  = GridapHybrid._get_test_fe_basis(P)

  urK_ur∂K = R(trial)
  urK,ur∂K = urK_ur∂K
  uK,u∂K   = trial
  vK,v∂K   = test 
  # ∫(vK*(urK-uK))dΩ+∫(vK*ur∂K)dΩ -                 # bulk projection terms
  #    ∫(v∂K*urK)d∂K+∫(v∂K*u∂K)d∂K-∫(v∂K*ur∂K)d∂K   # skeleton projection terms
  ∫(vK*(urK-uK))dΩ
  ∫(vK*ur∂K)dΩ                 # bulk projection terms
  ∫(v∂K*urK)d∂K
  ∫(v∂K*u∂K)d∂K
  ∫(v∂K*ur∂K)d∂K








  xK_x∂K = R(xh)

 @enter xK_x∂K_ΠK,xK_x∂K_Π∂K=P(xh)


trial = GridapHybrid._to_trial_basis(xh)
test  = GridapHybrid._get_test_fe_basis(P) 
@enter RHS_contribs = P.RHS_form(trial, test)

  @enter LHSf,RHSf = GridapHybrid._evaluate_forms(P,xh)
  cell_dofs = GridapHybrid._compute_cell_dofs(LHSf, RHSf)
  O = R.test_space
  GridapHybrid._generate_image_space_span(R,O,xh,cell_dofs,basis_style)


























  function r(u,v)
      uK_u∂K=R(u)
      vK_v∂K=R(v)
      uK,u∂K = uK_u∂K
      vK,v∂K = vK_v∂K 
      ∫(∇(vK)⋅∇(uK))dΩ + ∫(∇(vK)⋅∇(u∂K))dΩ +
         ∫(∇(v∂K)⋅∇(uK))dΩ + ∫(∇(v∂K)⋅∇(u∂K))dΩ
  end
  
  function s(u,v)
      Ω = get_triangulation(u)
      cmaps = collect1d(get_cell_map(Ω))
      h_array = lazy_map(foo,cmaps)
      # h_T = CellField(h_array,Ω,PhysicalDomain())
      h_T = CellField(h_array,Ω)
      h_T_1 = 1.0/h_T
   
  
      uK_u∂K_ΠK,uK_u∂K_Π∂K=P(u)
      vK_v∂K_ΠK,vK_v∂K_Π∂K=P(v)
  
      uK_ΠK  , u∂K_ΠK  = uK_u∂K_ΠK
      uK_Π∂K , u∂K_Π∂K = uK_u∂K_Π∂K
      
      vK_ΠK  , v∂K_ΠK  = vK_v∂K_ΠK
      vK_Π∂K , v∂K_Π∂K = vK_v∂K_Π∂K
  
      vK_Π∂K_vK_ΠK=vK_Π∂K-vK_ΠK
      v∂K_Π∂K_v∂K_ΠK=v∂K_Π∂K-v∂K_ΠK
      uK_Π∂K_uK_ΠK=uK_Π∂K-uK_ΠK
      u∂K_Π∂K_u∂K_ΠK=u∂K_Π∂K-u∂K_ΠK
  
      ∫(h_T_1*(vK_Π∂K_vK_ΠK)*(uK_Π∂K_uK_ΠK))d∂K + 
          ∫(h_T_1*(v∂K_Π∂K_v∂K_ΠK)*(u∂K_Π∂K_u∂K_ΠK))d∂K +
           ∫(h_T_1*(vK_Π∂K_vK_ΠK)*(u∂K_Π∂K_u∂K_ΠK))d∂K + 
            ∫(h_T_1*(v∂K_Π∂K_v∂K_ΠK)*(uK_Π∂K_uK_ΠK))d∂K
  end
  
  u(x) = x[1]^p+x[2]^p
  f(x) = -Δ(u)(x)
  
  a(u,v)=r(u,v)+s(u,v)
  l((vK,))=∫(vK*f)dΩ  

  uh  = get_trial_fe_basis(U)
  @enter Ruh = R(uh) 

  @enter op=HybridAffineFEOperator((u,v)->(a(u,v),l(v,)), U, V, [1], [2])
  
#   return A, h, ndofsΩ
# end

function domain_operation(operation, l::DomainContribution, r::DomainContribution)
  @check get_domains(r) == get_domains(l)
  o = DomainContribution()
  for trian in get_domains(l)
      cl = get_contribution(l, trian)
      cr = get_contribution(r, trian)
      c  = lazy_map(Broadcasting(operation), cl, cr)
      add_contribution!(o, trian, c) 
  end
  return o
end