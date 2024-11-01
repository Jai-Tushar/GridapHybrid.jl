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

  m( (u,u_c), (v,v_c) ) = ∫(∇(v)⋅∇(u))dΩ + ∫(v_c*u)dΩ + ∫(v*u_c)dΩ
  n( (uK,u∂K), (v,v_c) ) = ∫(∇(v)⋅∇(uK))dΩ + ∫(v_c*uK)dΩ + ∫((∇(v)⋅nK)*u∂K)d∂K - ∫((∇(v)⋅nK)*uK)d∂K 

  ReconstructionFEOperator((m,n), U, V)
end


function setup_projection_operator(UK_U∂K,VK_V∂K,R,dΩ,d∂K)
  m((uK,u∂K)  , (vK,v∂K) ) = ∫(vK*uK)dΩ + ∫(v∂K*u∂K)d∂K
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
  GridapHybrid.ProjectionFEOperator((m,n),UK_U∂K,VK_V∂K)
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

u(x) = x[1]^p+x[2]^p
f(x) = -Δ(u)(x)
  
  np = (2,2)
  ranks = with_debug() do distribute
      distribute(LinearIndices((prod(np),)))
  end

  domain = (0,1,0,1)
  mesh_partition = (2,2)
  model = CartesianDiscreteModel(ranks,np,domain, mesh_partition)

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
  
  # model1 = local_views(model).items[1];
  # dΩ1 = local_views(dΩ).items[1];
  # d∂K1 = local_views(d∂K).items[1];

  # nK1 = get_cell_normal_vector(d∂K1.quad.trian)
  # refferecᵤ = ReferenceFE(orthogonal_basis, Float64, order+1)
  # reffe_c   = ReferenceFE(monomial_basis  , Float64, order+1; subspace=:OnlyConstant)

  # Ω1 = dΩ1.quad.trian

  # VKR     = TestFESpace(Ω1  , refferecᵤ; conformity=:L2)
  # UKR     = TrialFESpace(VKR)
  # VKR_C   = TestFESpace(Ω1, reffe_c ; conformity=:L2, vector_type=Vector{Float64})
  # UKR_C   = TrialFESpace(VKR_C)

  # VV = MultiFieldFESpace([VKR,VKR_C])
  # UU = MultiFieldFESpace([UKR,UKR_C])

  # m( (u,u_c), (v,v_c) ) = ∫(∇(v)⋅∇(u))dΩ1 + ∫(v_c*u)dΩ1 + ∫(v*u_c)dΩ1
  # n( (uK,u∂K), (v,v_c) ) = ∫(∇(v)⋅∇(uK))dΩ1 + ∫(v_c*uK)dΩ1 + ∫((∇(v)⋅nK)*u∂K)d∂K1 - ∫((∇(v)⋅nK)*uK)d∂K1 

  # R = ReconstructionFEOperator((m,n), UU, VV)

  function setup_reconstruction_operator(model::GridapDistributed.GenericDistributedDiscreteModel, 
    order, dΩ::GridapDistributed.DistributedMeasure, d∂K::GridapDistributed.DistributedMeasure)
    R = map(local_views(model), local_views(dΩ), local_views(d∂K)) do m, dΩi, d∂Ki
        setup_reconstruction_operator(m, order, dΩi, d∂Ki)
    end
    return R
  end

  R = setup_reconstruction_operator(model, order, dΩ, d∂K)

  function setup_projection_operator(trial::GridapDistributed.DistributedMultiFieldFESpace, 
    test::GridapDistributed.DistributedMultiFieldFESpace, R::AbstractArray{<:ReconstructionFEOperator},
    dΩ::GridapDistributed.DistributedMeasure, d∂K::GridapDistributed.DistributedMeasure)
    P = map(local_views(trial), local_views(test), local_views(R), local_views(dΩ), local_views(d∂K)) do triali, testi, Ri, dΩi, d∂Ki
        setup_projection_operator(triali, testi, Ri, dΩi, d∂Ki)
    end
    return P
  end


  P = setup_projection_operator(U, V, R, dΩ, d∂K)

  # P = map(local_views(U), local_views(V), local_views(R), local_views(dΩ), local_views(d∂K))  do Ui, Vi, Ri, dΩi, d∂Ki
  #     setup_projection_operator(Ui, Vi, Ri, dΩi, d∂Ki)
  # end 

  xh = get_trial_fe_basis(U);
  yh = get_fe_basis(V);

  function (op::AbstractArray{<:ReconstructionFEOperator})(v::GridapDistributed.DistributedMultiFieldCellField) 
    
    image_span = map(local_views(op), local_views(v)) do opi, vi

      basis_style = GridapHybrid._get_basis_style(vi)
      LHSf, RHSf = GridapHybrid._evaluate_forms(opi, vi)
      cell_dofs = GridapHybrid._compute_cell_dofs(LHSf, RHSf)
      O = opi.test_space
      GridapHybrid._generate_image_space_span(opi,O,vi,cell_dofs,basis_style)

    end
    return image_span
  end

  xK_x∂K = R(xh)
  yK_y∂K = R(yh)

  
  function (op::AbstractArray{<:ProjectionFEOperator})(v::GridapDistributed.DistributedMultiFieldCellField)
    
    basis_style = _get_basis_style(v)
    LHSf,RHSf= _evaluate_forms(op,v)
    cell_dofs= _compute_cell_dofs(LHSf,RHSf)
    # We get the test space so that we don't deal with
    # the transpose underlying the trial basis
    O = op.test_space
    _generate_image_space_span(op,O,v,cell_dofs,basis_style)
  end



  xK_x∂K_ΠK,xK_x∂K_Π∂K=P(xh)
 

  xh1 = local_views(xh).items[1];
  P1  = local_views(P).items[1]; 


  basis_style = GridapHybrid._get_basis_style(xh1)
  @enter LHSf, RHSf = GridapHybrid._evaluate_forms(P1, xh1)
  # cell_dofs = GridapHybrid._compute_cell_dofs(LHSf, RHSf)
  # O = R1.test_space
  # GridapHybrid._generate_image_space_span(R1,O,xh1,cell_dofs,basis_style)

  biform, liform  = weakform(xh, yh)

  function r(u,v,dΩ)
      uK_u∂K=R(u)
      vK_v∂K=R(v)
      uK,u∂K = uK_u∂K
      vK,v∂K = vK_v∂K 
      ∫(∇(vK)⋅∇(uK))dΩ + ∫(∇(vK)⋅∇(u∂K))dΩ +
         ∫(∇(v∂K)⋅∇(uK))dΩ + ∫(∇(v∂K)⋅∇(u∂K))dΩ
  end
  
  function s(u,v,d∂K)
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
  
  a(u,v,dΩ,d∂K) = r(u,v,dΩ) + s(u,v,d∂K)
  l((vK,),dΩ) = ∫(vK*f)dΩ

  weakform = (u,v)->(a(u,v,dΩ,d∂K),l(v,dΩ))
  @enter op=HybridAffineFEOperator(weakform, U, V, [1], [2])
  


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