# module PoissonHHOTests
  using Gridap
  using GridapHybrid
  using Test
  using Plots

  function setup_reconstruction_operator(model, order, dΩ, d∂K, VK_V∂K)
    nK        = get_cell_normal_vector(d∂K.quad.trian)
    refferecᵤ = ReferenceFE(orthogonal_basis, Float64, order+1)
    reffe_nzm = ReferenceFE(orthogonal_basis, Float64, order+1; subspace=:NonZeroMean)
    reffe_zm  = ReferenceFE(orthogonal_basis, Float64, order+1; subspace=:ZeroMean)
    reffe_c   = ReferenceFE(monomial_basis  , Float64, order+1; subspace=:OnlyConstant)
    reffe_nc  = ReferenceFE(monomial_basis  , Float64, order+1; subspace=:ExcludeConstant)

    Ω = dΩ.quad.trian

    VKR     = TestFESpace(Ω  , refferecᵤ; conformity=:L2)
    UKR     = TrialFESpace(VKR)
    # UKR_NZM = TrialFESpace(TestFESpace(Ω, reffe_nzm; conformity=:L2))
    # UKR_ZM  = TrialFESpace(TestFESpace(Ω, reffe_zm; conformity=:L2))
    VKR_C   = TestFESpace(Ω, reffe_c ; conformity=:L2, vector_type=Vector{Float64})
    UKR_C   = TrialFESpace(VKR_C)
    # VKR_NC  = TestFESpace(Ω, reffe_nc; conformity=:L2, vector_type=Vector{Float64})

    # VKR_DS_DECOMP = MultiFieldFESpace([VKR_C,VKR_NC])
    # UKR_DS_DECOMP = MultiFieldFESpace([UKR_NZM,UKR_ZM])

    V = MultiFieldFESpace([VKR,VKR_C])
    U = MultiFieldFESpace([UKR,UKR_C])

    # m( (u_nzm,u_zm), (v_c,v_nc) ) = ∫(∇(v_nc)⋅∇(u_zm))dΩ + ∫(∇(v_nc)⋅∇(u_nzm))dΩ +
    #                                ∫(v_c*u_nzm)dΩ
    # n( (uK,u∂K), (v_c,v_nc)     ) = ∫(-Δ(v_nc)*uK)dΩ + ∫(v_c*uK)dΩ + ∫((∇(v_nc)⋅nK)*u∂K)d∂K

    m( (u,u_c), (v,v_c) ) = ∫(∇(v)⋅∇(u))dΩ + ∫(v_c*u)dΩ + ∫(v*u_c)dΩ
    n( (uK,u∂K), (v,v_c) ) = ∫(∇(v)⋅∇(uK))dΩ + ∫(v_c*uK)dΩ + ∫((∇(v)⋅nK)*u∂K)d∂K - ∫((∇(v)⋅nK)*uK)d∂K 

    LocalFEOperator((m,n), U, V, VK_V∂K)

    # LocalFEOperator((m,n),UKR,VKR;
    #                 trial_space_ds_decomp=UKR_DS_DECOMP,
    #                 test_space_ds_decomp=VKR_DS_DECOMP)
  end

  function setup_difference_operator(UK_U∂K,VK_V∂K,R,dΩ,d∂K)
    m((uK,u∂K)  , (vK,v∂K)) = ∫(vK*uK)dΩ + ∫(v∂K*u∂K)d∂K
    function n(uK_u∂K, (vK,v∂K))
      urK_ur∂K, ur∂KF = R(uK_u∂K)
      urK,ur∂K = urK_ur∂K
      uK,u∂K    = uK_u∂K
      #println((v∂K*ur∂KF)(get_cell_points(d∂K.quad))[1][1][2,2][1,1])
      println(size(ur∂KF(get_cell_points(d∂K.quad))[1][2][1,2][1]))
      xxx
      ∫(vK*(urK-uK))dΩ-∫(vK*ur∂K)dΩ-∫(v∂K*urK)d∂K-∫(v∂K*(u∂K-ur∂KF))d∂K
    end
    LocalFEOperator((m,n),UK_U∂K,VK_V∂K,VK_V∂K; field_type_at_common_faces=MultiValued())
   end

   p = 0
   u(x) = x[1]^p+x[2]^p                         # Ex 1
  #  u(x) = x[1]*(x[1]-1)^p*x[2]*(x[2]-1)^p         # Ex 2
   f(x)=-Δ(u)(x)

   #u(x)=x[1]+x[2]
   #f(x)=-Δ(u)(x)

  function solve_hho(cells,order)
      partition = (0,1,0,1)
      model = CartesianDiscreteModel(partition, cells)
      D  = num_cell_dims(model)
      Ω  = Triangulation(ReferenceFE{D},model)
      Γ  = Triangulation(ReferenceFE{D-1},model)
      ∂K = GridapHybrid.Skeleton(model)

      reffeᵤ    = ReferenceFE(lagrangian,Float64,order  ;space=:P)

      VK     = TestFESpace(Ω  , reffeᵤ; conformity=:L2)
      V∂K    = TestFESpace(Γ  , reffeᵤ; conformity=:L2,dirichlet_tags=collect(5:8))  # Ex 1
      # V∂K    = TestFESpace(Γ  , reffeᵤ; conformity=:L2,dirichlet_tags="boundary")   # Ex 2
      UK     = TrialFESpace(VK)
      U∂K    = TrialFESpace(V∂K,u)
      VK_V∂K = MultiFieldFESpace([VK,V∂K])
      UK_U∂K = MultiFieldFESpace([UK,U∂K])

      degree = 2*order+1
      dΩ     = Measure(Ω,degree)
      dΓ     = Measure(Γ,degree)
      d∂K    = Measure(∂K,degree)

      R=setup_reconstruction_operator(model, order, dΩ, d∂K, VK_V∂K)
      diff_op=setup_difference_operator(UK_U∂K,VK_V∂K,R,dΩ,d∂K)

      function r(u,v)
        uK_u∂K,_=R(u)
        vK_v∂K,_=R(v)
        uK,u∂K = uK_u∂K
        vK,v∂K = vK_v∂K 
        ∫(∇(vK)⋅∇(uK))dΩ + ∫(∇(vK)⋅∇(u∂K))dΩ +
           ∫(∇(v∂K)⋅∇(uK))dΩ + ∫(∇(v∂K)⋅∇(u∂K))dΩ
      end

      function s(u,v)
        h_T=CellField(get_array(∫(1)dΩ),Ω)
        h_T_1=1.0/h_T
        h_T_2=1.0/(h_T*h_T)

        δuK,δu∂K=diff_op(u)
        δvK,δv∂K=diff_op(v)
        # δvK_K,δvK_∂K=δvK
        # δuK_K,δuK_∂K=δuK
        # δv∂K_K,δv∂K_∂K=δv∂K
        # δu∂K_K,δu∂K_∂K=δu∂K
        # ∫(h_T_2*δvK_K*δuK_K)dΩ+
        #   ∫(h_T_2*δvK_K*δuK_∂K)dΩ+
        #      ∫(h_T_2*δvK_∂K*δuK_K)dΩ+
        #        ∫(h_T_2*δvK_∂K*δuK_∂K)dΩ+
        # ∫(h_T_1*δv∂K_∂K*δu∂K_∂K)d∂K
      end

      a(u,v)=r(u,v)+s(u,v)
      l((vK,))=∫(vK*f)dΩ


      op=HybridAffineFEOperator((u,v)->(a(u,v),l(v)), UK_U∂K, VK_V∂K, [1], [2])
      xh=solve(op)

      uhK,uh∂K=xh

      println(get_free_dof_values(uhK))

      e = u -uhK
      # @test sqrt(sum(∫(e⋅e)dΩ)) < 1.0e-12
      return sqrt(sum(∫(e⋅e)dΩ))
  end

  function conv_test(ns,order)
    el2 = Float64[]
    hs = Float64[]
    for n in ns
      l2 = solve_hho((n,n),order)
      println(l2)
      h = 1.0/n
      push!(el2,l2)
      push!(hs,h)
    end
    println(el2)
    el2, hs
  end

  function slope(hs,errors)
    x = log10.(hs)
    y = log10.(errors)
    linreg = hcat(fill!(similar(x), 1), x) \ y
    linreg[2]
  end

  solve_hho((2,2),0)

  # ns=[8,16,32,64,128]
  ns=[8,16,32,64]
  order=1
  el, hs = conv_test(ns,order)
  println("Slope L2-norm u: $(slope(hs,el))")
  slopek  =[Float64(ni)^(-(order)) for ni in ns]
  slopekp1=[Float64(ni)^(-(order+1)) for ni in ns]
  slopekp2=[Float64(ni)^(-(order+2)) for ni in ns]
  display(plot(hs,[el slopek slopekp1 slopekp2],
    xaxis=:log, yaxis=:log,
    label=["L2u (measured)" "slope k" "slope k+1" "slope k+2"],
    shape=:auto,
    xlabel="h",ylabel="L2 error",legend=:bottomright))
end
