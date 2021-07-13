module Experiments

using DrWatson
using Gridap
using ExploringGridapHybridization

include("../test/DarcyRTHTests.jl")

function solve_darcy_rth(n,k,d)
  outputs = Dict()
  if d==2
    domain=(0,1,0,1); partition = (n,n)
  else
    domain=(0,1,0,1,0,1); partition = (n,n,n)
  end
  model = CartesianDiscreteModel(domain,partition)
  ∂Topt = CellBoundaryOpt(model)

  outputs["Preassembly RTH [s]"] = @elapsed begin
     model_Γ,X,Y,M,L,cmat,cvec,cmat_cvec_condensed=
     DarcyRTHTests.preassembly_stage_darcy_hybrid_rt(model,∂Topt,k)
  end

  outputs["Trace assembly RTH [s]"] = @elapsed begin
    A,b,fdofscb=DarcyRTHTests.assembly_stage_darcy_hybrid_rt(model,∂Topt,M,L,cmat_cvec_condensed)
  end

  outputs["Trace solve RTH [s]"] = @elapsed begin
    x=DarcyRTHTests.solve_stage_darcy_hybrid_rt(A,b)
  end

  outputs["Back substitution RTH [s]"] = @elapsed begin
    sol_nonconforming=
       DarcyRTHTests.back_substitution_stage_darcy_hybrid_rt(∂Topt,
                     model,model_Γ,X,Y,M,L,x,cmat,cvec,fdofscb)
  end

  outputs["Total RTH [s]"]=outputs["Preassembly RTH [s]"]+
                           outputs["Trace assembly RTH [s]"]+
                           outputs["Trace solve RTH [s]"]+
                           outputs["Back substitution RTH [s]"]

  sol_conforming=DarcyRTHTests.solve_darcy_rt_hdiv(model,k)
  trian = Triangulation(model)
  degree = 2*(k+1)
  dΩ = Measure(trian,degree)
  uhc,_=sol_conforming
  uhnc,_,_=sol_nonconforming
  err_norm=sqrt(sum(∫((uhc-uhnc)⋅(uhc-uhnc))dΩ))

  outputs["num_cells"] = num_cells(model)
  outputs["l2_err_norm"] = err_norm
  outputs["n"] = n
  outputs["k"] = k
  outputs["d"] = d
  outputs["experiment"] = "solve_darcy_rth"
  return outputs
end


function solve_darcy_rt(n,k,d)
  if d==2
    domain=(0,1,0,1); partition = (n,n)
  else
    domain=(0,1,0,1,0,1); partition = (n,n,n)
  end
  model = CartesianDiscreteModel(domain,partition)

  outputs = Dict()

  outputs["Preassembly RT [s]"] = @elapsed begin
     X,Y,a,b=DarcyRTHTests.pre_assembly_stage_rt_hdiv(model,k)
  end
  outputs["Assembly RT [s]"] = @elapsed begin
     op=DarcyRTHTests.assembly_stage_rt_hdiv(X,Y,a,b)
  end
  outputs["Solve RT [s]"] = @elapsed begin
     sol_nonconforming=DarcyRTHTests.solve_stage_rt_hdiv(op)
  end
  outputs["Total RT [s]"]=outputs["Preassembly RT [s]"]+
                          outputs["Assembly RT [s]"]+
                          outputs["Solve RT [s]"]

  sol_conforming=DarcyRTHTests.solve_darcy_rt_hdiv(model,k)
  trian = Triangulation(model)
  degree = 2*(k+1)
  dΩ = Measure(trian,degree)
  uhc,_=sol_conforming
  uhnc,_=sol_nonconforming
  err_norm=sqrt(sum(∫((uhc-uhnc)⋅(uhc-uhnc))dΩ))

  outputs["num_cells"] = num_cells(model)
  outputs["l2_err_norm"] = err_norm
  outputs["n"] = n
  outputs["k"] = k
  outputs["d"] = d
  outputs["experiment"] = "solve_darcy_rt"
  return outputs
end

end # module
