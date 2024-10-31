using Gridap
using Gridap.Geometry, Gridap.ReferenceFEs, Gridap.Arrays, Gridap.CellData, Gridap.FESpaces
using GridapDistributed
using PartitionedArrays

np = (2,2)
ranks = with_debug() do distribute
    distribute(LinearIndices((prod(np),)))
end

domain = (0,1,0,1)
mesh_partition = (4,4)
model = CartesianDiscreteModel(ranks, np, domain, mesh_partition)
D  = num_cell_dims(model) 

# labels = get_face_labeling(model)
# add_tag_from_tags!(labels,"diri1",[6,])
# add_tag_from_tags!(labels,"diri0",[1,2,3,4,5,7,8])

order = 1
degree = order
Ωₕ = Triangulation(model)
Γ = Triangulation(model)
dΩ = Measure(Ωₕ,degree)

reffeᵤ = ReferenceFE(lagrangian,VectorValue{2,Float64},order)
reffeₚ = ReferenceFE(lagrangian,Float64,order-1;space=:P)

V = TestFESpace(model,reffeᵤ,conformity=:H1)
Q = TestFESpace(model,reffeₚ,conformity=:L2,constraint=:zeromean)
Y = MultiFieldFESpace([V,Q])

U = TrialFESpace(V)
P = TrialFESpace(Q)
X = MultiFieldFESpace([U,P])

yh = get_fe_basis(Y); 
uh,vh = yh;



ff = VectorValue(0.0,0.0)
a((u,p),(v,q)) = ∫( ∇(v)⊙∇(u) - (∇⋅v)*p + q*(∇⋅u) )dΩ
l((v,q)) = ∫( v⋅ff )dΩ

@enter op = AffineFEOperator(a,l,X,Y)

# u((x,y)) = (x+y)^order
# f(x) = -Δ(u,x)

# reffe = ReferenceFE(lagrangian,Float64,order)
# V = TestFESpace(model,reffe,dirichlet_tags="boundary")
# U = TrialFESpace(u,V)
# Ω = Triangulation(model)
# dΩ = Measure(Ω,2*order)

# # v = get_fe_basis(V)

# a(u,v) = ∫( ∇(v)⋅∇(u) )dΩ
# l(v) = ∫( v*f )dΩ

# @enter v = get_fe_basis(V)

# @enter op = AffineFEOperator(a,l,U,V)