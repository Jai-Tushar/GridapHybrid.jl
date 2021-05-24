using Gridap
using FillArrays
using Gridap.Geometry

struct CellBoundary{M<:DiscreteModel,BTP<:Triangulation,BTM<:Triangulation}
  model::M
  boundary_trian_plus::BTP
  boundary_trian_minus::BTM
  sign_flip

  function CellBoundary(model::M) where M<:DiscreteModel

     function _get_bgface_to_lcell_btm(model)
       bgface_to_lcell_btm  = Vector{Int8}(undef,num_facets(model))
       bgface_to_lcell_btm     .= 2

       Dc=num_cell_dims(model)
       Df=Dc-1
       topo = Gridap.Geometry.get_grid_topology(model)
       bgface_to_cells = Gridap.Geometry.get_faces(topo,Df,Dc)

       c=array_cache(bgface_to_cells)
       for i=1:length(bgface_to_cells)
         cells_around=getindex!(c,bgface_to_cells,i)
         if length(cells_around)==1
          bgface_to_lcell_btm[i] = 1
         end
       end
       bgface_to_lcell_btm
     end

     # TO-DO: here I am reusing the machinery for global RT FE spaces.
     #        Sure there is a way to decouple this from global RT FE spaces.
     function _get_sign_flip(model)
       basis,reffe_args,reffe_kwargs = ReferenceFE(raviart_thomas,Float64,0)
       cell_reffe = ReferenceFE(model,basis,reffe_args...;reffe_kwargs...)
       Gridap.FESpaces.get_sign_flip(model,cell_reffe)
     end


    face_to_bgface       = collect(1:num_facets(model))
    btp=BoundaryTriangulation(model,
                              face_to_bgface,
                              Fill(Int8(1),num_facets(model)))
    btm=BoundaryTriangulation(model,
                              face_to_bgface,
                              _get_bgface_to_lcell_btm(model))



    TBTP=typeof(btp)
    TBTM=typeof(btm)
    new{M,TBTP,TBTM}(model,btp,btm,_get_sign_flip(model))
  end
end

"""
Returns a cell-wise array which, for each cell, and each facet within the cell,
returns the unit outward normal to the boundary of the cell.
"""
function get_cell_normal_vector(cb::CellBoundary)
    np=get_normal_vector(cb.boundary_trian_plus)
    nm=get_normal_vector(cb.boundary_trian_minus)
    signed_cell_wise_facets_ids = _get_signed_cell_wise_facets(cb)
    np_array=Gridap.CellData.get_data(np)
    nm_array=Gridap.CellData.get_data(nm)
    num_cell_facets = _get_num_facets(cb)
    per_local_facet_normal_cell_wise_arrays =
       _get_per_local_facet_cell_wise_arrays(np_array,
                                             nm_array,
                                             signed_cell_wise_facets_ids,
                                             num_cell_facets)
    _block_arrays(per_local_facet_normal_cell_wise_arrays...)
end

"""
Returns a cell-wise array which, for each cell, and each facet within the cell,
returns the unit outward normal to the boundary of the cell owner of the facet.
"""
function get_cell_owner_normal_vector(cb::CellBoundary)
  np=get_normal_vector(cb.boundary_trian_plus)
  np_array=Gridap.CellData.get_data(np)
  num_cell_facets = _get_num_facets(cb)
  cell_wise_facets = _get_cell_wise_facets(cb)
  per_local_facet_normal_cell_wise_arrays =
      _get_per_local_facet_cell_wise_arrays(np_array,
                                            cell_wise_facets,
                                            num_cell_facets)
  _block_arrays(per_local_facet_normal_cell_wise_arrays...)
end

function quadrature_evaluation_points_and_weights(cell_boundary::CellBoundary, degree::Integer)
  model = cell_boundary.model
  D = num_cell_dims(model)
  p = lazy_map(Gridap.ReferenceFEs.get_polytope,get_reffes(model))
  @assert length(p) == 1
  p  = p[1]
  pf = Gridap.ReferenceFEs.Polytope{D-1}(p,1)
  qf = Gridap.ReferenceFEs.Quadrature(pf,degree)

  xf = Gridap.ReferenceFEs.get_coordinates(qf)
  wf = Gridap.ReferenceFEs.get_weights(qf)

  num_cell_facets = _get_num_facets(cell_boundary)
  xf_array_block=_block_arrays(Fill(Fill(xf,num_cells(model)),num_cell_facets)...)
  wf_array_block=_block_arrays(Fill(Fill(wf,num_cells(model)),num_cell_facets)...)

  (xf_array_block,wf_array_block)
  # TO-THINK: should actually be this triangulation the one that goes into
  # the CellQuadrature of CellBoundary?
  #Gridap.CellData.CellQuadrature(get_triangulation(model),qcb)
end

# This function plays a similar role of the change_domain function in Gridap.
# Except it does not, by now, deal with Triangulation and DomainStyle instances.
# (TO-THINK)
function restrict_to_cell_boundary(cb::CellBoundary, fe_basis::Gridap.FESpaces.FEBasis)
  model = cb.model
  D = num_cell_dims(model)
  trian = get_triangulation(fe_basis)
  if isa(trian,Triangulation{D-1,D})
    _restrict_to_cell_boundary_facet_fe_basis(cb,fe_basis)
  elseif isa(trian,Triangulation{D,D})
    _restrict_to_cell_boundary_cell_fe_basis(cb,fe_basis)
  end
end

function _restrict_to_cell_boundary_facet_fe_basis(cb::CellBoundary,
                                                   facet_fe_basis::Gridap.FESpaces.FEBasis)

  # TO-THINK:
  #     1) How to deal with Test/Trial FEBasis objects?
  #     2) How to deal with CellField objects which are NOT FEBasis objects?
  #     3) How to deal with differential operators applied to FEBasis/CellField objects?
  @assert Gridap.FESpaces.BasisStyle(facet_fe_basis) == Gridap.FESpaces.TrialBasis()
  D = num_cell_dims(cb.model)
  @assert isa(get_triangulation(facet_fe_basis),Triangulation{D-1,D})

  num_cell_facets  = _get_num_facets(cb)
  cell_wise_facets = _get_cell_wise_facets(cb)

  fta=Gridap.CellData.get_data(facet_fe_basis)
  fe_basis_restricted_to_lfacet_cell_wise =
    _get_per_local_facet_cell_wise_arrays(fta,
                                          cell_wise_facets,
                                          num_cell_facets)
  fe_basis_restricted_to_lfacet_cell_wise_expanded =
    [ _expand_facet_lid_fields_to_num_facets_blocks(
                            fe_basis_restricted_to_lfacet_cell_wise[facet_lid],
                            num_cell_facets,
                            facet_lid) for facet_lid=1:num_cell_facets ]

  # Cell-wise array of VectorBlock entries with as many entries as facets.
  # For each cell, and local facet, we get the cell FE Basis restricted to that facet.
  _block_arrays(fe_basis_restricted_to_lfacet_cell_wise_expanded...)
end

function _get_block_layout(fields_array::AbstractArray{<:AbstractArray{<:Gridap.Fields.Field}})
  Fill((1,1),length(fields_array))
end

function _get_block_layout(fields_array::AbstractArray{<:Gridap.Fields.ArrayBlock})
  lazy_map(x->((size(x),findall(x.touched))),fields_array)
end

function _restrict_to_cell_boundary_cell_fe_basis(cb::CellBoundary,
                                                  cell_fe_basis::Gridap.FESpaces.FEBasis)
  # TO-THINK:
  #     1) How to deal with Test/Trial FEBasis objects?
  #     2) How to deal with CellField objects which are NOT FEBasis objects?
  #     3) How to deal with differential operators applied to FEBasis objects?
  @assert Gridap.FESpaces.BasisStyle(cell_fe_basis) == Gridap.FESpaces.TestBasis()
  D = num_cell_dims(cb.model)
  @assert isa(get_triangulation(cell_fe_basis),Triangulation{D,D})

  num_cell_facets = _get_num_facets(cb)

  cfp=Gridap.CellData.change_domain(cell_fe_basis,
                                    cb.boundary_trian_plus,
                                    DomainStyle(cell_fe_basis))
  cfm=Gridap.CellData.change_domain(cell_fe_basis,
                                    cb.boundary_trian_minus,
                                    DomainStyle(cell_fe_basis))
  cfp_array=Gridap.CellData.get_data(cfp)
  cfm_array=Gridap.CellData.get_data(cfm)
  signed_cell_wise_facets_ids = _get_signed_cell_wise_facets(cb)

  # Array with as many cell-wise arrays as facets. For each facet, we get a
  # cell-wise array with the cell FE Basis restricted to that facet.
  per_local_facet_fe_basis_cell_wise_arrays =
     _get_per_local_facet_cell_wise_arrays(cfp_array,
                                           cfm_array,
                                           signed_cell_wise_facets_ids,
                                           num_cell_facets)

  # Cell-wise array of VectorBlock entries with as many entries as facets.
  # For each cell, and local facet, we get the cell FE Basis restricted to that facet.
  _block_arrays(per_local_facet_fe_basis_cell_wise_arrays...)
end

function _expand_facet_lid_fields_to_num_facets_blocks(fe_basis_restricted_to_lfacet_cell_wise,
                                                       num_cell_facets,
                                                       facet_lid)
    bl=_get_block_layout(fe_basis_restricted_to_lfacet_cell_wise)[1]
    lazy_map(Gridap.Fields.BlockMap(bl[1],bl[2]),
             lazy_map(Gridap.Fields.BlockMap((1,num_cell_facets),[CartesianIndex((1,facet_lid))]),
                      lazy_map(x->x[findfirst(x.touched)], fe_basis_restricted_to_lfacet_cell_wise)))
end

function _get_num_facets(cb::CellBoundary)
  model = cb.model
  p = lazy_map(Gridap.ReferenceFEs.get_polytope,get_reffes(model))
  @assert length(p) == 1
  num_facets(p[1])
end

function _get_cell_wise_facets(cb::CellBoundary)
  model = cb.model
  gtopo = get_grid_topology(model)
  D     = num_cell_dims(model)
  Gridap.Geometry.get_faces(gtopo, D, D-1)
end

function _get_signed_cell_wise_facets(cb::CellBoundary)
  lazy_map(Broadcasting((y,x)-> y ? -x : x),
                        cb.sign_flip,
                        _get_cell_wise_facets(cb))
end

function Gridap.Geometry.get_cell_map(cb::CellBoundary)
  num_cell_facets = _get_num_facets(cb)
  cell_map_tp=get_cell_map(cb.boundary_trian_plus)
  cell_map_tm=get_cell_map(cb.boundary_trian_minus)
  signed_cell_wise_facets_ids = _get_signed_cell_wise_facets(cb)
  # Array with as many cell-wise arrays as facets. For each facet, we get a
  # cell-wise array with the cell FE Basis restricted to that facet.
  per_local_facet_maps_cell_wise_arrays =
      _get_per_local_facet_cell_wise_arrays(cell_map_tp,
                                            cell_map_tm,
                                            signed_cell_wise_facets_ids,
                                            num_cell_facets)
  _block_arrays(per_local_facet_maps_cell_wise_arrays...)
end

function _get_per_local_facet_cell_wise_arrays(tpa, # Facet triangulation + array
                                               tma, # Facet triangulation - array
                                               signed_cell_wise_facets_ids,
                                               num_cell_facets)
  [ _get_local_facet_cell_wise_array(tpa,
                                     tma,
                                     signed_cell_wise_facets_ids,
                                     facet_lid) for facet_lid=1:num_cell_facets ]
end

function _get_local_facet_cell_wise_array(tpa,
                                          tma,
                                          signed_cell_wise_facets_ids,
                                          facet_lid)
  indices=lazy_map((x)->x[facet_lid],signed_cell_wise_facets_ids)
  lazy_map(Gridap.Arrays.PosNegReindex(tpa,tma),indices)
end

function _get_per_local_facet_cell_wise_arrays(fta, # Facet triangulation array
                                               cell_wise_facets_ids,
                                               num_cell_facets)
  [ _get_local_facet_cell_wise_array(fta,
                                     cell_wise_facets_ids,
                                     facet_lid) for facet_lid=1:num_cell_facets ]
end

function _get_local_facet_cell_wise_array(fta,
                                          cell_wise_facets_ids,
                                          facet_lid)
  indices=lazy_map((x)->x[facet_lid],cell_wise_facets_ids)
  lazy_map(Gridap.Arrays.Reindex(fta),indices)
end

function _block_arrays(a::AbstractArray...)
  num_blocks = length(a)
  lazy_map(Gridap.Fields.BlockMap(num_blocks,collect(1:num_blocks)),a...)
end

# This function will eventually play the role of its counterpart in Gridap
# function integrate(f::CellField,quad::CellQuadrature)
# TO-THINK: vh,uh currently as LazyArray{...}
#    Not sure if we are loosing generality by constraining a to be of type
#    LazyArray{...}. I need to restrict the cell-wise block array to each
#    individual block, and with a LazyArray{...} this is very efficient as
#    the array is already restricted to each block in the a.args member variable.
function integrate_low_level(cb::CellBoundary,
                             vh::Gridap.Arrays.LazyArray{<:Fill{<:Gridap.Fields.BlockMap}},
                             uh::Gridap.Arrays.LazyArray{<:Fill{<:Gridap.Fields.BlockMap}},
                             x::AbstractArray{<:Gridap.Fields.ArrayBlock{<:AbstractArray{<:Point}}},
                             w::AbstractArray{<:Gridap.Fields.ArrayBlock{<:AbstractVector}})
  vhx=lazy_map(evaluate,vh,x)
  uhx=lazy_map(evaluate,uh,x)
  vhx_mult_uhx=lazy_map(Gridap.Fields.BroadcastingFieldOpMap(*), vhx, uhx)
  j=lazy_map(∇,get_cell_map(cb))
  jx=lazy_map(evaluate,j,x)

  args = [ lazy_map(Gridap.Fields.IntegrationMap(),
                    _restrict_cell_array_block_to_block(vhx_mult_uhx,pos),
                    _restrict_cell_array_block_to_block(w,pos),
                    jx.args[pos]) for pos=1:length(vhx.args) ]
  lazy_map(+,args...)
end