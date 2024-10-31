# Hybrid distributed model machinery 
function GridapHybrid.Skeleton(model::GridapDistributed.DistributedDiscreteModel)
    cell_ids = get_cell_gids(model)
    trians = map(local_views(model),partition(cell_ids)) do model, indices
        Gridap.Geometry.TriangulationView(
            GridapHybrid.Skeleton(model),
            own_to_local(indices)
        )
    end
    return GridapDistributed.DistributedTriangulation(trians,model)
end

# Integration machinery for distributed hybrid models
const SkeletonView{Dc,Dp} = Gridap.Geometry.TriangulationView{Dc,Dp,<:GridapHybrid.SkeletonTriangulation}

function Gridap.Geometry.is_change_possible(strian::GridapHybrid.SkeletonTriangulation,ttrian::SkeletonView)
    is_change_possible(strian,ttrian.parent)
end

function Gridap.Geometry.is_change_possible(strian::SkeletonView,ttrian::GridapHybrid.SkeletonTriangulation)
    false
end

function Gridap.Geometry.is_change_possible(strian::BodyFittedTriangulation,ttrian::SkeletonView)
    is_change_possible(strian,ttrian.parent)
end

function Gridap.Geometry.is_change_possible(strian::SkeletonView,ttrian::BodyFittedTriangulation)
    false
end

# Keeping ReferenceDomain here due to Julia ambiguity. PhysicalDomain can be incorporated if needed.
function Gridap.CellData.change_domain(
    a::CellField,strian::GridapHybrid.SkeletonTriangulation,::ReferenceDomain,ttrian::SkeletonView,::ReferenceDomain
)
    @assert is_change_possible(strian,ttrian)
    b = change_domain(a,strian,ReferenceDomain(),ttrian.parent,ReferenceDomain())

    data = Gridap.Geometry.restrict(CellData.get_data(b),ttrian.cell_to_parent_cell);
    Gridap.CellData.similar_cell_field(b,data,ttrian,ReferenceDomain())
end

function Gridap.CellData.change_domain(
    a::CellField,strian::BodyFittedTriangulation,::ReferenceDomain,ttrian::SkeletonView,::ReferenceDomain
)
    @assert is_change_possible(strian,ttrian)
    b = change_domain(a,strian,ReferenceDomain(),ttrian.parent,ReferenceDomain())

    data = Gridap.Geometry.restrict(CellData.get_data(b),ttrian.cell_to_parent_cell);
    Gridap.CellData.similar_cell_field(b,data,ttrian,ReferenceDomain())
end

function Gridap.CellData.change_domain(
    a::Gridap.MultiField.MultiFieldFEBasisComponent,
    ttrian::SkeletonView,
    tdomain::DomainStyle)
    if get_triangulation(a) == ttrian # SkeletonView to SkeletonView, return Idem
        return a
    end
    b = change_domain(a, ttrian.parent, tdomain)
    b_sf = b.single_field
    b_data = Gridap.Geometry.restrict(CellData.get_data(b_sf),ttrian.cell_to_parent_cell)
    c_sf = Gridap.CellData.similar_cell_field(b_sf,b_data,ttrian,ReferenceDomain())
    return Gridap.MultiField.MultiFieldFEBasisComponent(b_data,c_sf,a.fieldid,a.nfields)
end


# Cut the data to the local view
function Gridap.FESpaces.get_cell_fe_data(
    fun, f, ttrian::SkeletonView
)
    data = Gridap.FESpaces.get_cell_fe_data(fun,f,ttrian.parent)
    return Gridap.Geometry.restrict(data,ttrian.cell_to_parent_cell)
end



# cellwise normal vector for distributed hybrid models

function GridapHybrid.get_cell_normal_vector(s::GridapDistributed.DistributedTriangulation)
    fields = map(local_views(s)) do si
        parent = get_cell_normal_vector(si.parent)
        data = Geometry.restrict(CellData.get_data(parent),si.cell_to_parent_cell)
        GenericCellField(data,si,ReferenceDomain())
    end
    GridapDistributed.DistributedCellField(fields,s)
end

function GridapHybrid.get_cell_owner_normal_vector(s::GridapDistributed.DistributedTriangulation)
    fields = map(local_views(s)) do si
        parent = get_cell_owner_normal_vector(si.parent)
        data = Geometry.restrict(CellData.get_data(parent),si.cell_to_parent_cell)
        GenericCellField(data,si,ReferenceDomain())
    end
    GridapDistributed.DistributedCellField(fields,s)
end

function GridapHybrid.get_cell_normal_vector(s::GridapHybrid.SkeletonView)
    parent = get_cell_normal_vector(s.parent)
    data = Geometry.restrict(CellData.get_data(parent),s.cell_to_parent_cell)
    GenericCellField(data,s,ReferenceDomain())
end


# For distributed models we need to differentiate between Ωu and Ωp (they have different object ids)
# where u and p are coming from different spaces (example velocity and pressure). The serial implementation 
# does not require this differentiation.
function Geometry.best_target(
    trian1::BodyFittedTriangulation{Dc},
    trian2::BodyFittedTriangulation{Dc}
  ) where {Dc}
    @check is_change_possible(trian1,trian2)
    @check is_change_possible(trian2,trian1)
    D1 = num_cell_dims(trian1)
    D2 = num_cell_dims(trian2)
    glue1 = get_glue(trian1,Val(D2))
    glue2 = get_glue(trian2,Val(D1))
    best_target(trian1,trian2,glue1,glue2)
end



# Hybrid FE operator for distributed models
function HybridAffineFEOperator(
    weakform::Function,
    trial::GridapDistributed.DistributedMultiFieldFESpace,
    test::GridapDistributed.DistributedMultiFieldFESpace,
    bulk_fields::TB,
    skeleton_fields::TS) where {TB <: Vector{<:Integer},TS <: Vector{<:Integer}}
  
    # Invoke weak form of the hybridizable system
      u = get_trial_fe_basis(trial)
      v = get_fe_basis(test)
      biform, liform  = weakform(u, v)
  
      M, L = GridapHybrid._setup_fe_spaces_skeleton_system(trial, test, skeleton_fields)
      
    # Transform DomainContribution objects of the hybridizable system into a
    # suitable form for assembling the linear system defined on the skeleton
    # (i.e., the hybrid system)
      data = map(local_views(biform), local_views(liform),local_views(M),local_views(L)) do biform, liform, M, L
          obiform, oliform = GridapHybrid._merge_bulk_and_skeleton_contributions(biform, liform)
      # Pair LHS and RHS terms associated to SkeletonTriangulation
          matvec, mat, vec = Gridap.FESpaces._pair_contribution_when_possible(obiform, oliform)
      # Add StaticCondensationMap to matvec terms
          matvec = GridapHybrid._add_static_condensation(matvec, bulk_fields, skeleton_fields)
  
          if (length(skeleton_fields) != 1)
              matvec = GridapHybrid._block_skeleton_system_contributions(matvec, L)
          end
  
          # uhd = GridapHybrid.attach_zero(M)
          uhd = zero(M)
          matvec, mat = Gridap.FESpaces._attach_dirichlet(matvec, mat, uhd)
  
          data = Gridap.FESpaces._collect_cell_matrix_and_vector(M, L, matvec, mat, vec)
          return data
      end 
  
      assem = SparseMatrixAssembler(M, L)
      A, b = assemble_matrix_and_vector(assem, data)
  
      skeleton_op = AffineFEOperator(M, L, A, b)
      HybridAffineFEOperator(weakform, trial, test, bulk_fields, skeleton_fields, skeleton_op)
end
  

# solve for distributed hybrid models
function Gridap.FESpaces.solve!(uh::GridapDistributed.DistributedMultiFieldCellField, solver::LinearFESolver, op::HybridAffineFEOperator, cache)
    # Solve linear system defined on the skeleton
      lh = solve(op.skeleton_op)
  
    # Invoke weak form of the hybridizable system
      u = get_trial_fe_basis(op.trial)
      v = get_fe_basis(op.test)
      biform, liform  = op.weakform(u, v)
  
    # Transform DomainContribution objects of the hybridizable system into a
    # suitable form for assembling the linear system defined on the skeleton
    # (i.e., the hybrid system)
      obiform, oliform = _merge_bulk_and_skeleton_contributions(biform, liform)
  
    # Pair LHS and RHS terms associated to SkeletonTriangulation
      matvec, _, _ = Gridap.FESpaces._pair_contribution_when_possible(obiform, oliform)
  
      free_dof_values = _compute_hybridizable_from_skeleton_free_dof_values(
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



