
const unsigned int velocity_degree = 1;



#include <deal.II/base/quadrature_lib.h>
#include <deal.II/base/function.h>
#include <deal.II/base/logstream.h>
#include <deal.II/base/utilities.h>
#include <deal.II/base/conditional_ostream.h>
#include <deal.II/base/timer.h>
#include <deal.II/base/symmetric_tensor.h>
#include <deal.II/base/index_set.h>
#include <deal.II/base/parameter_handler.h>


//#include <deal.II/lac/constraint_matrix.h>

#include <deal.II/lac/affine_constraints.templates.h>

#include <deal.II/lac/vector.h>
#include <deal.II/lac/block_vector.h>
#include <deal.II/lac/solver_cg.h>
#include <deal.II/lac/solver_bicgstab.h>
#include <deal.II/lac/precondition.h>
#include <deal.II/lac/solver_gmres.h>
#include <deal.II/lac/generic_linear_algebra.h>

//#include <deal.II/grid/tria.h>
#include <deal.II/grid/tria_accessor.h>
#include <deal.II/grid/tria_iterator.h>
#include <deal.II/grid/grid_generator.h>
#include <deal.II/grid/grid_out.h>
//#include <deal.II/grid/tria_boundary_lib.h>
#include <deal.II/grid/grid_tools.h>
#include <deal.II/grid/grid_refinement.h>

#include <deal.II/dofs/dof_handler.h>
#include <deal.II/dofs/dof_renumbering.h>
#include <deal.II/dofs/dof_accessor.h>
#include <deal.II/dofs/dof_tools.h>


#include <deal.II/fe/fe_q.h>
#include <deal.II/fe/fe_values.h>
#include <deal.II/fe/fe_system.h>
#include <deal.II/fe/fe_dgq.h>


#include <deal.II/matrix_free/matrix_free.h>
#include <deal.II/matrix_free/operators.h>
#include <deal.II/matrix_free/fe_evaluation.h>

#include <deal.II/numerics/vector_tools.h>
#include <deal.II/numerics/matrix_tools.h>
#include <deal.II/numerics/data_out.h>
#include <deal.II/numerics/error_estimator.h>


#include <deal.II/distributed/tria.h>
#include <deal.II/distributed/grid_refinement.h>

#include <deal.II/multigrid/mg_constrained_dofs.h>
#include <deal.II/multigrid/multigrid.h>
#include <deal.II/multigrid/mg_transfer.h>
#include <deal.II/multigrid/mg_transfer_matrix_free.h>

#include <deal.II/multigrid/mg_tools.h>
#include <deal.II/multigrid/mg_coarse.h>
#include <deal.II/multigrid/mg_smoother.h>
#include <deal.II/multigrid/mg_matrix.h>



#include <cmath>
#include <fstream>
#include <iostream>
#include <fstream>
#include <sstream>

namespace LA
{
using namespace dealii::LinearAlgebraTrilinos;
}


namespace StokesClass
{
using namespace dealii;

class QuietException {};


template <int dim>
class BoundaryValues : public Function<dim>
{
public:
  BoundaryValues () : Function<dim>(dim) {}
  virtual void vector_value (const Point<dim> &p,
                             Vector<double>   &value) const;
};
template <int dim>
void
BoundaryValues<dim>::vector_value (const Point<dim> &p,
                                   Vector<double>   &values) const
{
  (void)p;
  for (unsigned int i=0; i<values.size(); ++i)
    values(i) = 0.0;
  return;
}



template <int dim, int degree_v, typename number>
class ABlockOperator
    : public MatrixFreeOperators::Base<dim, LinearAlgebra::distributed::Vector<number>>
{
public:
  ABlockOperator ();
  void clear ();

  void attach_for_diag (const MatrixFree<dim, number> &mf,
                        const AffineConstraints<number> &c);

  virtual void compute_diagonal ();

  void new_compute_diag (LinearAlgebra::distributed::Vector<number> &      dst,
                         const LinearAlgebra::distributed::Vector<number> &src);

private:
  virtual void apply_add (LinearAlgebra::distributed::Vector<number> &dst,
                          const LinearAlgebra::distributed::Vector<number> &src) const;

  void local_apply (const dealii::MatrixFree<dim, number> &data,
                    LinearAlgebra::distributed::Vector<number> &dst,
                    const LinearAlgebra::distributed::Vector<number> &src,
                    const std::pair<unsigned int, unsigned int> &cell_range) const;

  void local_compute_diagonal (const MatrixFree<dim,number>                     &data,
                               LinearAlgebra::distributed::Vector<number>  &dst,
                               const unsigned int                               &dummy,
                               const std::pair<unsigned int,unsigned int>       &cell_range) const;


  void new_compute_diag_local (const MatrixFree<dim, number> &             data,
                               LinearAlgebra::distributed::Vector<number> &dst,
                               const LinearAlgebra::distributed::Vector<number> & src,
                               const std::pair<unsigned int, unsigned int> &cell_range) const;

  MatrixFree<dim, number> mf_plain;
  AffineConstraints<number> constraints;

};
template <int dim, int degree_v, typename number>
ABlockOperator<dim,degree_v,number>::ABlockOperator ()
  :
    MatrixFreeOperators::Base<dim, LinearAlgebra::distributed::Vector<number> >()
{}
template <int dim, int degree_v, typename number>
void
ABlockOperator<dim,degree_v,number>::clear ()
{
  MatrixFreeOperators::Base<dim,LinearAlgebra::distributed::Vector<number> >::clear();
}

template <int dim, int degree_v, typename number>
void
ABlockOperator<dim,degree_v,number>::attach_for_diag(const MatrixFree<dim, number> &  mf,
                                                     const AffineConstraints<number> &c)
{
  mf_plain = mf;
  constraints.copy_from(c);
}

template <int dim, int degree_v, typename number>
void
ABlockOperator<dim,degree_v,number>
::local_apply (const dealii::MatrixFree<dim, number>                 &data,
               LinearAlgebra::distributed::Vector<number>       &dst,
               const LinearAlgebra::distributed::Vector<number> &src,
               const std::pair<unsigned int, unsigned int>           &cell_range) const
{
  FEEvaluation<dim,degree_v,degree_v+1,dim,number> velocity (data,0);

  for (unsigned int cell=cell_range.first; cell<cell_range.second; ++cell)
  {
    velocity.reinit (cell);
    velocity.read_dof_values(src);
    velocity.evaluate (false, true, false);
    for (unsigned int q=0; q<velocity.n_q_points; ++q)
    {
      velocity.submit_symmetric_gradient
          (2.0*velocity.get_symmetric_gradient(q),q);
    }
    velocity.integrate (false, true);
    velocity.distribute_local_to_global (dst);
  }
}
template <int dim, int degree_v, typename number>
void
ABlockOperator<dim,degree_v,number>
::apply_add (LinearAlgebra::distributed::Vector<number> &dst,
             const LinearAlgebra::distributed::Vector<number> &src) const
{
  MatrixFreeOperators::Base<dim,LinearAlgebra::distributed::Vector<number> >::
      data->cell_loop(&ABlockOperator::local_apply, this, dst, src);
}
template <int dim, int degree_v, typename number>
void
ABlockOperator<dim,degree_v,number>
::compute_diagonal ()
{
  this->inverse_diagonal_entries.
      reset(new DiagonalMatrix<LinearAlgebra::distributed::Vector<number> >());
  LinearAlgebra::distributed::Vector<number> &inverse_diagonal =
      this->inverse_diagonal_entries->get_vector();
  this->data->initialize_dof_vector(inverse_diagonal);
  unsigned int dummy = 0;
  this->data->cell_loop (&ABlockOperator::local_compute_diagonal, this,
                         inverse_diagonal, dummy);

  this->set_constrained_entries_to_one(inverse_diagonal);

  for (unsigned int i=0; i<inverse_diagonal.local_size(); ++i)
  {
    Assert(inverse_diagonal.local_element(i) > 0.,
           ExcMessage("No diagonal entry in a positive definite operator "
                      "should be zero"));

    inverse_diagonal.local_element(i) =
        1./inverse_diagonal.local_element(i);
  }
}
template <int dim, int degree_v, typename number>
void
ABlockOperator<dim,degree_v,number>
::local_compute_diagonal (const MatrixFree<dim,number>                     &data,
                          LinearAlgebra::distributed::Vector<number>  &dst,
                          const unsigned int                               &,
                          const std::pair<unsigned int,unsigned int>       &cell_range) const
{
  FEEvaluation<dim,degree_v,degree_v+1,dim,number> velocity (data, 0);
  for (unsigned int cell=cell_range.first; cell<cell_range.second; ++cell)
  {
    velocity.reinit (cell);
    AlignedVector<VectorizedArray<number> > diagonal(velocity.dofs_per_cell);
    for (unsigned int i=0; i<velocity.dofs_per_cell; ++i)
    {
      for (unsigned int j=0; j<velocity.dofs_per_cell; ++j)
        velocity.begin_dof_values()[j] = VectorizedArray<number>();
      velocity.begin_dof_values()[i] = make_vectorized_array<number> (1.);

      velocity.evaluate (false,true,false);
      for (unsigned int q=0; q<velocity.n_q_points; ++q)
      {
        velocity.submit_symmetric_gradient
            (2.0*velocity.get_symmetric_gradient(q),q);
      }
      velocity.integrate (false,true);

      diagonal[i] = velocity.begin_dof_values()[i];
    }

    for (unsigned int i=0; i<velocity.dofs_per_cell; ++i)
      velocity.begin_dof_values()[i] = diagonal[i];
    velocity.distribute_local_to_global (dst);
  }
}


template <int dim, int degree_v, typename number>
void
ABlockOperator<dim,degree_v,number>
::new_compute_diag(LinearAlgebra::distributed::Vector<number> &      dst,
                   const LinearAlgebra::distributed::Vector<number> &src)
{
  this->data->cell_loop(&ABlockOperator::new_compute_diag_local, this, dst, src);
}

template <int dim, int degree_v, typename number>
void
ABlockOperator<dim,degree_v,number>
::new_compute_diag_local(const MatrixFree<dim, number> &             data,
                         LinearAlgebra::distributed::Vector<number> &dst,
                         const LinearAlgebra::distributed::Vector<number> & /*src*/,
                         const std::pair<unsigned int, unsigned int> &cell_range) const
{
  const bool is_mg = false;

  FEEvaluation<dim, degree_v, degree_v + 1, dim, number> fe_eval(data);
  FEEvaluation<dim, degree_v, degree_v + 1, dim, number> fe_eval_plain(mf_plain);

  for (unsigned int cell = cell_range.first; cell < cell_range.second; ++cell)
  {
    fe_eval.reinit(cell);
    fe_eval_plain.reinit(cell);

    // 1) initialize empty local diagonal
    AlignedVector<VectorizedArray<number>> diagonal(fe_eval.dofs_per_cell,
                                                    make_vectorized_array<number>(0.0));

    // 2) get DoF-indices
    std::vector<types::global_dof_index> dof_indices[VectorizedArray<number>::n_array_elements];
    for (unsigned int v = 0; v < data.n_components_filled(cell); v++)
    {
      dof_indices[v].resize(fe_eval.dofs_per_cell);

      auto cell_v = data.get_cell_iterator(cell, v);

      if (is_mg)
        cell_v->get_mg_dof_indices(dof_indices[v]);
      else
        cell_v->get_dof_indices(dof_indices[v]);

      // in the case of CG: shape functions are not ordered
        // lexicographically see
        // (https://www.dealii.org/8.5.1/doxygen/deal.II/classFE__Q.html)
        // so we have to fix the order
        auto temp = dof_indices[v];
        for (unsigned int j = 0; j < dof_indices[v].size(); j++)
          dof_indices[v][j] = temp[data.get_shape_info().lexicographic_numbering[j]];
    }

    // 3) loop over all local DoFs and setup local diagonal entry by entry
    for (unsigned int i = 0; i < fe_eval.dofs_per_cell; ++i)
    {
      // 3a) zero out local source vector
      for (unsigned int j = 0; j < fe_eval.dofs_per_cell; ++j)
        fe_eval.begin_dof_values()[j] = make_vectorized_array<number>(0.0);

      // 3b) create standard basis taking care constraints
      for (unsigned int v = 0; v < data.n_components_filled(cell); v++)
        if (!constraints.is_constrained(dof_indices[v][i]))
        {
          fe_eval.begin_dof_values()[i][v] = 1.0;

          for (unsigned int ii = 0; ii < dof_indices[v].size(); ii++)
          {
            if (!constraints.is_constrained(dof_indices[v][ii]))
              continue;
            auto &cs = *constraints.get_constraint_entries(dof_indices[v][ii]);
            for (auto c : cs)
              if (c.first == dof_indices[v][i])
                fe_eval.begin_dof_values()[ii][v] = c.second;
          }
        }

      // 3c) perform stand matrix-free operation
      fe_eval.evaluate(false, true, false);
      for (unsigned int q = 0; q < fe_eval.n_q_points; ++q)
        fe_eval.submit_gradient(fe_eval.get_gradient(q), q);
      fe_eval.integrate(false, true);

      // 3d)
      VectorizedArray<number> temp = 0.0;
      for (unsigned int v = 0; v < data.n_components_filled(cell); v++)
        if (!constraints.is_constrained(dof_indices[v][i]))
        {
          temp = fe_eval.begin_dof_values()[i][v];

          for (unsigned int ii = 0; ii < dof_indices[v].size(); ii++)
          {
            if (!constraints.is_constrained(dof_indices[v][ii]))
              continue;
            auto &cs = *constraints.get_constraint_entries(dof_indices[v][ii]);
            for (auto c : cs)
              if (c.first == dof_indices[v][i])
                temp += fe_eval.begin_dof_values()[ii][v] * c.second;
          }
        }
      diagonal[i] = temp;
    }

    // 4) write local diagonal back to the global diagonal
    for (unsigned int i = 0; i < fe_eval_plain.dofs_per_cell; ++i)
      fe_eval_plain.begin_dof_values()[i] = diagonal[i];
    fe_eval_plain.distribute_local_to_global(dst);
  }
}



template <int dim>
class StokesProblem
{
public:
  StokesProblem ();

  void run ();

private:
  void make_grid ();
  void setup_system ();
  void assemble_system ();

  void get_ablock_diagonals ();

  void output_results (const unsigned int cycle) const;

  typedef LA::MPI::Vector mb_vector_t;
  typedef LinearAlgebra::distributed::Vector<double> mf_vector_t;

  typedef ABlockOperator<dim,velocity_degree,double>  MatrixFreeOperators;


  unsigned int                              degree;

  FESystem<dim>                             fe;

  parallel::distributed::Triangulation<dim> triangulation;
  DoFHandler<dim>                           dof_handler;

  std::vector<IndexSet>                     owned_partitioning;
  std::vector<IndexSet>                     relevant_partitioning;

  AffineConstraints<double>                          constraints;

  LA::MPI::SparseMatrix                system_matrix;

  MatrixFreeOperators                       matrix_free_matrix;

  mb_vector_t                      inv_diag_mb;
  mf_vector_t                      inv_diag_mf;
  mf_vector_t                      new_inv_diag_mf;

  ConditionalOStream                        pcout;
};


template <int dim>
StokesProblem<dim>::StokesProblem ()
  :
    degree (velocity_degree),
    fe (FE_Q<dim>(degree), dim),
    triangulation (MPI_COMM_WORLD,
                   typename Triangulation<dim>::MeshSmoothing
                   (Triangulation<dim>::limit_level_difference_at_vertices |
                    Triangulation<dim>::smoothing_on_refinement |
                    Triangulation<dim>::smoothing_on_coarsening),
                   parallel::distributed::Triangulation<dim>::construct_multigrid_hierarchy),
    dof_handler (triangulation),
    pcout (std::cout,
           (Utilities::MPI::this_mpi_process(MPI_COMM_WORLD)
            == 0))
{}

template <int dim>
void StokesProblem<dim>::make_grid()
{
  GridGenerator::hyper_shell (triangulation, Point<dim>(), 1, 2, 0, true);

  triangulation.refine_global (0);
}


template <int dim>
void StokesProblem<dim>::setup_system ()
{
  dof_handler.clear();
  dof_handler.distribute_dofs (fe);

  IndexSet locally_relevant_dofs;
  DoFTools::extract_locally_relevant_dofs (dof_handler,
                                           locally_relevant_dofs);

  constraints.reinit (locally_relevant_dofs);

  DoFTools::make_hanging_node_constraints (dof_handler,
                                           constraints);

  std::set<unsigned int> dirichlet_boundary = {0};
  std::set<unsigned int> tangential_boundary = {1};

  BoundaryValues<dim> boundary;
  for (auto bid : dirichlet_boundary)
    VectorTools::interpolate_boundary_values (dof_handler,
                                              bid,
                                              boundary,
                                              constraints);

  VectorTools::compute_no_normal_flux_constraints (dof_handler,
                                                   /* first_vector_component= */
                                                   0,
                                                   tangential_boundary,
                                                   constraints);
  constraints.close ();

  {
    system_matrix.clear ();

    DynamicSparsityPattern dsp (locally_relevant_dofs);

    DoFTools::make_sparsity_pattern (dof_handler, dsp,
                                     constraints, false);

    SparsityTools::distribute_sparsity_pattern (dsp,
                                                dof_handler.compute_n_locally_owned_dofs_per_processor(),
                                                MPI_COMM_WORLD,
                                                locally_relevant_dofs);


    system_matrix.reinit (dof_handler.locally_owned_dofs(),
                          dof_handler.locally_owned_dofs(),
                          dsp,
                          MPI_COMM_WORLD);
  }

  inv_diag_mb.reinit (dof_handler.locally_owned_dofs(), locally_relevant_dofs, MPI_COMM_WORLD);





  typename MatrixFree<dim,double>::AdditionalData additional_data;
  additional_data.tasks_parallel_scheme =
      MatrixFree<dim,double>::AdditionalData::none;
  additional_data.mapping_update_flags = (update_values | update_gradients |
                                          update_JxW_values | update_quadrature_points);

  std::shared_ptr<MatrixFree<dim,double> >
      mf_storage(new MatrixFree<dim,double>());
  mf_storage->reinit(dof_handler, constraints,
                     QGauss<1>(velocity_degree+1), additional_data);

  matrix_free_matrix.clear();
  matrix_free_matrix.initialize(mf_storage);
  matrix_free_matrix.compute_diagonal();

  matrix_free_matrix.initialize_dof_vector(inv_diag_mf);
  inv_diag_mf = matrix_free_matrix.get_matrix_diagonal_inverse()->get_vector();
  for (auto indx : dof_handler.locally_owned_dofs())
    if (constraints.is_constrained(indx))
      inv_diag_mf(indx) = 0.0;




  MatrixFree<dim, double> mf_plain;
  AffineConstraints<double> constraints_empty;
  mf_plain.reinit(dof_handler, constraints_empty, QGauss<1>(velocity_degree + 1), additional_data);

  matrix_free_matrix.attach_for_diag(mf_plain,constraints);

  mf_vector_t dummy_vec;
  matrix_free_matrix.initialize_dof_vector(dummy_vec);
  matrix_free_matrix.initialize_dof_vector(new_inv_diag_mf);
  matrix_free_matrix.new_compute_diag(new_inv_diag_mf,dummy_vec);

}



template <int dim>
void StokesProblem<dim>::assemble_system ()
{
  system_matrix = 0;

  const QGauss<dim>  quadrature_formula(degree+1);

  FEValues<dim> fealues (fe, quadrature_formula,
                         update_values    |  update_gradients |
                         update_quadrature_points |
                         update_JxW_values);


  const unsigned int   dofs_per_cell = fe.dofs_per_cell;
  const unsigned int   n_q_points    = quadrature_formula.size();

  FullMatrix<double>   cell_matrix (dofs_per_cell, dofs_per_cell);

  std::vector<SymmetricTensor<2,dim> > symgrad_phi_u (dofs_per_cell);

  std::vector<types::global_dof_index> local_dof_indices (dofs_per_cell);

  FEValuesExtractors::Vector velocities(0);

  typename DoFHandler<dim>::active_cell_iterator
      cell = dof_handler.begin_active(),
      endc = dof_handler.end();
  for (; cell!=endc; ++cell)
    if (cell->is_locally_owned())
    {
      cell->get_dof_indices (local_dof_indices);
      cell_matrix = 0;
      fealues.reinit (cell);

      for (unsigned int q=0; q<n_q_points; ++q)
      {
        for (unsigned int k=0; k<dofs_per_cell; ++k)
        {
          symgrad_phi_u[k] = fealues[velocities].symmetric_gradient (k, q);
        }

        for (unsigned int i=0; i<dofs_per_cell; ++i)
          for (unsigned int j=0; j<dofs_per_cell; ++j)
          {
            cell_matrix(i,j) += ((2*(symgrad_phi_u[i]*symgrad_phi_u[j])
                                  * fealues.JxW(q)));
          }
      }

      constraints.distribute_local_to_global (cell_matrix,
                                              local_dof_indices,
                                              system_matrix);
    }

  system_matrix.compress (VectorOperation::add);


  // Matrix-based diagonal
  for (cell = dof_handler.begin_active(); cell!=endc; ++cell)
    if (cell->is_locally_owned())
    {
      cell->get_dof_indices (local_dof_indices);
      for (unsigned int i=0; i<dofs_per_cell; ++i)
        if (!constraints.is_constrained(local_dof_indices[i]))
        {
          inv_diag_mb(local_dof_indices[i]) = 1.0/system_matrix.diag_element(local_dof_indices[i]);
        }
    }
}


template <int dim>
void StokesProblem<dim>::output_results (const unsigned int cycle) const
{
  GridOut grid_out;
  grid_out.write_mesh_per_processor_as_vtu(triangulation,"mg-grid"+Utilities::int_to_string(cycle),true,false);

  std::ofstream grid("active-grid"+Utilities::int_to_string(cycle)+".vtu");
  grid_out.write_vtu(triangulation,grid);
}




template <int dim>
void StokesProblem<dim>::run ()
{
  make_grid ();

  setup_system ();
  assemble_system();

  //inv_diag_mb.print(std::cout);
  std::cout << "Matrix-based:" << std::endl;
  for (unsigned int i=0; i<inv_diag_mb.size(); ++i)
    std::cout << inv_diag_mb(i) << std::endl;

  std::cout << std::endl;

  //inv_diag_mf.print(std::cout);
  std::cout << "Matrix-free:" << std::endl;
  for (unsigned int i=0; i<inv_diag_mf.size(); ++i)
    std::cout << inv_diag_mf(i) << std::endl;

  std::cout << std::endl;

  //new_inv_diag_mf.print(std::cout);
  std::cout << "NEW Matrix-free:" << std::endl;
  for (unsigned int i=0; i<new_inv_diag_mf.size(); ++i)
    std::cout << new_inv_diag_mf(i) << std::endl;

  output_results(0);
}
}


int main(int argc, char *argv[])
{
  try
  {
    dealii::Utilities::MPI::MPI_InitFinalize mpi_initialization(argc, argv, 1);

    if (dealii::Utilities::MPI::this_mpi_process(MPI_COMM_WORLD) == 0)
      std::cout << "Running with "
                << dealii::Utilities::MPI::n_mpi_processes(MPI_COMM_WORLD)
                << "processors." << std::endl;

    StokesClass::StokesProblem<2> problem;
    problem.run ();
  }
  catch (std::exception &exc)
  {
    std::cerr << std::endl << std::endl
              << "----------------------------------------------------"
              << std::endl;
    std::cerr << "Exception on processing: " << std::endl
              << exc.what() << std::endl
              << "Aborting!" << std::endl
              << "----------------------------------------------------"
              << std::endl;

    return 1;
  }
  catch (...)
  {
    std::cerr << std::endl << std::endl
              << "----------------------------------------------------"
              << std::endl;
    std::cerr << "Unknown exception!" << std::endl
              << "Aborting!" << std::endl
              << "----------------------------------------------------"
              << std::endl;
    return 1;
  }

  return 0;
}
