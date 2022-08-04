__author__ = "Mikael Mortensen <mikaem@math.uio.no>"
__date__ = "2013-11-06"
__copyright__ = "Copyright (C) 2013 " + __author__
__license__ = "GNU Lesser GPL version 3 or any later version"

from dolfin import *
#import gc #garbage collector (us gc.collect() after del to prevent memory leak)
from ..NSfracStep import *
from ..NSfracStep import __all__

import numpy

def setup(u_components,backflow_flag,nonNewtonian_flag, beta_backflow,Resistance_flag,n_normal,facet_domains, u, v, p, q, bcs, les_model, nu, nut_,
          scalar_components, V, Q, x_, p_, u_,u_1, A_cache,
          velocity_update_solver, assemble_matrix, homogenize,
          GradFunction, DivFunction, LESsource,mesh, **NS_namespace):
    """Preassemble mass and diffusion matrices.

    Set up and prepare all equations to be solved. Called once, before
    going into time loop.

    """
    # Mass matrix
    M = assemble_matrix(inner(u, v) * dx)
    


    # Stiffness matrix (without viscosity coefficient)

    #if(backflow_flag):
    #    ds = Measure("ds")[facet_domains]
    #    K = lhs( inner(grad(u), grad(v)) * dx - 2.0/nu*beta_backflow *  inner(v, ( dot(u_,n_normal) - abs( dot(u_,n_normal) ) )/2. * u ) * (ds(3, domain=mesh, subdomain_data=facet_domains) + ds(4, domain=mesh, subdomain_data=facet_domains) )   )
    #else:
    #    K = assemble_matrix(inner(grad(u), grad(v)) * dx)

    # Allocate stiffness matrix for LES that changes with time
    KT = None if les_model is "NoModel" else (
        Matrix(M), inner(grad(u), grad(v)))

    # Pressure Laplacian.
    if (Resistance_flag):
     Ap2 = inner(grad(q), grad(p)) * dx
     Ap = assemble_matrix(inner(grad(q), grad(p)) * dx, bcs['p'])#Ap=[]
    else:
     Ap = assemble_matrix(inner(grad(q), grad(p)) * dx, bcs['p'])
     Ap2 = []

    # if les_model is "NoModel":
    # if not Ap.id() == K.id():
    # Compress matrix (creates new matrix)
    #Bp = Matrix()
    # Ap.compressed(Bp)
    #Ap = Bp
    # Replace cached matrix with compressed version
    #A_cache[(inner(grad(q), grad(p))*dx, tuple(bcs['p']))] = Ap

    # Allocate coefficient matrix (needs reassembling)
    A = Matrix(M)

    # Allocate Function for holding and computing the velocity divergence on Q
    divu = DivFunction(u_, Q, name='divu',
                       method=velocity_update_solver)

    # Allocate a dictionary of Functions for holding and computing pressure gradients
    gradp = {ui: GradFunction(p_, V, i=i, name='dpd' + ('x', 'y', 'z')[i],
                              bcs=homogenize(bcs[ui]),
                              method=velocity_update_solver)
             for i, ui in enumerate(u_components)}

    # Create dictionary to be returned into global NS namespace

    d = dict(A=A, M=M, Ap=Ap,Ap2=Ap2, divu=divu, gradp=gradp) #  d = dict(A=A, M=M, K=K, Ap=Ap, divu=divu, gradp=gradp)

    # Allocate coefficient matrix and work vectors for scalars. Matrix differs
    # from velocity in boundary conditions only
    if len(scalar_components) > 0:
        d.update(Ta=Matrix(M))
        if len(scalar_components) > 1:
            # For more than one scalar we use the same linear algebra solver for all.
            # For this to work we need some additional tensors. The extra matrix
            # is required since different scalars may have different boundary conditions
            Tb = Matrix(M)
            bb = Vector(x_[scalar_components[0]])
            bx = Vector(x_[scalar_components[0]])
            d.update(Tb=Tb, bb=bb, bx=bx)

    # Setup for solving convection
    u_ab = as_vector([Function(V) for i in range(len(u_components))])
    u_ab_prev = as_vector([Function(V) for i in range(len(u_components))]) #used for nonNewtonian Gamma calclulation
    a_conv = inner(v, dot(u_ab, nabla_grad(u))) * dx
    a_scalar = a_conv
    LT = None if les_model is "NoModel" else LESsource(
        nut_, u_ab, V, name='LTd')

    if bcs['p'] == []:
        attach_pressure_nullspace(Ap, x_, Q)


    if(nonNewtonian_flag and backflow_flag):
        visc_gamma = pow(0.5*inner(grad(u_ab_prev)+transpose(grad(u_ab_prev)), grad(u_ab_prev)+transpose(grad(u_ab_prev))), 0.5)
        #visc_gamma = pow(0.5*Strain(u_ab_prev), 0.5)
        #Carreau-Yasuda model. Cho and Kensey, 1991
        visc_nu = (1./1.06) * (0.0345 + (0.56 - 0.0345) * (1. + (1.902*visc_gamma)**1.25 )**( (0.22-1.)/1.25 )  )
        #K_weak = visc_nu * inner(grad(u), grad(v)) * dx #need to assemble again due to viscossity change
        ds = Measure("ds")[facet_domains]
        K = lhs( visc_nu*inner(grad(u), grad(v)) * dx - 2.0*beta_backflow *  inner(v, ( dot(u_,n_normal) - abs( dot(u_,n_normal) ) )/2. * u ) * (ds(3, domain=mesh, subdomain_data=facet_domains) + ds(4, domain=mesh, subdomain_data=facet_domains) )   )
    elif(nonNewtonian_flag and not backflow_flag):
         visc_gamma = pow(0.5*inner(grad(u_ab_prev)+transpose(grad(u_ab_prev)), grad(u_ab_prev)+transpose(grad(u_ab_prev))), 0.5)
         #visc_gamma = pow(0.5*Strain(u_ab_prev), 0.5)
         #Carreau-Yasuda model
         visc_nu = (1./1.06) * (0.0345 + (0.56 - 0.0345) * (1. + (1.902*visc_gamma)**1.25 )**( (0.22-1.)/1.25 )  )
         K = visc_nu * inner(grad(u), grad(v)) * dx
    elif(backflow_flag and not nonNewtonian_flag):
         #K_weak = inner(grad(u), grad(v)) * dx - 2.0/nu*beta_backflow *  inner(v, ( dot(u_1,n_normal) - abs( dot(u_1,n_normal) ) )/2. * u ) * (ds(3, domain=mesh, subdomain_data=facet_domains) + ds(4, domain=mesh, subdomain_data=facet_domains)) #beta_backflow = 2 and 10 works (at least till t=0.44)
         #K_weak_lhs = lhs(K_weak)
         #K_back = assemble_matrix(K_weak_lhs)
         #A.axpy(-0.5*nu, K_back, True)
         #del K_back
         #gc.collect() #still memory leak!
         #del K_weak_lhs
         #gc.collect()
         #del K_weak
         #gc.collect()
         ds = Measure("ds")(subdomain_data=facet_domains)
         K = lhs( nu*inner(grad(u), grad(v)) * dx - 2.0*beta_backflow *  inner(v, ( dot(u_,n_normal) - abs( dot(u_,n_normal) ) )/2. * u ) * ( ds(3, domain=mesh, subdomain_data=facet_domains) + ds(4, domain=mesh, subdomain_data=facet_domains) + ds(5, domain=mesh, subdomain_data=facet_domains) + ds(6, domain=mesh, subdomain_data=facet_domains) + ds(7, domain=mesh, subdomain_data=facet_domains) + ds(8, domain=mesh, subdomain_data=facet_domains) + ds(9, domain=mesh, subdomain_data=facet_domains) + ds(10, domain=mesh, subdomain_data=facet_domains) + ds(11, domain=mesh, subdomain_data=facet_domains) + ds(12, domain=mesh, subdomain_data=facet_domains) + ds(13, domain=mesh, subdomain_data=facet_domains) + ds(14, domain=mesh, subdomain_data=facet_domains) + ds(15, domain=mesh, subdomain_data=facet_domains) + ds(16, domain=mesh, subdomain_data=facet_domains) + ds(17, domain=mesh, subdomain_data=facet_domains) + ds(18, domain=mesh, subdomain_data=facet_domains) + ds(19, domain=mesh, subdomain_data=facet_domains) + ds(20, domain=mesh, subdomain_data=facet_domains) + ds(21, domain=mesh, subdomain_data=facet_domains) + ds(22, domain=mesh, subdomain_data=facet_domains) + ds(23, domain=mesh, subdomain_data=facet_domains) + ds(24, domain=mesh, subdomain_data=facet_domains) + ds(25, domain=mesh, subdomain_data=facet_domains) + ds(26, domain=mesh, subdomain_data=facet_domains) + ds(27, domain=mesh, subdomain_data=facet_domains) + ds(28, domain=mesh, subdomain_data=facet_domains) + ds(29, domain=mesh, subdomain_data=facet_domains) + ds(30, domain=mesh, subdomain_data=facet_domains) + ds(31, domain=mesh, subdomain_data=facet_domains) + ds(32, domain=mesh, subdomain_data=facet_domains) + ds(33, domain=mesh, subdomain_data=facet_domains) )   )
         #K = lhs( nu*inner(grad(u), grad(v)) * dx - 2.0*beta_backflow *  inner(v, ( dot(u_,n_normal) - abs( dot(u_,n_normal) ) )/2. * u ) * ( ds(5, domain=mesh, subdomain_data=facet_domains) )   )
         visc_nu = nu
    else:
         K = assemble_matrix(inner(grad(u), grad(v)) * dx)
         visc_nu = nu

    #test visc_gamma output (also pass K_test and K_test_rhs in update below)
    #K_test = q*gamma_soln*dx
    #K_test_rhs = q*visc_gamma*dx
    d.update(u_ab=u_ab,u_ab_prev=u_ab_prev, a_conv=a_conv, a_scalar=a_scalar, LT=LT, KT=KT,K=K,visc_nu = visc_nu)
    return d

def get_solvers(use_krylov_solvers, krylov_solvers,krylov_solvers_P, bcs,
                x_, Q, scalar_components, velocity_krylov_solver,
                pressure_krylov_solver, scalar_krylov_solver, **NS_namespace):
    """Return linear solvers.

    We are solving for
       - tentative velocity
       - pressure correction

       and possibly:
       - scalars

    """
    if use_krylov_solvers:
        ## tentative velocity solver ##
        u_prec = PETScPreconditioner(
            velocity_krylov_solver['preconditioner_type'])
        u_sol = PETScKrylovSolver(
            velocity_krylov_solver['solver_type'], u_prec)
        u_sol.prec = u_prec  # Keep from going out of scope
        # u_sol = KrylovSolver(velocity_krylov_solver['solver_type'],
        #                     velocity_krylov_solver['preconditioner_type'])
        #u_sol.parameters['preconditioner']['structure'] = 'same_nonzero_pattern'
        u_sol.parameters.update(krylov_solvers)

        ## pressure solver ##
        #p_prec = PETScPreconditioner('hypre_amg')
        #p_prec.parameters['report'] = True
        #p_prec.parameters['hypre']['BoomerAMG']['agressive_coarsening_levels'] = 0
        #p_prec.parameters['hypre']['BoomerAMG']['strong_threshold'] = 0.5
        #PETScOptions.set('pc_hypre_boomeramg_truncfactor', 0)
        #PETScOptions.set('pc_hypre_boomeramg_agg_num_paths', 1)
        p_sol = KrylovSolver(pressure_krylov_solver['solver_type'],
                             pressure_krylov_solver['preconditioner_type'])
        #p_sol.parameters['preconditioner']['structure'] = 'same'
        #p_sol.parameters['profile'] = True

        #p_sol.parameters.update(krylov_solvers)
        p_sol.parameters.update(krylov_solvers_P)

        sols = [u_sol, p_sol]
        ## scalar solver ##
        if len(scalar_components) > 0:
            c_prec = PETScPreconditioner(
                scalar_krylov_solver['preconditioner_type'])
            c_sol = PETScKrylovSolver(
                scalar_krylov_solver['solver_type'], c_prec)
            c_sol.prec = c_prec
            # c_sol = KrylovSolver(scalar_krylov_solver['solver_type'],
            # scalar_krylov_solver['preconditioner_type'])
            c_sol.parameters.update(krylov_solvers)
            #c_sol.parameters['preconditioner']['structure'] = 'same_nonzero_pattern'
            sols.append(c_sol)
        else:
            sols.append(None)
    else:
        ## tentative velocity solver ##
        u_sol = LUSolver()
        u_sol.parameters['same_nonzero_pattern'] = True
        ## pressure solver ##
        p_sol = LUSolver()
        p_sol.parameters['reuse_factorization'] = True
        if bcs['p'] == []:
            p_sol.normalize = True
        sols = [u_sol, p_sol]
        ## scalar solver ##
        if len(scalar_components) > 0:
            c_sol = LUSolver()
            sols.append(c_sol)
        else:
            sols.append(None)

    return sols


def assemble_first_inner_iter(nonNewtonian_flag,backflow_flag, beta_backflow,mesh,facet_domains,n_normal, A, a_conv, dt, M, scalar_components, les_model,
                              a_scalar, K, nu, nut_, u_components, LT, KT,
                              b_tmp, b0, x_1, x_2, u_ab,u_ab_prev,u_,u_1,assemble_matrix, u,v, bcs, **NS_namespace):
    """Called on first inner iteration of velocity/pressure system.

    Assemble convection matrix, compute rhs of tentative velocity and
    reset coefficient matrix for solve.

    """
    t0 = Timer("Assemble first inner iter")
    # Update u_ab used as convecting velocity
    for i, ui in enumerate(u_components):
        u_ab[i].vector().zero()
        u_ab[i].vector().axpy(1.5, x_1[ui])
        u_ab[i].vector().axpy(-0.5, x_2[ui])

    A = assemble(a_conv, tensor=A)
    A._scale(-0.5)            # Negative convection on the rhs
    A.axpy(1. / dt, M, True)  # Add mass


    # Set up scalar matrix for rhs using the same convection as velocity
    if len(scalar_components) > 0:
        Ta = NS_namespace['Ta']
        if a_scalar is a_conv:
            Ta.zero()
            Ta.axpy(1., A, True)
                
    #ds = Measure("ds")[facet_domains]
    # Add diffusion and compute rhs for all velocity components
    if(nonNewtonian_flag):
      for i, ui in enumerate(u_components):
        u_ab_prev[i].vector().zero()
        u_ab_prev[i].vector().axpy(1.0, x_1[ui])

    if (not nonNewtonian_flag and not backflow_flag):
       A.axpy(-0.5*nu, K, True)
    else:
       K2 = assemble(K)   #assemble_matrix(K)  # assemble_matrix will not update variables?
       A.axpy(-0.5, K2, True)

    #file_visc_output = File('/Users/aa3878/data/oasis_2017_changed/AAA_results_nonNewtonian/visc/' + 'visc_mu.pvd')
    #visc_nonN = project(visc_nu, FunctionSpace(mesh, "CG",1),solver_type='cg')
    #file_visc_output << visc_nonN

    if not les_model is "NoModel":
        assemble(nut_ * KT[1] * dx, tensor=KT[0])
        A.axpy(-0.5, KT[0], True)

    for i, ui in enumerate(u_components):
        # Start with body force
        b_tmp[ui].zero()
        b_tmp[ui].axpy(1., b0[ui])
        # Add transient, convection and diffusion
        b_tmp[ui].axpy(1., A * x_1[ui])
        if not les_model is "NoModel":
            LT.assemble_rhs(i)
            b_tmp[ui].axpy(1., LT.vector())

    # Reset matrix for lhs
    A._scale(-1.)
    A.axpy(2. / dt, M, True)
    [bc.apply(A) for bc in bcs['u0']]

def attach_pressure_nullspace(Ap, x_, Q):
    """Create null space basis object and attach to Krylov solver."""
    null_vec = Vector(x_['p'])
    Q.dofmap().set(null_vec, 1.0)
    null_vec *= 1.0 / null_vec.norm('l2')
    Aa = as_backend_type(Ap)
    null_space = VectorSpaceBasis([null_vec])
    Aa.set_nullspace(null_space)
    Aa.null_space = null_space


def velocity_tentative_assemble(ui, b, b_tmp, p_, gradp, **NS_namespace):
    """Add pressure gradient to rhs of tentative velocity system."""
    b[ui].zero()
    b[ui].axpy(1., b_tmp[ui])
    gradp[ui].assemble_rhs(p_)
    b[ui].axpy(-1., gradp[ui].rhs)

def velocity_tentative_solve(ui, A, bcs, x_, x_2, u_sol, b, udiff,
                             use_krylov_solvers, **NS_namespace):
    """Linear algebra solve of tentative velocity component."""
    #if use_krylov_solvers:
        #if ui == 'u0':
            #u_sol.parameters['preconditioner']['structure'] = 'same_nonzero_pattern'
        #else:
            #u_sol.parameters['preconditioner']['structure'] = 'same'
    [bc.apply(b[ui]) for bc in bcs[ui]]
    # x_2 only used on inner_iter 1, so use here as work vector
    x_2[ui].zero()
    x_2[ui].axpy(1., x_[ui])
    t1 = Timer("Tentative Linear Algebra Solve")
    u_sol.solve(A, x_[ui], b[ui])
    t1.stop()
    udiff[0] += norm(x_2[ui] - x_[ui])


def pressure_assemble(Resistance_flag,Q,n_normal,facet_domains,mesh,bcs,assemble_matrix, u_,b, x_, dt, Ap,Ap2, divu, **NS_namespace):
    """Assemble rhs of pressure equation."""
    #if (Resistance_flag):
    #   ds = Measure("ds")[facet_domains]
    #   Res_bc1 = Res1 * abs( assemble(dot(u_, n_normal)*ds(3, domain=mesh, subdomain_data=facet_domains) ) )
    #   Res_bc2 = Res2 * abs( assemble(dot(u_, n_normal)*ds(4, domain=mesh, subdomain_data=facet_domains) ) )
    #   print '+++++++++++++++++++++++Res_bc1', Res_bc1
    #   print '+++++++++++++++++++++++Res_bc2', Res_bc2
    #   bcs['p'] = [DirichletBC(Q,  Res_bc1 , facet_domains,3), DirichletBC(Q,  Res_bc2, facet_domains,4)]
    #   Ap = assemble_matrix(Ap2, bcs['p'])
    divu.assemble_rhs()  # Computes div(u_)*q*dx
    b['p'][:] = divu.rhs
    b['p']._scale(-1. / dt)
    b['p'].axpy(1., Ap * x_['p'])


def pressure_solve(Resistance_flag, Res1, Res2, Res3, Res4, Res5, Res6, Res7, Res8, Res9, Res10, Res11, Res12, Res13, Res14, Res15, Res16, Res17, Res18, Res19, Res20, Res21, Res22, Res23, Res24, Res25, Res26, Res27, Res28, Res29, Res30, Res31, tstep, Q,n_normal,facet_domains,mesh,assemble_matrix,u_,dp_, x_, Ap,Ap2, b, p_sol, bcs, **NS_namespace):
    """Solve pressure equation."""
    if (Resistance_flag and tstep>1):
     if MPI.rank(mpi_comm_world()) == 0:
         print 'Res1_initial = ', Res1, 'Res2_initial = ', Res2, 'Res3_initial = ', Res3, 'Res4_initial = ', Res4
         print 'Res5_initial = ', Res5, 'Res6_initial = ', Res6, 'Res7_initial = ', Res7, 'Res8_initial = ', Res8
         print 'Res9_initial = ', Res9, 'Res10_initial = ', Res10, 'Res11_initial = ', Res11, 'Res12_initial = ', Res12
         print 'Res13_initial = ', Res13, 'Res14_initial = ', Res14, 'Res15_initial = ', Res15, 'Res16_initial = ', Res16
         print 'Res17_initial = ', Res17, 'Res18_initial = ', Res18, 'Res19_initial = ', Res19, 'Res20_initial = ', Res20
         print 'Res21_initial = ', Res21, 'Res22_initial = ', Res22, 'Res23_initial = ', Res23, 'Res24_initial = ', Res24
         print 'Res25_initial = ', Res25, 'Res26_initial = ', Res26, 'Res27_initial = ', Res27, 'Res28_initial = ', Res28
         print 'Res29_initial = ', Res29, 'Res30_initial = ', Res30, 'Res31_initial = ', Res31, 

#     Res1_dum = Res1
#     Res2_dum = Res2
#     Res3_dum = Res3
#     Res4_dum = Res4

#     if tstep > 20:
#       Res1 = Res1_dum
#       Res2 = Res2_dum
#       Res3 = Res3_dum
#       Res4 = Res4_dum

#       if MPI.rank(mpi_comm_world()) == 0:
#          print 'Res1_dum = ', Res1_dum, 'Res2_dum = ', Res2_dum, 'Res3_dum = ', Res3_dum, 'Res4_dum = ', Res4_dum

       #ds = Measure("ds")[facet_domains]
     ds = Measure("ds")(subdomain_data=facet_domains)
     flow1 = abs( assemble(dot(u_, n_normal)*ds(3, domain=mesh, subdomain_data=facet_domains) ) )
     flow2 = abs( assemble(dot(u_, n_normal)*ds(4, domain=mesh, subdomain_data=facet_domains) ) )
     flow3 = abs( assemble(dot(u_, n_normal)*ds(5, domain=mesh, subdomain_data=facet_domains) ) )
     flow4 = abs( assemble(dot(u_, n_normal)*ds(6, domain=mesh, subdomain_data=facet_domains) ) )
     flow5 = abs( assemble(dot(u_, n_normal)*ds(7, domain=mesh, subdomain_data=facet_domains) ) )
     flow6 = abs( assemble(dot(u_, n_normal)*ds(8, domain=mesh, subdomain_data=facet_domains) ) )
     flow7 = abs( assemble(dot(u_, n_normal)*ds(9, domain=mesh, subdomain_data=facet_domains) ) )
     flow8 = abs( assemble(dot(u_, n_normal)*ds(10, domain=mesh, subdomain_data=facet_domains) ) )
     flow9 = abs( assemble(dot(u_, n_normal)*ds(11, domain=mesh, subdomain_data=facet_domains) ) )
     flow10 = abs( assemble(dot(u_, n_normal)*ds(12, domain=mesh, subdomain_data=facet_domains) ) )
     flow11 = abs( assemble(dot(u_, n_normal)*ds(13, domain=mesh, subdomain_data=facet_domains) ) )
     flow12 = abs( assemble(dot(u_, n_normal)*ds(14, domain=mesh, subdomain_data=facet_domains) ) )
     flow13 = abs( assemble(dot(u_, n_normal)*ds(15, domain=mesh, subdomain_data=facet_domains) ) )
     flow14 = abs( assemble(dot(u_, n_normal)*ds(16, domain=mesh, subdomain_data=facet_domains) ) )
     flow15 = abs( assemble(dot(u_, n_normal)*ds(17, domain=mesh, subdomain_data=facet_domains) ) )
     flow16 = abs( assemble(dot(u_, n_normal)*ds(18, domain=mesh, subdomain_data=facet_domains) ) )
     flow17 = abs( assemble(dot(u_, n_normal)*ds(19, domain=mesh, subdomain_data=facet_domains) ) )
     flow18 = abs( assemble(dot(u_, n_normal)*ds(20, domain=mesh, subdomain_data=facet_domains) ) )
     flow19 = abs( assemble(dot(u_, n_normal)*ds(21, domain=mesh, subdomain_data=facet_domains) ) )
     flow20 = abs( assemble(dot(u_, n_normal)*ds(22, domain=mesh, subdomain_data=facet_domains) ) )
     flow21 = abs( assemble(dot(u_, n_normal)*ds(23, domain=mesh, subdomain_data=facet_domains) ) )
     flow22 = abs( assemble(dot(u_, n_normal)*ds(24, domain=mesh, subdomain_data=facet_domains) ) )
     flow23 = abs( assemble(dot(u_, n_normal)*ds(25, domain=mesh, subdomain_data=facet_domains) ) )
     flow24 = abs( assemble(dot(u_, n_normal)*ds(26, domain=mesh, subdomain_data=facet_domains) ) )
     flow25 = abs( assemble(dot(u_, n_normal)*ds(27, domain=mesh, subdomain_data=facet_domains) ) )
     flow26 = abs( assemble(dot(u_, n_normal)*ds(28, domain=mesh, subdomain_data=facet_domains) ) )
     flow27 = abs( assemble(dot(u_, n_normal)*ds(29, domain=mesh, subdomain_data=facet_domains) ) )
     flow28 = abs( assemble(dot(u_, n_normal)*ds(30, domain=mesh, subdomain_data=facet_domains) ) )
     flow29 = abs( assemble(dot(u_, n_normal)*ds(31, domain=mesh, subdomain_data=facet_domains) ) )
     flow30 = abs( assemble(dot(u_, n_normal)*ds(32, domain=mesh, subdomain_data=facet_domains) ) )
     flow31 = abs( assemble(dot(u_, n_normal)*ds(33, domain=mesh, subdomain_data=facet_domains) ) )
     inflow = abs( assemble(dot(u_, n_normal)*ds(1, domain=mesh, subdomain_data=facet_domains) ) )


#       total_flow = flow1 + flow2 + flow3 + flow4
#       ratios = numpy.zeros(4)
#       current_ratios = numpy.zeros(4)
#       ratios_threshold = numpy.zeros(4) 
#       target_ratios = [0.4264, 0.2871, 0.1455, 0.141]

#       current_ratios[0] = flow1/(inflow + 1e-16)
#       current_ratios[1] = flow2/(inflow + 1e-16)
#       current_ratios[2] = flow3/(inflow + 1e-16)
#       current_ratios[3] = flow4/(inflow + 1e-16)

#       for i in xrange(4):
#            ratios[i] = current_ratios[i]/target_ratios[i]
#            ratios_threshold[i] = ratios[i] < 0.99 or ratios[i] > 1.01
	    
#	    if ratios_threshold[0]:
#		Res1 *= ratios[0]
#		if MPI.rank(mpi_comm_world()) == 0:
#           	   print 'Res1 has changed!'
#	    if not ratios_threshold[0]:
#		if ratios_threshold[1]:
#		   Res2 *= ratios[1]
#                   if MPI.rank(mpi_comm_world()) == 0:
#                      print 'Res2 has changed!'
#		if not ratios_threshold[1]:
#		     if ratios_threshold[2]:
#                        Res3 *= ratios[2]
#                        if MPI.rank(mpi_comm_world()) == 0:
#                           print 'Res3 has changed!'
#                     if not ratios_threshold[2]:
#			if ratios_threshold[3]:
#                           Res4 *= ratios[3]
#                           if MPI.rank(mpi_comm_world()) == 0:
#                              print 'Res4 has changed!'
#	    Res1_dum = Res1
#	    Res2_dum = Res2
#	    Res3_dum = Res3
#	    Res4_dum = Res4		

#            if ratios_threshold[i] and i == 3:
#              Res4 *= ratios[i]
#              if MPI.rank(mpi_comm_world()) == 0:
#                print 'Res4 has changed!'
#            elif ratios_threshold[i] and i == 2:
#              Res3 *= ratios[i] 
#              if MPI.rank(mpi_comm_world()) == 0:
#                print 'Res3 has changed!'
#            elif ratios_threshold[i] and i == 1:
#              Res2 *= ratios[i] 
#              if MPI.rank(mpi_comm_world()) == 0:
#                print 'Res2 has changed!'
#            elif ratios_threshold[i] and i == 0:
#              Res1 *= ratios[i]
#              if MPI.rank(mpi_comm_world()) == 0:
#                print 'Res1 has changed!'

     Res_bc1 =  Res1 * flow1
     Res_bc2 =  Res2 * flow2
     Res_bc3 =  Res3 * flow3
     Res_bc4 =  Res4 * flow4
     Res_bc5 =  Res5 * flow5
     Res_bc6 =  Res6 * flow6
     Res_bc7 =  Res7 * flow7
     Res_bc8 =  Res8 * flow8
     Res_bc9 =  Res9 * flow9
     Res_bc10 =  Res10 * flow10
     Res_bc11 =  Res11 * flow11
     Res_bc12 =  Res12 * flow12
     Res_bc13 =  Res13 * flow13
     Res_bc14 =  Res14 * flow14
     Res_bc15 =  Res15 * flow15
     Res_bc16 =  Res16 * flow16
     Res_bc17 =  Res17 * flow17
     Res_bc18 =  Res18 * flow18
     Res_bc19 =  Res19 * flow19
     Res_bc20 =  Res20 * flow20
     Res_bc21 =  Res21 * flow21
     Res_bc22 =  Res22 * flow22
     Res_bc23 =  Res23 * flow23
     Res_bc24 =  Res24 * flow24
     Res_bc25 =  Res25 * flow25
     Res_bc26 =  Res26 * flow26
     Res_bc27 =  Res27 * flow27
     Res_bc28 =  Res28 * flow28
     Res_bc29 =  Res29 * flow29
     Res_bc30 =  Res30 * flow30
     Res_bc31 =  Res31 * flow31

     if MPI.rank(mpi_comm_world()) == 0:
        print '+++++++++++++++++++++++ Res_bc: flow1:', flow1
        print '+++++++++++++++++++++++ Res_bc: flow2:', flow2
        print '+++++++++++++++++++++++ Res_bc: flow3:', flow3
        print '+++++++++++++++++++++++ Res_bc: flow4:', flow4
        print '+++++++++++++++++++++++ Res_bc: flow5:', flow5
        print '+++++++++++++++++++++++ Res_bc: flow6:', flow6
        print '+++++++++++++++++++++++ Res_bc: flow7:', flow7
        print '+++++++++++++++++++++++ Res_bc: flow8:', flow8
        print '+++++++++++++++++++++++ Res_bc: flow9:', flow9
        print '+++++++++++++++++++++++ Res_bc: flow10:', flow10
        print '+++++++++++++++++++++++ Res_bc: flow11:', flow11
        print '+++++++++++++++++++++++ Res_bc: flow12:', flow12
        print '+++++++++++++++++++++++ Res_bc: flow13:', flow13
        print '+++++++++++++++++++++++ Res_bc: flow14:', flow14
        print '+++++++++++++++++++++++ Res_bc: flow15:', flow15
        print '+++++++++++++++++++++++ Res_bc: flow16:', flow16
        print '+++++++++++++++++++++++ Res_bc: flow17:', flow17
        print '+++++++++++++++++++++++ Res_bc: flow18:', flow18
        print '+++++++++++++++++++++++ Res_bc: flow19:', flow19
        print '+++++++++++++++++++++++ Res_bc: flow20:', flow20
        print '+++++++++++++++++++++++ Res_bc: flow21:', flow21
        print '+++++++++++++++++++++++ Res_bc: flow22:', flow22
        print '+++++++++++++++++++++++ Res_bc: flow23:', flow23
        print '+++++++++++++++++++++++ Res_bc: flow24:', flow24
        print '+++++++++++++++++++++++ Res_bc: flow25:', flow25
        print '+++++++++++++++++++++++ Res_bc: flow26:', flow26
        print '+++++++++++++++++++++++ Res_bc: flow27:', flow27
        print '+++++++++++++++++++++++ Res_bc: flow28:', flow28
        print '+++++++++++++++++++++++ Res_bc: flow29:', flow29
        print '+++++++++++++++++++++++ Res_bc: flow30:', flow30
        print '+++++++++++++++++++++++ Res_bc: flow31:', flow31
        print '+++++++++++++++++++++++ Res_bc: out1 to inlet ratio:', flow1/(inflow + 1e-16)
        print '+++++++++++++++++++++++ Res_bc: out2 to inlet ratio:', flow2/(inflow + 1e-16)
        print '+++++++++++++++++++++++ Res_bc: out3 to inlet ratio:', flow3/(inflow + 1e-16)
        print '+++++++++++++++++++++++ Res_bc: out4 to inlet ratio:', flow4/(inflow + 1e-16)
        print '+++++++++++++++++++++++ Res_bc: out5 to inlet ratio:', flow5/(inflow + 1e-16)
        print '+++++++++++++++++++++++ Res_bc: out6 to inlet ratio:', flow6/(inflow + 1e-16)
        print '+++++++++++++++++++++++ Res_bc: out7 to inlet ratio:', flow7/(inflow + 1e-16)
        print '+++++++++++++++++++++++ Res_bc: out8 to inlet ratio:', flow8/(inflow + 1e-16)
        print '+++++++++++++++++++++++ Res_bc: out9 to inlet ratio:', flow9/(inflow + 1e-16)
        print '+++++++++++++++++++++++ Res_bc: out10 to inlet ratio:', flow10/(inflow + 1e-16)
        print '+++++++++++++++++++++++ Res_bc: out11 to inlet ratio:', flow11/(inflow + 1e-16)
        print '+++++++++++++++++++++++ Res_bc: out12 to inlet ratio:', flow12/(inflow + 1e-16)
        print '+++++++++++++++++++++++ Res_bc: out13 to inlet ratio:', flow13/(inflow + 1e-16)
        print '+++++++++++++++++++++++ Res_bc: out14 to inlet ratio:', flow14/(inflow + 1e-16)
        print '+++++++++++++++++++++++ Res_bc: out15 to inlet ratio:', flow15/(inflow + 1e-16)
        print '+++++++++++++++++++++++ Res_bc: out16 to inlet ratio:', flow16/(inflow + 1e-16)
        print '+++++++++++++++++++++++ Res_bc: out17 to inlet ratio:', flow17/(inflow + 1e-16)
        print '+++++++++++++++++++++++ Res_bc: out18 to inlet ratio:', flow18/(inflow + 1e-16)
        print '+++++++++++++++++++++++ Res_bc: out19 to inlet ratio:', flow19/(inflow + 1e-16)
        print '+++++++++++++++++++++++ Res_bc: out20 to inlet ratio:', flow20/(inflow + 1e-16)
        print '+++++++++++++++++++++++ Res_bc: out21 to inlet ratio:', flow21/(inflow + 1e-16)
        print '+++++++++++++++++++++++ Res_bc: out22 to inlet ratio:', flow22/(inflow + 1e-16)
        print '+++++++++++++++++++++++ Res_bc: out23 to inlet ratio:', flow23/(inflow + 1e-16)
        print '+++++++++++++++++++++++ Res_bc: out24 to inlet ratio:', flow24/(inflow + 1e-16)
        print '+++++++++++++++++++++++ Res_bc: out25 to inlet ratio:', flow25/(inflow + 1e-16)
        print '+++++++++++++++++++++++ Res_bc: out26 to inlet ratio:', flow26/(inflow + 1e-16)
        print '+++++++++++++++++++++++ Res_bc: out27 to inlet ratio:', flow27/(inflow + 1e-16)
        print '+++++++++++++++++++++++ Res_bc: out28 to inlet ratio:', flow28/(inflow + 1e-16)
        print '+++++++++++++++++++++++ Res_bc: out29 to inlet ratio:', flow29/(inflow + 1e-16)
        print '+++++++++++++++++++++++ Res_bc: out30 to inlet ratio:', flow30/(inflow + 1e-16)
        print '+++++++++++++++++++++++ Res_bc: out31 to inlet ratio:', flow31/(inflow + 1e-16)
        print '+++++++++++++++++++++++ Res_bc: total outflow:', flow1 + flow2 + flow3 + flow4 + flow5 + flow6 + flow7 + flow8 + flow9 + flow10 + flow11 + flow12 + flow13 + flow14 + flow15 + flow16 + flow17 + flow18 + flow19 + flow20 + flow21 + flow22 + flow23 + flow24 + flow25 + flow26 + flow27 + flow28 + flow29 + flow30 + flow31
        print '+++++++++++++++++++++++ inlet_bc: inflow:', inflow
        print '+++++++++++++++++++++++ Res1 = ', Res1, 'Res2 = ', Res2, 'Res3 =', Res3, 'Res4 =', Res4
        print '+++++++++++++++++++++++ Res5 = ', Res5, 'Res6 = ', Res6, 'Res7 =', Res7, 'Res8 =', Res8
        print '+++++++++++++++++++++++ Res9 = ', Res9, 'Res10 = ', Res10, 'Res11 =', Res11, 'Res12 =', Res12
        print '+++++++++++++++++++++++ Res13 = ', Res13, 'Res14 = ', Res14, 'Res15 =', Res15, 'Res16 =', Res16
        print '+++++++++++++++++++++++ Res17 = ', Res17, 'Res18 = ', Res18, 'Res19 =', Res19, 'Res20 =', Res20
        print '+++++++++++++++++++++++ Res21 = ', Res21, 'Res22 = ', Res22, 'Res23 =', Res23, 'Res24 =', Res24
        print '+++++++++++++++++++++++ Res25 = ', Res25, 'Res26 = ', Res26, 'Res27 =', Res27, 'Res28 =', Res28
        print '+++++++++++++++++++++++ Res29 = ', Res29, 'Res30 = ', Res30, 'Res31 =', Res31

     bcs['p'] = [DirichletBC(Q,  Res_bc1 , facet_domains,3), DirichletBC(Q,  Res_bc2, facet_domains,4), DirichletBC(Q,  Res_bc3, facet_domains,5), DirichletBC(Q,  Res_bc4, facet_domains,6), DirichletBC(Q,  Res_bc5, facet_domains,7), DirichletBC(Q,  Res_bc6, facet_domains,8), DirichletBC(Q,  Res_bc7, facet_domains,9), DirichletBC(Q,  Res_bc8, facet_domains,10), DirichletBC(Q,  Res_bc9, facet_domains,11),\
                 DirichletBC(Q,  Res_bc10 , facet_domains,12), DirichletBC(Q,  Res_bc11, facet_domains,13), DirichletBC(Q,  Res_bc12, facet_domains,14), DirichletBC(Q,  Res_bc13, facet_domains,15), DirichletBC(Q,  Res_bc14, facet_domains,16), DirichletBC(Q,  Res_bc15, facet_domains,17), DirichletBC(Q,  Res_bc16, facet_domains,18), DirichletBC(Q,  Res_bc17, facet_domains,19), DirichletBC(Q,  Res_bc18, facet_domains,20), \
                 DirichletBC(Q,  Res_bc19 , facet_domains,21), DirichletBC(Q,  Res_bc20, facet_domains,22), DirichletBC(Q,  Res_bc21, facet_domains,23), DirichletBC(Q,  Res_bc22, facet_domains,24), DirichletBC(Q,  Res_bc23, facet_domains,25), DirichletBC(Q,  Res_bc24, facet_domains,26), DirichletBC(Q,  Res_bc25, facet_domains,27), DirichletBC(Q,  Res_bc26, facet_domains,28), DirichletBC(Q,  Res_bc27, facet_domains,29), \
                 DirichletBC(Q,  Res_bc28 , facet_domains,30), DirichletBC(Q,  Res_bc29, facet_domains,31), DirichletBC(Q,  Res_bc30, facet_domains,32), DirichletBC(Q,  Res_bc31, facet_domains,33)]
       #Ap = assemble_matrix(Ap2, bcs['p'])
    [bc.apply(b['p']) for bc in bcs['p']]
    dp_.vector().zero()
    dp_.vector().axpy(1., x_['p'])
    # KrylovSolvers use nullspace for normalization of pressure
    if hasattr(Ap, 'null_space'):
        p_sol.null_space.orthogonalize(b['p'])


    t1 = Timer("Pressure Linear Algebra Solve")
    p_sol.solve(Ap, x_['p'], b['p'])

    t1.stop()
    # LUSolver use normalize directly for normalization of pressure
    if hasattr(p_sol, 'normalize'):
        normalize(x_['p'])

    dp_.vector().axpy(-1., x_['p'])
    dp_.vector()._scale(-1.)


def velocity_update(u_components, bcs, gradp, dp_, dt, x_, **NS_namespace):
    """Update the velocity after regular pressure velocity iterations."""
    for ui in u_components:
        gradp[ui](dp_)
        x_[ui].axpy(-dt, gradp[ui].vector())
        [bc.apply(x_[ui]) for bc in bcs[ui]]

def scalar_assemble(a_scalar, a_conv, Ta, dt, M, scalar_components, Schmidt_T, KT,
                    nu, nut_, Schmidt, b, K, x_1, b0, les_model, **NS_namespace):
    """Assemble scalar equation."""
    # Just in case you want to use a different scalar convection
    if not a_scalar is a_conv:
        assemble(a_scalar, tensor=Ta)
        Ta._scale(-0.5)            # Negative convection on the rhs
        Ta.axpy(1. / dt, M, True)    # Add mass

    # Compute rhs for all scalars
    for ci in scalar_components:
        # Add diffusion
        Ta.axpy(-0.5 * nu / Schmidt[ci], K, True)
        if not les_model is "NoModel":
            Ta.axpy(-0.5 / Schmidt_T[ci], KT[0], True)

        # Compute rhs
        b[ci].zero()
        b[ci].axpy(1., Ta * x_1[ci])
        b[ci].axpy(1., b0[ci])

        # Subtract diffusion
        Ta.axpy(0.5 * nu / Schmidt[ci], K, True)
        if not les_model is "NoModel":
            Ta.axpy(0.5 / Schmidt_T[ci], KT[0], True)

    # Reset matrix for lhs - Note scalar matrix does not contain diffusion
    Ta._scale(-1.)
    Ta.axpy(2. / dt, M, True)


def scalar_solve(ci, scalar_components, Ta, b, x_, bcs, c_sol,
                 nu, Schmidt, K, **NS_namespace):
    """Solve scalar equation."""

    Ta.axpy(0.5 * nu / Schmidt[ci], K, True)  # Add diffusion
    if len(scalar_components) > 1:
        # Reuse solver for all scalars. This requires the same matrix and vectors to be used by c_sol.
        Tb, bb, bx = NS_namespace['Tb'], NS_namespace['bb'], NS_namespace['bx']
        Tb.zero()
        Tb.axpy(1., Ta, True)
        bb.zero()
        bb.axpy(1., b[ci])
        bx.zero()
        bx.axpy(1., x_[ci])
        [bc.apply(Tb, bb) for bc in bcs[ci]]
        c_sol.solve(Tb, bx, bb)
        x_[ci].zero()
        x_[ci].axpy(1., bx)

    else:
        [bc.apply(Ta, b[ci]) for bc in bcs[ci]]
        c_sol.solve(Ta, x_[ci], b[ci])
    Ta.axpy(-0.5 * nu / Schmidt[ci], K, True)  # Subtract diffusion
    # x_[ci][x_[ci] < 0] = 0.               # Bounded solution
    #x_[ci].set_local(maximum(0., x_[ci].array()))
    # x_[ci].apply("insert")


def IC_Laplacian(V, V_ave, q_, q_1, q_2, sys_comp, u_components, facet_domains, inlet_flag_flow, mesh,
                  Res1, Res2, Res3, Res4, Res5, Res6, Res7, Res8, Res9, Res10, Res11, Res12, Res13, Res14,
                  Res15, Res16, Res17, Res18, Res19, Res20, Res21, Res22, Res23, Res24, Res25, Res26, Res27,
                  Res28, Res29, Res30, Res31, Resistance_flag, **NS_namespace):

  # ******************************************************************************
  # ************************** Defining Function Spaces **************************
  # ******************************************************************************

  Q = FunctionSpace(mesh, 'CG', 1)
  Vv = VectorFunctionSpace(V.mesh(), V.ufl_element().family(),
                                 V.ufl_element().degree())

  my_phi = TrialFunction(Q)
  vel_IC_ = Function(V, name="IC_vel")
  phi_IC = Function(Q, name="phi_vel")
  phi_p = Function(Q, name="phi_p")
  my_w = TestFunction(Q)

  # ******************************************************************************
  # ***** Calculating the Average Velocity Based on the Specified Flow Rate ******
  # ******************************************************************************
  inlet_facet_ID = 1
  ds = Measure("ds")(subdomain_data=facet_domains)
  if (inlet_flag_flow):
    dsi = ds(inlet_facet_ID, domain=mesh, subdomain_data=facet_domains)
    # Compute area of boundary tesselation by integrating 1.0 over all facets
    A = assemble(Constant(1.0, name="one")*dsi)
    V_ave = -V_ave*1e3 / A

  # ******************************************************************************
  # ********************************** Velocity **********************************
  # ******************************************************************************

  # *********************** Specifying Boundary Conditions ***********************
  # The Laplace equation is : div(grad(phi)) = 0
  # The velocity can be calculated as : grad(phi) = Velocity
  # Inlet boundary condition: grad(phi)*n = V_ave
  # Wall boundary conditions: grad(phi)*n = 0
  # Outlet boundary conditions: phi = 0

  my_out1 = DirichletBC(Q, 0.0, facet_domains, 3) # Inlet: 0 drichlet
  my_out2 = DirichletBC(Q, 0.0, facet_domains, 4) # Inlet: 0 drichlet
  my_out3 = DirichletBC(Q, 0.0, facet_domains, 5) # Inlet: 0 drichlet
  my_out4 = DirichletBC(Q, 0.0, facet_domains, 6) # Inlet: 0 drichlet
  my_out5 = DirichletBC(Q, 0.0, facet_domains, 7) # Inlet: 0 drichlet
  my_out6 = DirichletBC(Q, 0.0, facet_domains, 8) # Inlet: 0 drichlet
  my_out7 = DirichletBC(Q, 0.0, facet_domains, 9) # Inlet: 0 drichlet
  my_out8 = DirichletBC(Q, 0.0, facet_domains, 10) # Inlet: 0 drichlet
  my_out9 = DirichletBC(Q, 0.0, facet_domains, 11) # Inlet: 0 drichlet
  my_out10 = DirichletBC(Q, 0.0, facet_domains, 12) # Inlet: 0 drichlet
  my_out11 = DirichletBC(Q, 0.0, facet_domains, 13) # Inlet: 0 drichlet
  my_out12 = DirichletBC(Q, 0.0, facet_domains, 14) # Inlet: 0 drichlet
  my_out13 = DirichletBC(Q, 0.0, facet_domains, 15) # Inlet: 0 drichlet
  my_out14 = DirichletBC(Q, 0.0, facet_domains, 16) # Inlet: 0 drichlet
  my_out15 = DirichletBC(Q, 0.0, facet_domains, 17) # Inlet: 0 drichlet
  my_out16 = DirichletBC(Q, 0.0, facet_domains, 18) # Inlet: 0 drichlet
  my_out17 = DirichletBC(Q, 0.0, facet_domains, 19) # Inlet: 0 drichlet
  my_out18 = DirichletBC(Q, 0.0, facet_domains, 20) # Inlet: 0 drichlet
  my_out19 = DirichletBC(Q, 0.0, facet_domains, 21) # Inlet: 0 drichlet
  my_out20 = DirichletBC(Q, 0.0, facet_domains, 22) # Inlet: 0 drichlet
  my_out21 = DirichletBC(Q, 0.0, facet_domains, 23) # Inlet: 0 drichlet
  my_out22 = DirichletBC(Q, 0.0, facet_domains, 24) # Inlet: 0 drichlet
  my_out23 = DirichletBC(Q, 0.0, facet_domains, 25) # Inlet: 0 drichlet
  my_out24 = DirichletBC(Q, 0.0, facet_domains, 26) # Inlet: 0 drichlet
  my_out25 = DirichletBC(Q, 0.0, facet_domains, 27) # Inlet: 0 drichlet
  my_out26 = DirichletBC(Q, 0.0, facet_domains, 28) # Inlet: 0 drichlet
  my_out27 = DirichletBC(Q, 0.0, facet_domains, 29) # Inlet: 0 drichlet
  my_out28 = DirichletBC(Q, 0.0, facet_domains, 30) # Inlet: 0 drichlet
  my_out29 = DirichletBC(Q, 0.0, facet_domains, 31) # Inlet: 0 drichlet
  my_out30 = DirichletBC(Q, 0.0, facet_domains, 32) # Inlet: 0 drichlet
  my_out31 = DirichletBC(Q, 0.0, facet_domains, 33) # Inlet: 0 drichlet

  my_out_bcs = [my_out1, my_out2, my_out3, my_out4, my_out5, my_out6, my_out7, my_out8, my_out9, my_out10,\
                my_out11, my_out12, my_out13, my_out14, my_out15, my_out16, my_out17, my_out18, my_out19,\
                my_out20, my_out21, my_out22, my_out23, my_out24, my_out25, my_out26, my_out27, my_out28,\
                my_out29, my_out30, my_out31]

  # ************************* Laplacian Eq for Velocity ****************************

  if MPI.rank(mpi_comm_world()) == 0:
    print '\033[1;31;40m' + 'Laplacian initialization is in process ...' + '\033[0m'

  lap_IC_weakform = dot(grad(my_phi), grad(my_w)) * dx - V_ave * my_w * ds(1)

  my_a, my_L = lhs(lap_IC_weakform), rhs(lap_IC_weakform)

  # ************************ Solving the Variational Form **************************

  if MPI.rank(mpi_comm_world()) == 0:
    print '\033[1;31;40m' + 'Solving for Velocities ...' + '\033[0m'

  solve(my_a == my_L, phi_IC, my_out_bcs)

  if MPI.rank(mpi_comm_world()) == 0:
    print '\033[1;31;40m' + 'Starting Initialization for Velocity Field ...' + '\033[0m'

  vel_IC_ = project(grad(phi_IC), Vv, solver_type='cg', preconditioner_type='hypre_amg')
  u0x = project(vel_IC_[0], V, solver_type='cg', preconditioner_type='hypre_amg')
  u1x = project(vel_IC_[1], V, solver_type='cg', preconditioner_type='hypre_amg')
  u2x = project(vel_IC_[2], V, solver_type='cg', preconditioner_type='hypre_amg')

  # ******************** Applying Initialization for Velocity Field at Time-step t

  if MPI.rank(mpi_comm_world()) == 0:
    print '\033[1;31;40m' + 'Applying Initialization for Velocity Field at Time-step t ...' + '\033[0m'

  q_['u0'].vector()[:] = u0x.vector()[:]
  q_['u1'].vector()[:] = u1x.vector()[:]
  q_['u2'].vector()[:] = u2x.vector()[:]

  # ******************** Applying Initialization for Velocity Field at Time-step t - dt

  if MPI.rank(mpi_comm_world()) == 0:
    print '\033[1;31;40m' + 'Applying Initialization for Velocity Field at Time-step t - dt ...' + '\033[0m'

  q_1['u0'].vector()[:] = q_['u0'].vector()[:]
  q_1['u1'].vector()[:] = q_['u1'].vector()[:]
  q_1['u2'].vector()[:] = q_['u2'].vector()[:]

  if MPI.rank(mpi_comm_world()) == 0:
    print '\033[1;31;40m' + 'Applying Initialization for Velocity Field at Time-step t - 2dt ...' + '\033[0m'

  # ******************** Applying Initialization for Velocity Field at Time-step t - 2dt

  q_2['u0'].vector()[:] = q_1['u0'].vector()[:]
  q_2['u1'].vector()[:] = q_1['u1'].vector()[:]
  q_2['u2'].vector()[:] = q_1['u2'].vector()[:]


  # ******************************************************************************
  # ********************************** Pressure **********************************
  # ******************************************************************************

  # *********************** Specifying Boundary Conditions ***********************
  # The Laplace equation is : div(grad(phi)) = 0
  # The pressure can be calculated as : phi = Pressure
  # Inlet boundary condition: Dirichlet
  # Wall boundary conditions: grad(phi)*n = 0
  # Outlet boundary conditions: grad(phi)*n = 0

  p_in = -115 * 133.322 * 1e-1   # inlet pressure; units: g/cm.s2 = 1e-1 Pa = 1 mmHg * 133.322 Pa/mmHg * 1e-1 (g/cm.s2)/Pa

  # **************************** Specifying BC for inlet ***************************

  my_in_p = DirichletBC(Q, p_in, facet_domains, inlet_facet_ID) # Inlet: dirichlet

  # *************************** Specifying BC for outlet ***************************

  if (Resistance_flag):

    # ************* Create vector of the segregated velocity components ************

    u_dic  = as_vector([q_ [ui] for ui in u_components]) # Velocity vector at t

    # ******************** Calculating the flowrates at outlets ********************
    n_normal= FacetNormal(mesh)

    flow1 = abs( assemble(dot(u_dic, n_normal)*ds(3, domain=mesh, subdomain_data=facet_domains) ) )
    flow2 = abs( assemble(dot(u_dic, n_normal)*ds(4, domain=mesh, subdomain_data=facet_domains) ) )
    flow3 = abs( assemble(dot(u_dic, n_normal)*ds(5, domain=mesh, subdomain_data=facet_domains) ) )
    flow4 = abs( assemble(dot(u_dic, n_normal)*ds(6, domain=mesh, subdomain_data=facet_domains) ) )
    flow5 = abs( assemble(dot(u_dic, n_normal)*ds(7, domain=mesh, subdomain_data=facet_domains) ) )
    flow6 = abs( assemble(dot(u_dic, n_normal)*ds(8, domain=mesh, subdomain_data=facet_domains) ) )
    flow7 = abs( assemble(dot(u_dic, n_normal)*ds(9, domain=mesh, subdomain_data=facet_domains) ) )
    flow8 = abs( assemble(dot(u_dic, n_normal)*ds(10, domain=mesh, subdomain_data=facet_domains) ) )
    flow9 = abs( assemble(dot(u_dic, n_normal)*ds(11, domain=mesh, subdomain_data=facet_domains) ) )
    flow10 = abs( assemble(dot(u_dic, n_normal)*ds(12, domain=mesh, subdomain_data=facet_domains) ) )
    flow11 = abs( assemble(dot(u_dic, n_normal)*ds(13, domain=mesh, subdomain_data=facet_domains) ) )
    flow12 = abs( assemble(dot(u_dic, n_normal)*ds(14, domain=mesh, subdomain_data=facet_domains) ) )
    flow13 = abs( assemble(dot(u_dic, n_normal)*ds(15, domain=mesh, subdomain_data=facet_domains) ) )
    flow14 = abs( assemble(dot(u_dic, n_normal)*ds(16, domain=mesh, subdomain_data=facet_domains) ) )
    flow15 = abs( assemble(dot(u_dic, n_normal)*ds(17, domain=mesh, subdomain_data=facet_domains) ) )
    flow16 = abs( assemble(dot(u_dic, n_normal)*ds(18, domain=mesh, subdomain_data=facet_domains) ) )
    flow17 = abs( assemble(dot(u_dic, n_normal)*ds(19, domain=mesh, subdomain_data=facet_domains) ) )
    flow18 = abs( assemble(dot(u_dic, n_normal)*ds(20, domain=mesh, subdomain_data=facet_domains) ) )
    flow19 = abs( assemble(dot(u_dic, n_normal)*ds(21, domain=mesh, subdomain_data=facet_domains) ) )
    flow20 = abs( assemble(dot(u_dic, n_normal)*ds(22, domain=mesh, subdomain_data=facet_domains) ) )
    flow21 = abs( assemble(dot(u_dic, n_normal)*ds(23, domain=mesh, subdomain_data=facet_domains) ) )
    flow22 = abs( assemble(dot(u_dic, n_normal)*ds(24, domain=mesh, subdomain_data=facet_domains) ) )
    flow23 = abs( assemble(dot(u_dic, n_normal)*ds(25, domain=mesh, subdomain_data=facet_domains) ) )
    flow24 = abs( assemble(dot(u_dic, n_normal)*ds(26, domain=mesh, subdomain_data=facet_domains) ) )
    flow25 = abs( assemble(dot(u_dic, n_normal)*ds(27, domain=mesh, subdomain_data=facet_domains) ) )
    flow26 = abs( assemble(dot(u_dic, n_normal)*ds(28, domain=mesh, subdomain_data=facet_domains) ) )
    flow27 = abs( assemble(dot(u_dic, n_normal)*ds(29, domain=mesh, subdomain_data=facet_domains) ) )
    flow28 = abs( assemble(dot(u_dic, n_normal)*ds(30, domain=mesh, subdomain_data=facet_domains) ) )
    flow29 = abs( assemble(dot(u_dic, n_normal)*ds(31, domain=mesh, subdomain_data=facet_domains) ) )
    flow30 = abs( assemble(dot(u_dic, n_normal)*ds(32, domain=mesh, subdomain_data=facet_domains) ) )
    flow31 = abs( assemble(dot(u_dic, n_normal)*ds(33, domain=mesh, subdomain_data=facet_domains) ) )

    # ******************** Calculating the pressure at outlets *********************

    Res_bc1 =  -Res1 * flow1
    Res_bc2 =  -Res2 * flow2
    Res_bc3 =  -Res3 * flow3
    Res_bc4 =  -Res4 * flow4
    Res_bc5 =  -Res5 * flow5
    Res_bc6 =  -Res6 * flow6
    Res_bc7 =  -Res7 * flow7
    Res_bc8 =  -Res8 * flow8
    Res_bc9 =  -Res9 * flow9
    Res_bc10 =  -Res10 * flow10
    Res_bc11 =  -Res11 * flow11
    Res_bc12 =  -Res12 * flow12
    Res_bc13 =  -Res13 * flow13
    Res_bc14 =  -Res14 * flow14
    Res_bc15 =  -Res15 * flow15
    Res_bc16 =  -Res16 * flow16
    Res_bc17 =  -Res17 * flow17
    Res_bc18 =  -Res18 * flow18
    Res_bc19 =  -Res19 * flow19
    Res_bc20 =  -Res20 * flow20
    Res_bc21 =  -Res21 * flow21
    Res_bc22 =  -Res22 * flow22
    Res_bc23 =  -Res23 * flow23
    Res_bc24 =  -Res24 * flow24
    Res_bc25 =  -Res25 * flow25
    Res_bc26 =  -Res26 * flow26
    Res_bc27 =  -Res27 * flow27
    Res_bc28 =  -Res28 * flow28
    Res_bc29 =  -Res29 * flow29
    Res_bc30 =  -Res30 * flow30
    Res_bc31 =  -Res31 * flow31

    # *********************** Specifying outlet pressure BC ************************

    my_bcs_p = [DirichletBC(Q,  Res_bc1, facet_domains,3), DirichletBC(Q,  Res_bc2, facet_domains,4), DirichletBC(Q,  Res_bc3, facet_domains,5),\
                    DirichletBC(Q,  Res_bc4, facet_domains,6), DirichletBC(Q,  Res_bc5, facet_domains,7), DirichletBC(Q,  Res_bc6, facet_domains,8),\
                    DirichletBC(Q,  Res_bc7, facet_domains,9), DirichletBC(Q,  Res_bc8, facet_domains,10), DirichletBC(Q,  Res_bc9, facet_domains,11),\
                    DirichletBC(Q,  Res_bc10, facet_domains,12), DirichletBC(Q,  Res_bc11, facet_domains,13), DirichletBC(Q,  Res_bc12, facet_domains,14),\
                    DirichletBC(Q,  Res_bc13, facet_domains,15), DirichletBC(Q,  Res_bc14, facet_domains,16), DirichletBC(Q,  Res_bc15, facet_domains,17),\
                    DirichletBC(Q,  Res_bc16, facet_domains,18), DirichletBC(Q,  Res_bc17, facet_domains,19), DirichletBC(Q,  Res_bc18, facet_domains,20),\
                    DirichletBC(Q,  Res_bc19, facet_domains,21), DirichletBC(Q,  Res_bc20, facet_domains,22), DirichletBC(Q,  Res_bc21, facet_domains,23),\
                    DirichletBC(Q,  Res_bc22, facet_domains,24), DirichletBC(Q,  Res_bc23, facet_domains,25), DirichletBC(Q,  Res_bc24, facet_domains,26),\
                    DirichletBC(Q,  Res_bc25, facet_domains,27), DirichletBC(Q,  Res_bc26, facet_domains,28), DirichletBC(Q,  Res_bc27, facet_domains,29),\
                    DirichletBC(Q,  Res_bc28 , facet_domains,30), DirichletBC(Q,  Res_bc29, facet_domains,31), DirichletBC(Q,  Res_bc30, facet_domains,32),\
                    DirichletBC(Q,  Res_bc31, facet_domains,33), my_in_p]

  else:
    my_bcs_p = [my_in_p]
  

  # ************************ Laplacian Eq for Pressure *****************************

  lap_P_weakform = dot(grad(my_phi), grad(my_w)) * dx

  my_a_p, my_L_p = lhs(lap_P_weakform), rhs(lap_P_weakform)

  # ************************ Solving the Variational Form **************************

  if MPI.rank(mpi_comm_world()) == 0:
    print '\033[1;31;40m' + 'Solving for Pressure ...' + '\033[0m'

  solve(my_a_p == my_L_p, phi_p, my_bcs_p)

  # ********************* Updating the Solution Dictionaries ***********************

  if MPI.rank(mpi_comm_world()) == 0:
    print '\033[1;31;40m' + 'Starting Initialization for Pressure Field ...' + '\033[0m'

  p_all = project(phi_p, V, solver_type='cg', preconditioner_type='hypre_amg')

  # ******************** Applying Initialization for Pressure Field at Time-step t

  if MPI.rank(mpi_comm_world()) == 0:
    print '\033[1;31;40m' + 'Applying Initialization for Pressure Field at Time-step t ...' + '\033[0m'
  
  q_['p'].vector()[:] = p_all.vector()[:]

  if MPI.rank(mpi_comm_world()) == 0:
    print '\033[1;31;40m' + 'Initialization: Done!' + '\033[0m'







