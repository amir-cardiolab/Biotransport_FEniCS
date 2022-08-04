#!/usr/bin/env python

__author__ = "Mikael Mortensen <mikaem@math.uio.no>"
__date__ = "2013-11-06"
__copyright__ = "Copyright (C) 2013 " + __author__
__license__ = "GNU Lesser GPL version 3 or any later version"

"""
This module implements a generic form of the fractional step method for
solving the incompressible Navier-Stokes equations. There are several
possible implementations of the pressure correction and the more low-level
details are chosen at run-time and imported from any one of:

  solvers/NSfracStep/IPCS_ABCN.py    # Implicit convection
  solvers/NSfracStep/IPCS_ABE.py     # Explicit convectionesh
  solvers/NSfracStep/IPCS.py         # Naive implict convection
  solvers/NSfracStep/BDFPC.py        # Naive Backwards Differencing IPCS in rotational form
  solvers/NSfracStep/BDFPC_Fast.py   # Fast Backwards Differencing IPCS in rotational form
  solvers/NSfracStep/Chorin.py       # Naive

The naive solvers are very simple and not optimized. They are intended
for validation of the other optimized versions. The fractional step method
can be used both non-iteratively or with iterations over the pressure-
velocity system.

The velocity vector is segregated, and we use three (in 3D) scalar
velocity components.

Each new problem needs to implement a new problem module to be placed in
the problems/NSfracStep folder. From the problems module one needs to import
a mesh and a control dictionary called NS_parameters. See
problems/NSfracStep/__init__.py for all possible parameters.

"""

#!!!!!!!!!Amir: changed imports in some of the files similar to the old 2016.2 oasis version to get rid of the complex path imports

import sys, os #added by Amir

#sys.path.append(os.getcwd()) #added by Amir
import importlib
#from oasis import * #added by amir
#from oasis.common import *
from common import * #change by Amir

#from problems import * #added by Amir
#from problems.NSfracStep import * #added by Amir



commandline_kwargs = parse_command_line()

default_problem = 'DrivenCavity'

if(0):
 problemname = commandline_kwargs.get('problem', default_problem)
 try:
    #problemmod = importlib.import_module('.'.join(('oasis.problems.NSfracStep', problemname)))
    problemmod = importlib.import_module('.' + problemname,package='problems.NSfracStep') #changed by Amir
 except ImportError:
    problemmod = importlib.import_module(problemname)
 #except: #commented by Amir
 #    raise RuntimeError(problemname+' not found')

 vars().update(**vars(problemmod))
 # Update problem spesific parameters
 problem_parameters(**vars())
 # Update current namespace with NS_parameters and commandline_kwargs ++
 vars().update(post_import_problem(**vars()))
 # Import chosen functionality from solvers
 solver = importlib.import_module('.'.join(('oasis.solvers.NSfracStep', solver)))
 vars().update({name:solver.__dict__[name] for name in solver.__all__})

if(1):
 exec("from problems.NSfracStep.{} import *".format(commandline_kwargs.get('problem', default_problem)))
 # Update current namespace with NS_parameters and commandline_kwargs ++
 vars().update(post_import_problem(**vars()))
 # Import chosen functionality from solvers
 exec("from solvers.NSfracStep.{} import *".format(solver))

# Create lists of components solved for
dim = mesh.geometry().dim()
u_components = ['u' + str(x) for x in range(dim)]
sys_comp = u_components + ['p'] + scalar_components
uc_comp = u_components + scalar_components

# Set up initial folders for storing results
newfolder, tstepfiles,tstepfiles_wss = create_initial_folders(**vars())

# Declare FunctionSpaces and arguments
V = Q = FunctionSpace(mesh, 'CG', velocity_degree,
                      constrained_domain=constrained_domain)
if velocity_degree != pressure_degree:
    Q = FunctionSpace(mesh, 'CG', pressure_degree,
                      constrained_domain=constrained_domain)

u = TrialFunction(V)
v = TestFunction(V)
p = TrialFunction(Q)
q = TestFunction(Q)

bc_file =  '/scratch/mm4238/transport/porcines/4609_B/mesh/BCnodeFacets.xml'
facet_domains = MeshFunction('size_t', mesh,bc_file )

# Reading the hyperbolic profile for inlet velocity BC; Added by Mostafa
#V_inletBC_x = Function(V,'/Users/mm4238/Data/CBLab/Biochem_Tr_Budof/Simulations/cylinder_test/FinalMeshFiles/bctxml_x.xml')
#V_inletBC_y = Function(V,'/Users/mm4238/Data/CBLab/Biochem_Tr_Budof/Simulations/cylinder_test/FinalMeshFiles/bctxml_y.xml')
#V_inletBC_z = Function(V,'/Users/mm4238/Data/CBLab/Biochem_Tr_Budof/Simulations/cylinder_test/FinalMeshFiles/bctxml_z.xml')

inlet_facet_ID = 1
if(inlet_rotation):
    # Creating hyperbolic profile for inlet velocity BC assuming a circular inlet surface
    d = mesh.geometry().dim()
    x = SpatialCoordinate(mesh)
    ds = Measure("ds")(subdomain_data=facet_domains)
    dsi = ds(inlet_facet_ID, domain=mesh, subdomain_data=facet_domains)
    # Compute area of boundary tesselation by integrating 1.0 over all facets
    A = assemble(Constant(1.0, name="one")*dsi)
    if MPI.rank(mpi_comm_world()) == 0:
            print 'Inlet area:', A
    #if (inlet_flag_flow):
    #    v_inlet_BC = v_inlet_BC / A
    inlet_radius = numpy.sqrt(A / pi) # This old estimate is a few % lower because of boundary discretization errors
    inlet_radius2 = A / pi #It should be r**2 in the parabolic eqn below!
    if MPI.rank(mpi_comm_world()) == 0:
        print 'Inlet radius:', inlet_radius
    # Compute barycenter by integrating x components over all facets
    center = [assemble(x[i]*dsi) / A for i in xrange(d)]
    if MPI.rank(mpi_comm_world()) == 0:
            print 'Inlet center:', center
    # Creating a parabolic velocity profile in z-direction
    v_inlet_BC = 1.0
    vel_z = Expression("u * (1 - (pow(cent0-x[0], 2) + pow(cent1-x[1], 2) ) / r)", r=inlet_radius2, u=-2.0*v_inlet_BC, cent0=center[0],cent1=center[1],cent2=center[2], degree=3)
    # Compute average normal (assuming boundary is actually flat)
    n_normal= FacetNormal(mesh)
    ni = numpy.array([assemble(n_normal[i]*dsi) for i in xrange(d)])
    n_len = numpy.sqrt(sum([ni[i]**2 for i in xrange(d)])) # Should always be 1!?
    normal = ni/n_len
    # Angle between the average normal vector z-axis
    arg_acos = numpy.dot(normal,[0,0,1])
    theta_ang = acos(arg_acos)*pi/180
    # Finding the rotation axis perpendicular to average normal vector
    vec_prep = numpy.cross(normal,[0,0,1])
    #vec_prep = vec_prep/norm(vec_prep)
    # Rotating the parabolic velocity profile to the direction of average normal vector
    R_matrix = [[cos(theta_ang) + pow(vec_prep[0],2)*(1 - cos(theta_ang)), vec_prep[0]*vec_prep[1]*(1 - cos(theta_ang)) - vec_prep[2]*sin(theta_ang), vec_prep[0]*vec_prep[2]*(1 - cos(theta_ang)) + vec_prep[1]*sin(theta_ang)],
            [vec_prep[1]*vec_prep[0]*(1 - cos(theta_ang)) + vec_prep[2]*sin(theta_ang), cos(theta_ang) + pow(vec_prep[1],2)*(1 - cos(theta_ang)), vec_prep[1]*vec_prep[2]*(1 - cos(theta_ang)) - vec_prep[0]*sin(theta_ang)],
            [vec_prep[2]*vec_prep[0]*(1 - cos(theta_ang)) - vec_prep[1]*sin(theta_ang), vec_prep[2]*vec_prep[1]*(1 - cos(theta_ang)) + vec_prep[0]*sin(theta_ang), cos(theta_ang) + pow(vec_prep[2],2)*(1 - cos(theta_ang))] ]
    #inlet_vel = R_matrix.dot([0,0,vel_z])
    bc_in_1  = DirichletBC(V, vel_z , facet_domains,1)
    print bc_in_1

#if(geometry_rotation):
if(1):
#Rotating the mesh so that the average normal vector aligned with z-axis passing the centroid
      d = mesh.geometry().dim()
      x = SpatialCoordinate(mesh)
      ds = Measure("ds")(subdomain_data=facet_domains)
      dsi = ds(inlet_facet_ID, domain=mesh, subdomain_data=facet_domains)
      # Compute area of boundary tesselation by integrating 1.0 over all facets
      A = assemble(Constant(1.0, name="one")*dsi)
      if MPI.rank(mpi_comm_world()) == 0:
              print 'Inlet area:', A
      inlet_radius = numpy.sqrt(A / pi) # This old estimate is a few % lower because of boundary discretization errors
      inlet_radius2 = A / pi #It should be r**2 in the parabolic eqn below!
      if MPI.rank(mpi_comm_world()) == 0:
          print 'Inlet radius:', inlet_radius
      # Compute barycenter by integrating x components over all facets
      center = [assemble(x[i]*dsi) / A for i in xrange(d)]
      if MPI.rank(mpi_comm_world()) == 0:
              print 'Inlet center:', center
      center = Point(numpy.array(center))
      # Compute average normal (assuming boundary is actually flat)
      n_normal= FacetNormal(mesh)
      ni = numpy.array([assemble(n_normal[i]*dsi) for i in xrange(d)])
      n_len = numpy.sqrt(sum([ni[i]**2 for i in xrange(d)])) # Should always be 1!?
      normal = ni/n_len
      if MPI.rank(mpi_comm_world()) == 0:
              print 'normal vector: ', normal
      alpha_ang = (normal[0]/abs(normal[0]))*acos(normal[0])*180./pi
      beta_ang = (normal[1]/abs(normal[1]))*acos(normal[1])*180./pi
      gamma_ang = -(normal[2]/abs(normal[2]))*acos(normal[2])*180./pi
      #MeshTransformation.rotate(mesh, gamma_ang, 0, center)
      mesh.rotate(gamma_ang, 0, center)

      d = mesh.geometry().dim()
      x = SpatialCoordinate(mesh)
      ds = Measure("ds")(subdomain_data=facet_domains)
      dsi = ds(inlet_facet_ID, domain=mesh, subdomain_data=facet_domains)
      # Compute area of boundary tesselation by integrating 1.0 over all facets
      A = assemble(Constant(1.0, name="one")*dsi)
      inlet_radius = numpy.sqrt(A / pi) # This old estimate is a few % lower because of boundary discretization errors
      inlet_radius2 = A / pi #It should be r**2 in the parabolic eqn below!
      if MPI.rank(mpi_comm_world()) == 0:
          print 'Inlet radius after rotation (1):', inlet_radius
      # Compute barycenter by integrating x components over all facets
      center = [assemble(x[i]*dsi) / A for i in xrange(d)]
      if MPI.rank(mpi_comm_world()) == 0:
              print 'Inlet center after rotation (1):', center
      # Compute barycenter by integrating x components over all facets
      center = [assemble(x[i]*dsi) / A for i in xrange(d)]
      center = Point(numpy.array(center))
      n_normal= FacetNormal(mesh)
      ni = numpy.array([assemble(n_normal[i]*dsi) for i in xrange(d)])
      n_len = numpy.sqrt(sum([ni[i]**2 for i in xrange(d)])) # Should always be 1!?
      normal = ni/n_len
      if MPI.rank(mpi_comm_world()) == 0:
              print 'normal vector after rotation (1): ',normal
      alpha_ang = (normal[0]/abs(normal[0]))*acos(normal[0])*180./pi
      beta_ang = (normal[1]/abs(normal[1]))*acos(normal[1])*180./pi
      gamma_ang = (normal[2]/abs(normal[2]))*acos(normal[2])*180./pi

      #MeshTransformation.rotate(mesh, gamma_ang, 1, center)
      mesh.rotate(gamma_ang, 1, center)

      d = mesh.geometry().dim()
      x = SpatialCoordinate(mesh)
      ds = Measure("ds")(subdomain_data=facet_domains)
      dsi = ds(inlet_facet_ID, domain=mesh, subdomain_data=facet_domains)
      A = assemble(Constant(1.0, name="one")*dsi)
      if MPI.rank(mpi_comm_world()) == 0:
              print 'Inlet area after rotation:', A
      inlet_radius = numpy.sqrt(A / pi) # This old estimate is a few % lower because of boundary discretization errors
      inlet_radius2 = A / pi #It should be r**2 in the parabolic eqn below!
      if MPI.rank(mpi_comm_world()) == 0:
          print 'Inlet radius after rotation:', inlet_radius
      # Compute barycenter by integrating x components over all facets
      center = [assemble(x[i]*dsi) / A for i in xrange(d)]
      if MPI.rank(mpi_comm_world()) == 0:
              print 'Inlet center after rotation:', center
      n_normal= FacetNormal(mesh)
      ni = numpy.array([assemble(n_normal[i]*dsi) for i in xrange(d)])
      n_len = numpy.sqrt(sum([ni[i]**2 for i in xrange(d)])) # Should always be 1!?
      normal = ni/n_len
      if MPI.rank(mpi_comm_world()) == 0:
              print 'normal vector after rotation: ',normal



# Use dictionary to hold all FunctionSpaces
VV = dict((ui, V) for ui in uc_comp)
VV['p'] = Q

# Create dictionaries for the solutions at three timesteps
q_  = dict((ui, Function(VV[ui], name=ui)) for ui in sys_comp)
q_1 = dict((ui, Function(VV[ui], name=ui + "_1")) for ui in sys_comp)
q_2 = dict((ui, Function(V, name=ui + "_2")) for ui in u_components)

# Read in previous solution if restarting
init_from_restart(**vars())

# Create vectors of the segregated velocity components
u_  = as_vector([q_ [ui] for ui in u_components]) # Velocity vector at t
u_1 = as_vector([q_1[ui] for ui in u_components]) # Velocity vector at t - dt
u_2 = as_vector([q_2[ui] for ui in u_components]) # Velocity vector at t - 2*dt

# Adams Bashforth  ion of velocity at t - dt/2
U_AB = 1.5 * u_1 - 0.5 * u_2

# Create short forms for accessing the solution vectors
x_ = dict((ui, q_[ui].vector()) for ui in sys_comp)        # Solution vectors t
x_1 = dict((ui, q_1[ui].vector()) for ui in sys_comp)      # Solution vectors t - dt
x_2 = dict((ui, q_2[ui].vector()) for ui in u_components)  # Solution vectors t - 2*dt

# Create vectors to hold rhs of equations
b = dict((ui, Vector(x_[ui])) for ui in sys_comp)      # rhs vectors (final)
b_tmp = dict((ui, Vector(x_[ui])) for ui in sys_comp)  # rhs temp storage vectors

# Short forms pressure and scalars
p_ = q_['p']                # pressure at t
p_1 = q_1['p']              # pressure at t - dt
dp_ = Function(Q)           # pressure correction
for ci in scalar_components:
    exec("{}_   = q_ ['{}']".format(ci, ci))
    exec("{}_1  = q_1['{}']".format(ci, ci))

print_solve_info = use_krylov_solvers and krylov_solvers['monitor_convergence']

# Anything problem specific
vars().update(pre_solve_hook(**vars())) #place it before update, since it needs n_normal and facet_domains. Also before create_bcs

# Boundary conditions
if (flag_ramp):
    t = initial_time_ramp
t_cycle = t
bcs = create_bcs(**vars())

if(0):
 # LES setup
 #exec("from oasis.solvers.NSfracStep.LES.{} import *".format(les_model))
 lesmodel = importlib.import_module('.'.join(('oasis.solvers.NSfracStep.LES', les_model)))
 vars().update({name:lesmodel.__dict__[name] for name in lesmodel.__all__})
if(1):
 # LES setup
 exec("from solvers.NSfracStep.LES.{} import *".format(les_model))

vars().update(les_setup(**vars()))

# Initialize solution
initialize(**vars())

#  Fetch linear algebra solvers
u_sol, p_sol, c_sol = get_solvers(**vars())

# Get constant body forces
f = body_force(**vars())
assert(isinstance(f, Coefficient))
b0 = dict((ui, assemble(v * f[i] * dx)) for i, ui in enumerate(u_components))

# Get scalar sources
fs = scalar_source(**vars())
for ci in scalar_components:
    assert(isinstance(fs[ci], Coefficient))
    b0[ci] = assemble(v * fs[ci] * dx)


##test gamma
#V_test = FunctionSpace(mesh, 'CG', 1)
#gamma_soln = TrialFunction(V_test)
#gamma_solution = Function(V_test)
#file_gamma = File('/Users/aa3878/data/nonNewtonian/AAA18/results_oasis/O1/test_gamma/gamma.pvd')

# Preassemble and allocate
vars().update(setup(**vars()))



tic()
stop = False
total_timer = OasisTimer("Start simulations", True)
t_cycle = t
while t < (T - tstep * DOLFIN_EPS) and not stop:
    if (t_cycle > Time_last):
        t_cycle = t_cycle - Time_last
    if(1): #update time in BC if time-dependent inflow expression is specified
      bcs = create_bcs(**vars())
    t += dt
    t_cycle +=dt
    tstep += 1
    inner_iter = 0
    udiff = array([1e8])  # Norm of velocity change over last inner iter
    num_iter = max(iters_on_first_timestep, max_iter) if tstep == 1 else max_iter

    start_timestep_hook(**vars())

    while udiff[0] > max_error and inner_iter < num_iter:
        inner_iter += 1

        t0 = OasisTimer("Tentative velocity")
        if inner_iter == 1:
            les_update(**vars())
            assemble_first_inner_iter(**vars())
        udiff[0] = 0.0
        for i, ui in enumerate(u_components):
            t1 = OasisTimer('Solving tentative velocity ' + ui, print_solve_info)
            velocity_tentative_assemble(**vars())
            velocity_tentative_hook(**vars())
            velocity_tentative_solve(**vars())
            t1.stop()

        
        # if (Resistance_flag):

        #  if tstep > 20:

        #    #ds = Measure("ds")[facet_domains]
        #    ds = Measure("ds")(subdomain_data=facet_domains)
        #    flow1 = abs( assemble(dot(u_, n_normal)*ds(3, domain=mesh, subdomain_data=facet_domains) ) )
        #    flow2 = abs( assemble(dot(u_, n_normal)*ds(4, domain=mesh, subdomain_data=facet_domains) ) )
        #    flow3 = abs( assemble(dot(u_, n_normal)*ds(5, domain=mesh, subdomain_data=facet_domains) ) )
        #    flow4 = abs( assemble(dot(u_, n_normal)*ds(6, domain=mesh, subdomain_data=facet_domains) ) )
        #    inflow = abs( assemble(dot(u_, n_normal)*ds(1, domain=mesh, subdomain_data=facet_domains) ) )


        #    total_flow = flow1 + flow2 + flow3 + flow4
        #    ratios = numpy.zeros(4)
        #    current_ratios = numpy.zeros(4)
        #    ratios_threshold = numpy.zeros(4)
        #    target_ratios = [0.4264, 0.2871, 0.1455, 0.141]

        #    current_ratios[0] = flow1/(inflow + 1e-16)
        #    current_ratios[1] = flow2/(inflow + 1e-16)
        #    current_ratios[2] = flow3/(inflow + 1e-16)
        #    current_ratios[3] = flow4/(inflow + 1e-16)

        #    for i in xrange(4):
        #         ratios[i] = current_ratios[i]/target_ratios[i]
        #         if MPI.rank(mpi_comm_world()) == 0:
        #             print 'ratio(',i,') = ', ratios[i]
        #         ratios_threshold[i] = ratios[i] < 0.99 or ratios[i] > 1.01

        #         if ratios_threshold[i] and i == 3:
        #           Res4 *= ratios[i]
        #           if MPI.rank(mpi_comm_world()) == 0:
        #             print 'Res4 has changed!'
        #         elif ratios_threshold[i] and i == 2:
        #           Res3 *= ratios[i]
        #           if MPI.rank(mpi_comm_world()) == 0:
        #             print 'Res3 has changed!'
        #         elif ratios_threshold[i] and i == 1:
        #           Res2 *= ratios[i]
        #           if MPI.rank(mpi_comm_world()) == 0:
        #             print 'Res2 has changed!'
        #         elif ratios_threshold[i] and i == 0:
        #           Res1 *= ratios[i]
        #           if MPI.rank(mpi_comm_world()) == 0:
        #             print 'Res1 has changed!'

        #  if MPI.rank(mpi_comm_world()) == 0:
        #        print 'Res1 = ', Res1, 'Res2 = ', Res2, 'Res3 = ', Res3, 'Res4 = ', Res4



        t0 = OasisTimer("Pressure solve", print_solve_info)
        pressure_assemble(**vars())
        pressure_hook(**vars())
        pressure_solve(**vars())
        t0.stop()

        print_velocity_pressure_info(**vars())

    # Update velocity
    t0 = OasisTimer("Velocity update")
    if MPI.rank(mpi_comm_world()) == 0:
        print 'updating vel'
    velocity_update(**vars())
    t0.stop()

    # Solve for scalars
    if len(scalar_components) > 0:
        scalar_assemble(**vars())
        for ci in scalar_components:
            t1 = OasisTimer('Solving scalar {}'.format(ci), print_solve_info)
            scalar_hook(**vars())
            scalar_solve(**vars())
            t1.stop()


    temporal_hook(**vars())

    # Save solution if required and check for killoasis file
    stop = save_solution(**vars())


    # Update to a new timestep
    if MPI.rank(mpi_comm_world()) == 0:
        print 'updating soln'
        print 'Time-step: ', tstep
    for ui in u_components:
        x_2[ui].zero()
        x_2[ui].axpy(1.0, x_1[ui])
        x_1[ui].zero()
        x_1[ui].axpy(1.0, x_[ui])
        #new method (Amir):
        #x_2[ui].set_local(x_1[ui].array())
        #x_2[ui].apply("insert")
        #x_1[ui].set_local(x_[ui].array())
        #x_1[ui].apply("insert")


    for ci in scalar_components:
        x_1[ci].zero()
        x_1[ci].axpy(1., x_[ci])

    if MPI.rank(mpi_comm_world()) == 0:
        print 'update done'

    # Print some information
    if tstep % print_intermediate_info == 0:
        info_green( 'Time = {0:2.4e}, timestep = {1:6d}, End time = {2:2.4e}'.format(t, tstep, T))
        info_red('Total computing time on previous {0:d} timesteps = {1:f}'.format(
            print_intermediate_info, toc()))
        list_timings(TimingClear_clear, [TimingType_wall])
        tic()

    ##test visc_gamma
    #if (tstep % 20 == 0):
    #    print '------projecting..'
    #    solve(K_test == K_test_rhs, gamma_solution,solver_parameters={'linear_solver': 'gmres'})
    #    file_gamma <<gamma_solution

    # AB projection for pressure on next timestep
    if AB_projection_pressure and t < (T - tstep * DOLFIN_EPS) and not stop:
        x_['p'].axpy(0.5, dp_.vector())



total_timer.stop()
list_timings(TimingClear_keep, [TimingType_wall])
info_red('Total computing time = {0:f}'.format(total_timer.elapsed()[0]))
oasis_memory('Final memory use ')
total_initial_dolfin_memory = MPI.sum(mpi_comm_world(), initial_memory_use)
info_red('Memory use for importing dolfin = {} MB (RSS)'.format(
    total_initial_dolfin_memory))
info_red('Total memory use of solver = ' +
         str(oasis_memory.memory - total_initial_dolfin_memory) + " MB (RSS)")

# Final hook
theend_hook(**vars())
