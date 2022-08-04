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

bc_file =  '/scratch/mm4238/prostate_CFD/prostate_ext_Mar21/mesh/BCnodeFacets.xml'
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
if(0):
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


# ******************************************************************************************
# ****************************** Automated Geometry Rotation *******************************
# ******************************************************************************************

if (1):

    target_normal = [0, 0, 1]

    # Calculating centroid, radius, and normal vector of the inlet surface

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

     
    dot_product = numpy.dot(normal, target_normal)
    cross_product = numpy.cross(normal, target_normal)

    v_skew = numpy.zeros((3, 3))

    v_skew[0][0] = 0.0
    v_skew[0][1] = -cross_product[2]
    v_skew[0][2] = cross_product[1]

    v_skew[1][0] = cross_product[2]
    v_skew[1][1] = 0.0
    v_skew[1][2] = -cross_product[0]

    v_skew[2][0] = -cross_product[1]
    v_skew[2][1] = cross_product[0]
    v_skew[0][1] = 0.0

    rotation_matrix = numpy.identity(3) + v_skew + numpy.matmul(v_skew, v_skew) * (1/(1 + dot_product))

    # Finding the Euler angles

    alpha_ang = numpy.arctan2(rotation_matrix[2][1], rotation_matrix[2][2]) * 180/numpy.pi
    beta_ang = numpy.arctan2(-rotation_matrix[2][0], numpy.sqrt(rotation_matrix[2][1]**2 + rotation_matrix[2][2]**2) ) * 180/numpy.pi
    gamma_ang = numpy.arctan2(rotation_matrix[1][0], rotation_matrix[0][0]) * 180/numpy.pi

    if MPI.rank(mpi_comm_world()) == 0:
            print 'Euler Angels: ', 'theta_x = ', alpha_ang, 'theta_y = ', beta_ang, 'theta_z = ', gamma_ang 


    # Rotation around x-axis

    mesh.rotate(alpha_ang, 0, center)

    d = mesh.geometry().dim()
    x = SpatialCoordinate(mesh)
    ds = Measure("ds")(subdomain_data=facet_domains)
    dsi = ds(inlet_facet_ID, domain=mesh, subdomain_data=facet_domains)
    # Compute area of boundary tesselation by integrating 1.0 over all facets
    A = assemble(Constant(1.0, name="one")*dsi)
    inlet_radius = numpy.sqrt(A / pi) # This old estimate is a few % lower because of boundary discretization errors
    inlet_radius2 = A / pi #It should be r**2 in the parabolic eqn below!
    if MPI.rank(mpi_comm_world()) == 0:
        print 'Inlet radius after rotation aound X-axis:', inlet_radius
    # Compute barycenter by integrating x components over all facets
    center = [assemble(x[i]*dsi) / A for i in xrange(d)]
    if MPI.rank(mpi_comm_world()) == 0:
            print 'Inlet center after rotation aound X-axis:', center
    # Compute barycenter by integrating x components over all facets
    center = [assemble(x[i]*dsi) / A for i in xrange(d)]
    center = Point(numpy.array(center))
    n_normal= FacetNormal(mesh)
    ni = numpy.array([assemble(n_normal[i]*dsi) for i in xrange(d)])
    n_len = numpy.sqrt(sum([ni[i]**2 for i in xrange(d)])) # Should always be 1!?
    normal = ni/n_len

    if MPI.rank(mpi_comm_world()) == 0:
            print 'normal vector after rotation around X-axis: ', normal

    # Rotation around y-axis
    mesh.rotate(-beta_ang, 1, center)


    d = mesh.geometry().dim()
    x = SpatialCoordinate(mesh)
    ds = Measure("ds")(subdomain_data=facet_domains)
    dsi = ds(inlet_facet_ID, domain=mesh, subdomain_data=facet_domains)
    # Compute area of boundary tesselation by integrating 1.0 over all facets
    A = assemble(Constant(1.0, name="one")*dsi)
    inlet_radius = numpy.sqrt(A / pi) # This old estimate is a few % lower because of boundary discretization errors
    inlet_radius2 = A / pi #It should be r**2 in the parabolic eqn below!
    if MPI.rank(mpi_comm_world()) == 0:
        print 'Inlet radius after rotation aound Y-axis:', inlet_radius
    # Compute barycenter by integrating x components over all facets
    center = [assemble(x[i]*dsi) / A for i in xrange(d)]
    if MPI.rank(mpi_comm_world()) == 0:
            print 'Inlet center after rotation aound Y-axis:', center
    # Compute barycenter by integrating x components over all facets
    center = [assemble(x[i]*dsi) / A for i in xrange(d)]
    center = Point(numpy.array(center))
    n_normal= FacetNormal(mesh)
    ni = numpy.array([assemble(n_normal[i]*dsi) for i in xrange(d)])
    n_len = numpy.sqrt(sum([ni[i]**2 for i in xrange(d)])) # Should always be 1!?
    normal = ni/n_len

    if MPI.rank(mpi_comm_world()) == 0:
            print 'normal vector after rotation around Y-axis: ', normal

    # Rotation around z-axis
    mesh.rotate(-gamma_ang, 2, center)


    d = mesh.geometry().dim()
    x = SpatialCoordinate(mesh)
    ds = Measure("ds")(subdomain_data=facet_domains)
    dsi = ds(inlet_facet_ID, domain=mesh, subdomain_data=facet_domains)
    # Compute area of boundary tesselation by integrating 1.0 over all facets
    A = assemble(Constant(1.0, name="one")*dsi)
    inlet_radius = numpy.sqrt(A / pi) # This old estimate is a few % lower because of boundary discretization errors
    inlet_radius2 = A / pi #It should be r**2 in the parabolic eqn below!
    if MPI.rank(mpi_comm_world()) == 0:
        print 'Inlet radius after rotation aound Z-axis:', inlet_radius
    # Compute barycenter by integrating x components over all facets
    center = [assemble(x[i]*dsi) / A for i in xrange(d)]
    if MPI.rank(mpi_comm_world()) == 0:
            print 'Inlet center after rotation aound Z-axis:', center
    # Compute barycenter by integrating x components over all facets
    center = [assemble(x[i]*dsi) / A for i in xrange(d)]
    center = Point(numpy.array(center))
    n_normal= FacetNormal(mesh)
    ni = numpy.array([assemble(n_normal[i]*dsi) for i in xrange(d)])
    n_len = numpy.sqrt(sum([ni[i]**2 for i in xrange(d)])) # Should always be 1!?
    normal = ni/n_len

    if MPI.rank(mpi_comm_world()) == 0:
            print 'normal vector after rotation around Z-axis: ', normal





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

        
        if (Resistance_flag and tstep%10==0):

         if tstep > 20:

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


           total_flow = flow1 + flow2 + flow3 + flow4 + flow5 + flow6 + flow7 + flow8 + flow9 + flow10 + flow11 + flow12 + flow13 + flow14 + flow15 + flow16 + flow17 + flow18 + flow19 + flow20 + flow21 + flow22 + flow23 + flow24 + flow25 + flow26 + flow27 + flow28 + flow29 + flow30 + flow31
           ratios = numpy.zeros(31)
           current_ratios = numpy.zeros(31)
           ratios_threshold = numpy.zeros(31)
           target_ratios = [0.19019, 0.04548, 0.03555, 0.0191, 0.01058, 0.01002, 0.17685, 0.00233, 0.0263, 0.00534, 0.00808, 0.03579, 0.04292, 0.02369, 0.01002, 0.00535, 0.01446, 0.00233, 0.0161, 0.00534, 0.0787, 0.02862, 0.01985, 0.02369, 0.0075, 0.03579, 0.02012, 0.01058, 0.06477, 0.01228, 0.01228]

           current_ratios[0] = flow1/(inflow + 1e-16)
           current_ratios[1] = flow2/(inflow + 1e-16)
           current_ratios[2] = flow3/(inflow + 1e-16)
           current_ratios[3] = flow4/(inflow + 1e-16)
           current_ratios[4] = flow5/(inflow + 1e-16)
           current_ratios[5] = flow6/(inflow + 1e-16)
           current_ratios[6] = flow7/(inflow + 1e-16)
           current_ratios[7] = flow8/(inflow + 1e-16)
           current_ratios[8] = flow9/(inflow + 1e-16)
           current_ratios[9] = flow10/(inflow + 1e-16)
           current_ratios[10] = flow11/(inflow + 1e-16)
           current_ratios[11] = flow12/(inflow + 1e-16)
           current_ratios[12] = flow13/(inflow + 1e-16)
           current_ratios[13] = flow14/(inflow + 1e-16)
           current_ratios[14] = flow15/(inflow + 1e-16)
           current_ratios[15] = flow16/(inflow + 1e-16)
           current_ratios[16] = flow17/(inflow + 1e-16)
           current_ratios[17] = flow18/(inflow + 1e-16)
           current_ratios[18] = flow19/(inflow + 1e-16)
           current_ratios[19] = flow20/(inflow + 1e-16)
           current_ratios[20] = flow21/(inflow + 1e-16)
           current_ratios[21] = flow22/(inflow + 1e-16)
           current_ratios[22] = flow23/(inflow + 1e-16)
           current_ratios[23] = flow24/(inflow + 1e-16)
           current_ratios[24] = flow25/(inflow + 1e-16)
           current_ratios[25] = flow26/(inflow + 1e-16)
           current_ratios[26] = flow27/(inflow + 1e-16)
           current_ratios[27] = flow28/(inflow + 1e-16)
           current_ratios[28] = flow29/(inflow + 1e-16)
           current_ratios[29] = flow30/(inflow + 1e-16)
           current_ratios[30] = flow31/(inflow + 1e-16)

           for i in xrange(31):
                ratios[i] = current_ratios[i]/target_ratios[i]
                if MPI.rank(mpi_comm_world()) == 0:
                    print 'ratio(',i,') = ', ratios[i]
                ratios_threshold[i] = ratios[i] < 0.999 or ratios[i] > 1.001

                if ratios_threshold[i] and i == 30:
                  Res31 *= ratios[i]
                  if Res31<1e-6:
                    Res31 = 2
                  if MPI.rank(mpi_comm_world()) == 0:
                    print 'Res31 has changed!'
                elif ratios_threshold[i] and i == 29:
                  Res30 *= ratios[i]
                  if Res30<1e-6:
                    Res30 = 2
                  if MPI.rank(mpi_comm_world()) == 0:
                    print 'Res30 has changed!'
                elif ratios_threshold[i] and i == 28:
                  Res29 *= ratios[i]
                  if Res29<1e-6:
                    Res29 = 2
                  if MPI.rank(mpi_comm_world()) == 0:
                    print 'Res29 has changed!'
                elif ratios_threshold[i] and i == 27:
                  Res28 *= ratios[i]
                  if Res28<1e-6:
                    Res28 = 2
                  if MPI.rank(mpi_comm_world()) == 0:
                    print 'Res28 has changed!'
                elif ratios_threshold[i] and i == 26:
                  Res27 *= ratios[i]
                  if Res27<1e-6:
                    Res27 = 2
                  if MPI.rank(mpi_comm_world()) == 0:
                    print 'Res27 has changed!'
                elif ratios_threshold[i] and i == 25:
                  Res26 *= ratios[i]
                  if Res26<1e-6:
                    Res26 = 2
                  if MPI.rank(mpi_comm_world()) == 0:
                    print 'Res26 has changed!'
                elif ratios_threshold[i] and i == 24:
                  Res25 *= ratios[i]
                  if Res25<1e-6:
                    Res25 = 2
                  if MPI.rank(mpi_comm_world()) == 0:
                    print 'Res25 has changed!'
                elif ratios_threshold[i] and i == 23:
                  Res24 *= ratios[i]
                  if Res24<1e-6:
                    Res24 = 2
                  if MPI.rank(mpi_comm_world()) == 0:
                    print 'Res24 has changed!'
                elif ratios_threshold[i] and i == 22:
                  Res23 *= ratios[i]
                  if Res23<1e-6:
                    Res23 = 2
                  if MPI.rank(mpi_comm_world()) == 0:
                    print 'Res23 has changed!'
                elif ratios_threshold[i] and i == 21:
                  Res22 *= ratios[i]
                  if Res22<1e-6:
                    Res22 = 2
                  if MPI.rank(mpi_comm_world()) == 0:
                    print 'Res22 has changed!'
                elif ratios_threshold[i] and i == 20:
                  Res21 *= ratios[i]
                  if Res21<1e-6:
                    Res21 = 2
                  if MPI.rank(mpi_comm_world()) == 0:
                    print 'Res21 has changed!'
                elif ratios_threshold[i] and i == 19:
                  Res20 *= ratios[i]
                  if Res20<1e-6:
                    Res20 = 2
                  if MPI.rank(mpi_comm_world()) == 0:
                    print 'Res20 has changed!'
                elif ratios_threshold[i] and i == 18:
                  Res19 *= ratios[i]
                  if Res19<1e-6:
                    Res19 = 2
                  if MPI.rank(mpi_comm_world()) == 0:
                    print 'Res19 has changed!'
                elif ratios_threshold[i] and i == 17:
                  Res18 *= ratios[i]
                  if Res18<1e-6:
                    Res18 = 2
                  if MPI.rank(mpi_comm_world()) == 0:
                    print 'Res18 has changed!'
                elif ratios_threshold[i] and i == 16:
                  Res17 *= ratios[i]
                  if Res17<1e-6:
                    Res17 = 2
                  if MPI.rank(mpi_comm_world()) == 0:
                    print 'Res17 has changed!'
                elif ratios_threshold[i] and i == 15:
                  Res16 *= ratios[i]
                  if Res16<1e-6:
                    Res16 = 2
                  if MPI.rank(mpi_comm_world()) == 0:
                    print 'Res16 has changed!'
                elif ratios_threshold[i] and i == 14:
                  Res15 *= ratios[i]
                  if Res15<1e-6:
                    Res15 = 2
                  if MPI.rank(mpi_comm_world()) == 0:
                    print 'Res15 has changed!'
                elif ratios_threshold[i] and i == 13:
                  Res14 *= ratios[i]
                  if Res14<1e-6:
                    Res14 = 2
                  if MPI.rank(mpi_comm_world()) == 0:
                    print 'Res14 has changed!'
                elif ratios_threshold[i] and i == 12:
                  Res13 *= ratios[i]
                  if Res13<1e-6:
                    Res13 = 2
                  if MPI.rank(mpi_comm_world()) == 0:
                    print 'Res13 has changed!'
                elif ratios_threshold[i] and i == 11:
                  Res12 *= ratios[i]
                  if Res12<1e-6:
                    Res12 = 2
                  if MPI.rank(mpi_comm_world()) == 0:
                    print 'Res12 has changed!'
                elif ratios_threshold[i] and i == 10:
                  Res11 *= ratios[i]
                  if Res11<1e-6:
                    Res11 = 2
                  if MPI.rank(mpi_comm_world()) == 0:
                    print 'Res11 has changed!'
                elif ratios_threshold[i] and i == 9:
                  Res10 *= ratios[i]
                  if Res10<1e-6:
                    Res10 = 2
                  if MPI.rank(mpi_comm_world()) == 0:
                    print 'Res10 has changed!'
                elif ratios_threshold[i] and i == 8:
                  Res9 *= ratios[i]
                  if Res9<1e-6:
                    Res9 = 2
                  if MPI.rank(mpi_comm_world()) == 0:
                    print 'Res9 has changed!'
                elif ratios_threshold[i] and i == 7:
                  Res8 *= ratios[i]
                  if Res8<1e-6:
                    Res8 = 2
                  if MPI.rank(mpi_comm_world()) == 0:
                    print 'Res8 has changed!'
                elif ratios_threshold[i] and i == 6:
                  Res7 *= ratios[i]
                  if Res7<1e-6:
                    Res7 = 2
                  if MPI.rank(mpi_comm_world()) == 0:
                    print 'Res7 has changed!'
                elif ratios_threshold[i] and i == 5:
                  Res6 *= ratios[i]
                  if Res6<1e-6:
                    Res6 = 2
                  if MPI.rank(mpi_comm_world()) == 0:
                    print 'Res6 has changed!'
                elif ratios_threshold[i] and i == 4:
                  Res5 *= ratios[i]
                  if Res5<1e-6:
                    Res5 = 2
                  if MPI.rank(mpi_comm_world()) == 0:
                    print 'Res5 has changed!'
                elif ratios_threshold[i] and i == 3:
                  Res4 *= ratios[i]
                  if Res4<1e-6:
                    Res4 = 2
                  if MPI.rank(mpi_comm_world()) == 0:
                    print 'Res4 has changed!'
                elif ratios_threshold[i] and i == 2:
                  Res3 *= ratios[i]
                  if Res3<1e-6:
                    Res3 = 2
                  if MPI.rank(mpi_comm_world()) == 0:
                    print 'Res3 has changed!'
                elif ratios_threshold[i] and i == 1:
                  Res2 *= ratios[i]
                  if Res2<1e-6:
                    Res2 = 2
                  if MPI.rank(mpi_comm_world()) == 0:
                    print 'Res2 has changed!'
                elif ratios_threshold[i] and i == 0:
                  Res1 *= ratios[i]
                  if Res1<1e-6:
                    Res1 = 2
                  if MPI.rank(mpi_comm_world()) == 0:
                    print 'Res1 has changed!'
                


                # if ratios_threshold[0]:
                #     Res1 *= ratios[0]
                #     if MPI.rank(mpi_comm_world()) == 0:
                #        print 'Res1 has changed!'
                # if not ratios_threshold[0]:
                #     if ratios_threshold[1]:
                #        Res2 *= ratios[1]
                #        if MPI.rank(mpi_comm_world()) == 0:
                #           print 'Res2 has changed!'
                #     if not ratios_threshold[1]:
                #          if ratios_threshold[2]:
                #             Res3 *= ratios[2]
                #             if MPI.rank(mpi_comm_world()) == 0:
                #                print 'Res3 has changed!'
                #          if not ratios_threshold[2]:
                #             if ratios_threshold[3]:
                #                Res4 *= ratios[3]
                #                if MPI.rank(mpi_comm_world()) == 0:
                #                   print 'Res4 has changed!'
         if MPI.rank(mpi_comm_world()) == 0:
               print 'Res1 = ', Res1, 'Res2 = ', Res2, 'Res3 =', Res3
               print 'Res4 = ', Res4, 'Res5 = ', Res5, 'Res6 =', Res6
               print 'Res7 = ', Res7, 'Res8 = ', Res8, 'Res9 =', Res9
               print 'Res10 = ', Res10, 'Res11 = ', Res11, 'Res12 =', Res12
               print 'Res13 = ', Res13, 'Res14 = ', Res14, 'Res15 =', Res15
               print 'Res16 = ', Res16, 'Res17 = ', Res17, 'Res18 =', Res18
               print 'Res19 = ', Res19, 'Res20 = ', Res20, 'Res21 =', Res21
               print 'Res22 = ', Res22, 'Res23 = ', Res23, 'Res24 =', Res24
               print 'Res25 = ', Res25, 'Res26 = ', Res26, 'Res27 =', Res27
               print 'Res28 = ', Res28, 'Res29 = ', Res29, 'Res30 =', Res30
               print 'Res31 = ', Res31



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
