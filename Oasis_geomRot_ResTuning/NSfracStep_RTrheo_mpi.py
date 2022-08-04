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

#RTrheo: the new RT based non-Newtonian model

import sys, os #added by Amir

#sys.path.append(os.getcwd()) #added by Amir
import importlib
#from oasis import * #added by amir
#from oasis.common import *
from common import * #change by Amir
import numpy
#from problems import * #added by Amir
#from problems.NSfracStep import * #added by Amir


#########Set parameters below for the RT  rheology model

#Set the RT_thresh and RT_filename parameter in the problem file
file_initial = 0
file_stop = 19
file_interval = 1
tsteps_per_file = 842   #263       #842   #(T=0.95/19)/(dt)   #105   #(T=1.053/19)/(dt)

########

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



V_RT = FunctionSpace(mesh, 'CG', 1)
RT_time = Function(V_RT)
RT_time2 = Function(V_RT)
#v2d_V = dof_to_vertex_map(V_RT)
v2d = vertex_to_dof_map(V_RT)
#dofmap = V_RT.dofmap()
#my_first, my_last = dofmap.ownership_range() # global
#dofs = V_RT.dofmap().dofs(mesh, 0)

# Preassemble and allocate
vars().update(setup(**vars()))



tic()
stop = False
total_timer = OasisTimer("Start simulations", True)


t_cycle = t

num_nodes = mesh.num_vertices()
num_nodes_p = num_nodes
RT_nparray = numpy.zeros(num_nodes_p)
file_final = file_initial + file_interval

RTvalues = RT_time.vector().get_local()
while t < (T - tstep * DOLFIN_EPS) and not stop:
   


    RT_initial = Function(V_RT,RT_filename +str(file_initial)+'.xml')
    RT_final = Function(V_RT,RT_filename +str(file_final)+'.xml')

    for tstep_inter in xrange(tsteps_per_file):
     if (t_cycle > Time_last): #this was outside for loop for RT and RT10 results
         t_cycle = t_cycle - Time_last
     if (t >= T - tstep * DOLFIN_EPS ):
       break
     #update time in BC if time-dependent inflow expression is specified
     bcs = create_bcs(**vars())

     alpha_interp = (tstep_inter * 1.0) / (tsteps_per_file * 1.0)
     RT_time2.vector().set_local((1. - alpha_interp) * RT_initial.vector().get_local()+ alpha_interp *  RT_final.vector().get_local())
   
     #for i in xrange(num_nodes_p):
        #RT_nparray[i] = 1.0 * ( RT_time2.vector()[i] > RT_thresh) #RT_time.vector()[i] does not work in parallel!
    
     RT_time.vector().set_local( 1.0 * ( RT_time2.vector().get_local()[:] > RT_thresh ) ) #this works in parallel
     RT_time.vector().apply('')
     
     #check:
     #File('/scratch/aa3878/IA_oasis/RT_check.pvd') << RT_time
     #File('/Users/aa3878/data/nonNewtonian/AAA18/results_oasis/temp/RT_check.pvd') << RT_time #works!
     
    
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

        t0 = OasisTimer("Pressure solve", print_solve_info)
        pressure_assemble(**vars())
        pressure_hook(**vars())
        pressure_solve(**vars())
        t0.stop()

        print_velocity_pressure_info(**vars())

     # Update velocity
     t0 = OasisTimer("Velocity update")
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
     for ui in u_components:
        x_2[ui].zero()
        x_2[ui].axpy(1.0, x_1[ui])
        x_1[ui].zero()
        x_1[ui].axpy(1.0, x_[ui])

     for ci in scalar_components:
        x_1[ci].zero()
        x_1[ci].axpy(1., x_[ci])

     # Print some information
     if tstep % print_intermediate_info == 0:
        info_green( 'Time = {0:2.4e}, timestep = {1:6d}, End time = {2:2.4e}'.format(t, tstep, T))
        info_red('Total computing time on previous {0:d} timesteps = {1:f}'.format(
            print_intermediate_info, toc()))
        list_timings(TimingClear_clear, [TimingType_wall])
        tic()

     # AB projection for pressure on next timestep
     if AB_projection_pressure and t < (T - tstep * DOLFIN_EPS) and not stop:
        x_['p'].axpy(0.5, dp_.vector())

     file_final = file_initial + file_interval
     if file_final > file_stop:
        file_final = file_start

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
