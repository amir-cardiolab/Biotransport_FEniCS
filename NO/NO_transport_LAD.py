__author__ = "Mostafa Mahmoudi <mm4238@nau.edu>"
__date__ = "29-06-2022"
__copyright__ = "Copyright (C) 2022 " + __author__
__license__ = "Free - Refer to README.MD to learn how to cite this package."


from dolfin import *

import numpy as np

import time


# TO RUN: mpirun -np #of_procs python filename.py

#********************************************************
#********************* FE Parameters ********************
#********************************************************

set_log_level(30)


relative_tolerance = 1e-13 #1e-6 This was original value

stabilized = True #False

aneurysm_bc_type = 'neumann'

parameters['form_compiler']['representation'] = 'uflacs'

parameters['form_compiler']['optimize'] = True

parameters['form_compiler']['cpp_optimize'] = True

#********************************************************
#******************* Working Directory ******************
#********************************************************

root_dir = '/scratch/mm4238/transport/3A_2016_LAD/'

if MPI.rank(mpi_comm_world()) == 0:
	print 'Root_dir:', root_dir

results_dir = root_dir + 'NO_transport/Results/'

mesh_filename = root_dir + 'mesh_transport/3A_2016_LAD.xml'

bc_file = root_dir + 'mesh_transport/BCnodeFacets.xml' #ORder: inlet, wall, outlet1

#********************************************************
#********************* Reading Mesh *********************
#********************************************************

if MPI.rank(mpi_comm_world()) == 0:
	print 'Loading mesh:', mesh_filename

mesh = Mesh(mesh_filename)

facet_domains = MeshFunction('size_t', mesh, bc_file)

#mesh_in = HDF5File(mpi_comm_world(), mesh_filename, 'r')

#mesh_in.read(mesh, 'mesh', False)

#mesh_in.close()


#********************************************************
#**************** Specifying Subdomains *****************
#********************************************************


# Define a MeshFunction over 3 subdomains
subdomains = MeshFunction('size_t', mesh, 3)


V0 = FunctionSpace(mesh, 'DG', 0)
D_multiple = Function(V0) #2 different Diffusion coefs

# Defining the planes to create the region of interest
P1 = [15.19896, 23.9704, -4.91926]
n1 = [0.885152, -0.4096, 0.220757]

P2 = [13.62255, 24.20334, -5.312]
n2 = [-0.8656, 0.275995, -0.41785]

# my_eps = 0.01
class Omega_inside(SubDomain): #Region of interest
	def inside(self, x, on_boundary):
		return False  if (  n1[0] * (x[0] - P1[0] ) + n1[1] * ( x[1] - P1[1] ) + n1[2] * ( x[2] - P1[2] ) < 0  )  or (  n2[0] * (x[0] - P2[0] ) + n2[1] * ( x[1] - P2[1] ) + n2[2] * ( x[2] - P2[2] ) < 0 )  else True

class Omega_out1(SubDomain): #Inlet Region
	def inside(self, x, on_boundary):  #With eqn of a plane
		return True if (  n1[0] * (x[0] - P1[0] ) + n1[1] * ( x[1] - P1[1] ) + n1[2] * ( x[2] - P1[2] ) > 0 )    else False

class Omega_out2(SubDomain):  #Outlet Region
	def inside(self, x, on_boundary):
		return True if (  n2[0] * (x[0] - P2[0] ) + n2[1] * ( x[1] - P2[1] ) + n2[2] * ( x[2] - P2[2] ) > 0  ) else False


n_normal = FacetNormal(mesh)


#********************************************************
#*********** Reading Velocity and WSS Fields ************
#********************************************************


#******************** Velocity Field ********************
velocity_filename = root_dir + 'vel_xml/velocity_fineMesh_'

velocity_start =  191 #2020
velocity_stop = 290   #3000
velocity_interval = 1 #20

#*********************** WSS Field **********************
wss_filename = root_dir + 'wss_xml/wss_fineMesh_'

wss_start =  191 #2020
wss_stop = 290   #3000
wss_interval = 1 #20

#velocity_in = HDF5File(mpi_comm_world(), velocity_filename, 'r')
#velocity_prefix = '/velocity/vector_'


#********************************************************
#************** Reading Initial Conditions **************
#********************************************************

#Initial Condition from file
IC_flag = False #if True uses an I.C file
IC_file = root_dir + 'results/concentration2nd_D5_16000.h5'
IC_start_index = 16000

if MPI.rank(mpi_comm_world()) == 0:

	print 'Num global vertices', mesh.num_vertices()


#********************************************************
#******************* Saving Parameters ******************
#********************************************************

max_save_interval = 1 *1  #For writing max calues

solution_save_interval = 500 *1  #writing solution

solution_save_last= 45000 *1 #write more files for last cycles
solution_save_last_freq = 100 *1


#********************************************************
#*************** Governing Eq. Parameters ***************
#********************************************************

h =  CellDiameter(mesh)

basis_order = 1

#********************* Time Controls ********************

D = 3.3e-5
D_high = 1e-3

t_start = 0.0

t_stop = 10.0 #10cycles (T=0.88)

n_tsteps = 5000 * 10  #240000

tsteps_per_file = 50 * 1  #number of time steps btwn files (delta_t_file = 0.0176). = delta_t_file/(dt=T/1000)

dt = (t_stop - t_start) / n_tsteps #dt = 0.00088

#********************************************************
#********* Assigning various Diffusivity Coeffs *********
#********************************************************

# Mark subdomains with numbers 0 and 1
subdomains.set_all(2) #initialize the whole domain
subdomain0 = Omega_out1()
subdomain0.mark(subdomains, 0)
subdomain1 = Omega_out2()
subdomain1.mark(subdomains, 1)
subdomain2 = Omega_inside()
subdomain2.mark(subdomains, 2)

# Loop over all cell numbers, find corresponding
# subdomain number and fill cell value in k
sss = len(subdomains.array())
D_values = [D_high, D_high, D ]  # values of k in the two subdomains
D_numpy = np.zeros((sss))
for cell_no in xrange(sss):
	subdomain_no = subdomains.array()[cell_no]
	D_numpy[cell_no] = D_values[subdomain_no]


D_multiple.vector().set_local(D_numpy)

#*************** Check for Diff. Coeffs ****************
out_D_multiple = File(results_dir + 'Domain.pvd') 
out_D_multiple << D_multiple


#out_D = HDF5File(mpi_comm_world(),results_dir + 'D_coef.h5','w')
#out_bcs << facet_domains
#out_D.write(D_multiple,'D')
#out_D.close()

#********************************************************
#****************** Required Functions ******************
#********************************************************
# Create FunctionSpaces

Q = FunctionSpace(mesh, 'CG', basis_order)

V = VectorFunctionSpace(mesh, 'CG', 1)

v2d_V = dof_to_vertex_map(V)


#********************************************************
#*************** NO Transport Parameters ****************
#********************************************************

#rate of NO production constants

R_b = 2.13

R_b_space = Function(Q)
R_b_space = interpolate(Constant(R_b), Q) 

R_max = 457.5

K_d = np.log(2.) / 10.

T = 2e-4 #cm - Endothelial Thickness

a = 35.

#*************** Concentration Functions ****************
# Define functions

v = TestFunction(Q)
c_trial = TrialFunction(Q)
c = Function(Q, name="NO Concentration")

c_prev = Function(Q)


#********************** Velocities **********************

velocity = Function(V)

velocity_initial = Function(V)

velocity_final = Function(V)

#************************* WSS **************************
#WSS 

wss = Function(V)

wss_initial = Function(V)

wss_final = Function(V)

#Concentration functions

cdot = Function(Q)
cdot_prev = Function(Q)
dc = Function(Q)

flux = TrialFunction(Q)

phi = TestFunction(Q)

flux_sol = Function(Q, name="NO Flux")

#********************************************************
#****************** Boundary Conditions *****************
#********************************************************

#***************** BCs for O2 Transport *****************

bc1 = DirichletBC(Q, 0.0, facet_domains, 1) # Inlet: 0 drichlet

bc2 = DirichletBC(Q, 0.0, facet_domains, 2) # Wall: flux

bc3 = DirichletBC(Q, 0.0, facet_domains, 3) # Outlet1: 0 dirichlet

bc4 = DirichletBC(Q, 0.0, facet_domains, 4) # Outlet2: 0 dirichlet

bc5 = DirichletBC(Q, 0.0, facet_domains, 5) # Outlet2: 0 dirichlet

bc6 = DirichletBC(Q, 0.0, facet_domains, 6) # Outlet2: 0 dirichlet

#****************** BCs for dc Equation *****************
# For non-zero Dirichlet BC on transport equation, the BC for dc equation should be ZERO
# For Neumann BC on transport equation, the BC for dc equation should NOT be specified

bc_dc1 = DirichletBC(Q, 0.0, facet_domains, 1) # Inlet: 0 drichlet

bc_dc2 = DirichletBC(Q, 0.0, facet_domains, 2) # Wall: flux

bc_dc3 = DirichletBC(Q, 0.0, facet_domains, 3) # Outlet1: 0 dirichlet

bc_dc4 = DirichletBC(Q, 0.0, facet_domains, 4) # Outlet2: 0 dirichlet

bc_dc5 = DirichletBC(Q, 0.0, facet_domains, 5) # Outlet2: 0 dirichlet

bc_dc6 = DirichletBC(Q, 0.0, facet_domains, 6) # Outlet2: 0 dirichlet


if aneurysm_bc_type == 'neumann':
	bcs = [bc1, bc3, bc4, bc5, bc6] 
	bcs_dc = [bc_dc1, bc_dc3, bc_dc4, bc_dc5, bc_dc6] 

elif aneurysm_bc_type == 'dirichlet':
	bcs = [bc1, bc2]
	bcs_dc = [bc_dc1, bc_dc2]

else:
	print 'Need to pick Neumann or Dirichlet BC for aneurysm'



if aneurysm_bc_type == 'neumann':
	ds = Measure("ds")(subdomain_data=facet_domains)

#Generalized alpha
#***************** Generalized alpha  ******************
#Generalized alpha
m = v * cdot * dx

M = assemble(v * c_trial * dx)
rho_inf = 0.0 #.2
alpha_m = .5 * (3. - rho_inf) / (1. + rho_inf)
alpha_f = 1. / (1. + rho_inf)
gamma = .5 + alpha_m - alpha_f


#********************************************************
#****************** Solution Procedure  *****************
#********************************************************

tstep = 0

# Set initial condition
if (IC_flag == True):
 IC_in =  HDF5File(mpi_comm_world(), IC_file, 'r')
 IC_in.read(c_prev, '/concentration/vector_0')
 IC_in.close()
 tstep = IC_start_index
 t_start = IC_start_index * dt
 if MPI.rank(mpi_comm_world()) == 0:
	print 'Using I.C.'
else:
 c_prev.assign(interpolate(Constant(0.0), Q))

cdot_prev.assign(interpolate(Constant(0.0), Q))

t = t_start

file_initial = velocity_start
file_final = file_initial + velocity_interval

wss_file_initial = wss_start
wss_file_final = wss_file_initial + wss_interval


if file_final > velocity_stop:
	file_final = velocity_start


#out = HDF5File(mpi_comm_world(),results_dir + 'concentration.h5','w')
#out_velocity = HDF5File(mpi_comm_world(),results_dir + 'velocity.h5','w')

#out_max_vals = open(results_dir + 'max_vals.dat', 'w')

#prm = parameters["krylov_solver"]
#prm["nonzero_initial_guess"] = True


# Time-stepping

start_time = time.clock()
#problem = LinearVariationalProblem(a, L, c_sol, bcs)

#We set solver inside the for loop here
#solver = LinearVariationalSolver(problem)
#solver.parameters["linear_solver"] ="bicgstab"
#solver.parameters["preconditioner"] ="bjacobi"
#solver.parameters['krylov_solver']['nonzero_initial_guess'] = True


prm = parameters['krylov_solver'] # short form

#Uncomment below of ill-conditioning issues arise
#prm['absolute_tolerance'] = 1e-17
#prm['relative_tolerance'] = 1e-10
file_conc_output = File(results_dir + 'NO_Concentration.pvd')
#velocity_test_output = File('/scratch/af2289/aneurysm_test/task1/' + 'velTest_primaryCode/test_velocity.pvd') 
while tstep < n_tsteps:

	velocity_initial = Function(VectorFunctionSpace(mesh, 'CG', 1),velocity_filename +str(file_initial)+'.xml')
	velocity_final = Function(VectorFunctionSpace(mesh, 'CG', 1),velocity_filename +str(file_final)+'.xml')

	wss_initial = Function(VectorFunctionSpace(mesh, 'CG', 1),wss_filename +str(wss_file_initial)+'.xml')
	wss_final = Function(VectorFunctionSpace(mesh, 'CG', 1),wss_filename +str(wss_file_final)+'.xml')


	for tstep_inter in xrange(tsteps_per_file):
		tstep += 1
		if tstep > n_tsteps:
			break
		t += dt



		alpha = (tstep_inter * 1.0) / (tsteps_per_file * 1.0)

		velocity.vector().set_local(  (1. - alpha) * velocity_initial.vector().get_local()
							  + alpha * velocity_final.vector().get_local())
		velocity.vector().apply('')

		wss.vector().set_local(  (1. - alpha) * wss_initial.vector().get_local()
							  + alpha * wss_final.vector().get_local())
		wss.vector().apply('')

		#velocity_test_output << velocity 
	
		mag_wss = abs(sqrt(inner(wss , wss)))
		tau_ratio = abs( abs(mag_wss) / abs((mag_wss + a)) )
		R_NO = R_b_space + R_max * abs( tau_ratio )

		# Initialize c and cdot
		c.vector().set_local((1. - alpha_f) * c_prev.vector().get_local()+ alpha_f * c_prev.vector().get_local())
		c.vector().apply('')
		cdot.vector().set_local((1. - alpha_m) * cdot_prev.vector().get_local() + alpha_m * (gamma - 1.) / gamma * cdot_prev.vector().get_local())
		cdot.vector().apply('')


		n = - v * dot(velocity , grad(c_trial) ) * dx \
			- D_multiple * dot(grad(v), grad(c_trial)) * dx \
				- K_d * c_trial * v * dx

		if aneurysm_bc_type == 'neumann':
			rhs = T * (R_NO) * v * ds(2)

		if stabilized: #!! c_prev in res_n should go to rhs below
			res_n = dot(velocity , grad(c_trial) )  - div(D_multiple * grad(c_trial)) + K_d * c_trial  #Maybe has to be advective form like the weak form 
			tau_m = (4. / dt**2 \
				 + dot(velocity, velocity) / h**2 \
				 + 9. * basis_order**4 * D_multiple**2 / h**4)**(-.5)
			n -= tau_m * res_n * dot(grad(v),velocity) * dx 
			m1 =  v * c_trial * dx  +  tau_m * c_trial* dot(grad(v), velocity) * dx # the cdot term in stablization should go here 
			M = assemble(m1)

		N = assemble(n)
		N_rhs = assemble(rhs)

		iteration = 0
		err = 1. + relative_tolerance
		while err > relative_tolerance:
			iteration += 1
			K = alpha_m / gamma / dt / alpha_f * M - N
			G = -1. * M * cdot.vector() + N * c.vector() + N_rhs    #assemble(N) # Actually -G from Jansen
			for bc in bcs_dc:
				bc.apply(K, G)
			solve(K, dc.vector(), G, 'gmres', 'default') #prec:'bjacobi'
			c.vector().set_local(c.vector().get_local() + dc.vector().get_local())
			c.vector().apply('')
			cdot.vector().set_local(
				(1. - alpha_m / gamma) * cdot_prev.vector().get_local()
				+ alpha_m / gamma / dt / alpha_f
				  * (c.vector().get_local() - c_prev.vector().get_local()))
			cdot.vector().apply('')
			err = norm(dc.vector(), 'linf') / max(norm(c.vector(), 'linf'), DOLFIN_EPS)

			if MPI.rank(mpi_comm_world()) == 0:
				print 'Iteration:', iteration, 'Error:', err

			for bc in bcs:
				bc.apply(c.vector())


		c.vector().set_local(
				(1. - 1. / alpha_f) * c_prev.vector().get_local()
				+ 1. / alpha_f * c.vector().get_local())
		c.vector().apply('')

		cdot.vector().set_local(
				(1. - 1. / alpha_m) * cdot_prev.vector().get_local()
				+ 1. / alpha_m * cdot.vector().get_local())
		cdot.vector().apply('')
		c_prev.vector().set_local(c.vector().get_local())
		c_prev.vector().apply('')
		cdot_prev.vector().set_local(cdot.vector().get_local())
		cdot_prev.vector().apply('')


		# Plot and/or save solution

		if tstep % max_save_interval == 0:
			if aneurysm_bc_type == 'dirichlet':

				flux_form = -D * inner(grad(c), n)

				LHS_flux = assemble(flux * phi * ds, keep_diagonal=True)

				RHS_flux = assemble(flux_form * phi * ds)

				LHS_flux.ident_zeros()

				solve(LHS_flux, flux_sol.vector(), RHS_flux)

				max_val = MPI.max(mpi_comm_world(),
								  np.amax(flux_sol.vector().get_local()))

				min_val = MPI.min(mpi_comm_world(),
								  np.amin(flux_sol.vector().get_local()))

				if MPI.rank(mpi_comm_world()) == 0:

					print 'time =', t, 'of', t_stop, '...', \
						  'Max=', max_val, \
						  'Min=', min_val, \
						  'Time elapsed:', (time.clock() - start_time) / 3600., \
						  'Time remaining:', 1. / 3600. * (n_tsteps-tstep) \
										 * (time.clock()-start_time) / tstep


			else:

				max_val =   MPI.max(mpi_comm_world(),np.amax(c.vector().get_local()))
				min_val = MPI.min(mpi_comm_world(), np.amin(c.vector().get_local()))


				if MPI.rank(mpi_comm_world()) == 0:
					print 'time =', t, 'of', t_stop, '...', \
						  'Max=', max_val, \
						  'Min=', min_val, \
						  'Time elapsed:', (time.clock() - start_time) / 3600., \
						  'Time remaining:', 1. / 3600. * (n_tsteps-tstep) \
										 * (time.clock()-start_time) / tstep


		if ( tstep % solution_save_interval == 0  ) :
			file_conc_output << c

			#out = HDF5File(mpi_comm_world(),results_dir + 'concentration2nd_D5_' + str(tstep) + '.h5' ,'w')
			#out.write(c,'concentration',tstep)
			#out.close()

			if aneurysm_bc_type == 'dirichlet':
				flux_form = -D * inner(grad(c), n_normal )
				LHS_flux = assemble(flux * phi * ds, keep_diagonal=True)
				RHS_flux = assemble(flux_form * phi * ds)
				LHS_flux.ident_zeros()
				solve(LHS_flux, flux_sol.vector(), RHS_flux)
				out_flux.write(flux_sol,'flux_sol', tstep)

		if tstep % solution_save_last_freq  == 0 and \
				(tstep >= solution_save_last) :
			file_conc_output << c

			#out_velocity = HDF5File(mpi_comm_world(),results_dir + 'velocity_'+ str(tstep) + '.h5','w')
			#out_velocity.write(velocity,'velocity',tstep)
			#out_velocity.close()
			#out = HDF5File(mpi_comm_world(),results_dir + 'concentration2nd_D5_' + str(tstep) + '.h5' ,'w')
			#out.write(c,'concentration',tstep)
			#out.close()

			if aneurysm_bc_type == 'dirichlet':
				flux_form = -D * inner(grad(c), n_normal)
				LHS_flux = assemble(flux * phi * ds, keep_diagonal=True)
				RHS_flux = assemble(flux_form * phi * ds)
				LHS_flux.ident_zeros()
				solve(LHS_flux, flux_sol.vector(), RHS_flux)
				out_flux.write(flux_sol,'flux_sol', tstep)


	file_initial = file_final
	file_final = file_initial + velocity_interval

	wss_file_initial = wss_file_final
	wss_file_final = wss_file_initial + wss_interval

	if file_final > velocity_stop:
		file_final = velocity_start

	if wss_file_final > wss_stop:
		wss_file_final = wss_start

