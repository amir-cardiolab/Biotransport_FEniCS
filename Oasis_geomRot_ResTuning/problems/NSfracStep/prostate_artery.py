__author__ = "Mikael Mortensen <mikaem@math.uio.no>"
__date__ = "2013-06-25"
__copyright__ = "Copyright (C) 2013 " + __author__
__license__ = "GNU Lesser GPL version 3 or any later version"

from ..NSfracStep import *
import numpy
from numpy import cos, pi, cosh
from os import getcwd
import pickle

import prostate_artery_BC
#import prostate_artery_SS_BC

#What works (backflow): Order=2, beta=2. , dt =0.001/2
#Order=1, beta=100. (50.) dt =0.001
#Order=1, beta=1.  dt =0.001053 /2.,

### AAA P18
restart_folder = False
# Create a mesh
def mesh(**params):
    #m = Mesh('/scratch/aa3878/18_oasis/P18vel_mesh.xml')
    m = Mesh('/scratch/mm4238/prostate_CFD/org_geom/prostate_arteries.xml')
    return m


#def problem_parameters(commandline_kwargs, NS_parameters, **NS_namespace):
#if "restart_folder" in commandline_kwargs.keys():
if restart_folder:
        restart_folder = commandline_kwargs["restart_folder"]
        restart_folder = path.join(getcwd(), restart_folder)
        f = open(path.join(restart_folder, 'params.dat'), 'r')
        NS_parameters.update(pickle.load(f))
        NS_parameters['T'] = NS_parameters['T'] + 10 * NS_parameters['dt']
        NS_parameters['restart_folder'] = restart_folder
        globals().update(NS_parameters)

else:
        # Override some problem specific parameters
        NS_parameters.update(
            solver="IPCS_ABCN_Res_prostate_artery",
            nu=0.004/(1.06/1000),
            T=  1.0*3,
            dt= 1.0 /(4000. * 10.) , # 0.95 /(2000.*5.)
            folder= '/scratch/mm4238/prostate_CFD/prostate_IC/Results',
            velocity_degree=1,
            save_step= 1 * 1, #40 * 5
            save_start= 1,
            checkpoint=100*1e9,
            print_intermediate_info=10,
            nonNewtonian_flag = False,
            backflow_flag = True, #!!!! need to mannualy specify the outlet faces in IPCS_ABCN.py (currently faces 3 and 4 assumed outlet)
            beta_backflow = 0.5 , # btwn 0 and 1 (actually beta=2.0 and higher works!)
            Resistance_flag = False, #make sure to specify outlet faces in IPCS_ABCN pressure_solve
            flag_H5 = True, #current MPI does not support h5 file
            Res1 = 540.1923, #resistance at the first outlet
            Res2 = 2242.4417, #resistance at the second outlet
            Res3 = 1324.0297, #resistance at the third outlet
            Res4 = 446.4148, #resistance at the fourth outlet
            Res5 = 8529.6729, #resistance at the fourth outlet
            Res6 = 803.7197, #resistance at the fourth outlet
            Res7 = 10910.0123, #resistance at the fourth outlet
            Res8 = 4164.3648, #resistance at the fourth outlet
            Res9 = 8264.6489, #resistance at the fourth outlet
            Res10 = 1007.8841, #resistance at the first outlet
            Res11 = 8892.4914, #resistance at the second outlet
            Res12 = 6580.1766, #resistance at the third outlet
            Res13 = 9048.0155, #resistance at the fourth outlet
            Res14 = 373.9059, #resistance at the fourth outlet
            Res15 = 2282.3115, #resistance at the fourth outlet
            Res16 = 10922.4809, #resistance at the fourth outlet
            Res17 = 3311.2918, #resistance at the fourth outlet
            Res18 = 4884.0144, #resistance at the fourth outlet
            Res19 = 6229.6638, #resistance at the first outlet
            Res20 = 5162.8159, #resistance at the second outlet
            Res21 = 0.0, #resistance at the third outlet
            Res22 = 2239.8622, #resistance at the fourth outlet
            Res23 = 5256.4919, #resistance at the fourth outlet
            Res24 = 344.9889, #resistance at the fourth outlet
            Res25 = 629.4818, #resistance at the fourth outlet
            Res26 = 1280.2442, #resistance at the fourth outlet
            Res27 = 8263.5831, #resistance at the fourth outlet
            Res28 = 850.3659, #resistance at the first outlet
            Res29 = 2235.8731, #resistance at the second outlet
            Res30 = 5339.6312, #resistance at the third outlet
            Res31 = 2165.2183, #resistance at the fourth outlet
            inlet_rotation = False,
            IC_lap = True, # True if Laplacian initialization is needed
            V_ave = 8.345204543, # average velocity/flowrate for initialization
            inlet_flag_flow = True, #if True, flowrate is input else avg velocity
            flag_wss = True,
            flag_ramp = False, #for the first cycle only start from 0 flow rate at the inlet
            initial_time_ramp = -0.019, # should be a negative value. Same value entered at the first time entry in the BC file with a corresponding zero flow.
            use_krylov_solvers=True)
        NS_parameters['krylov_solvers']['monitor_convergence'] = True
        #set_log_level(ERROR)
        globals().update(NS_parameters)

# Changed by Mostafa: V_inletBC_x,V_inletBC_y,V_inletBC_z
def create_bcs(t,t_cycle,Time_array,v_IR_array, V, Q, facet_domains, mesh, **NS_namespace):
    # Specify boundary conditions
    
    #bc_file =  '/scratch/aa3878/18_oasis/BCnodeFacets.xml' #specify it in pre_solve_hook function below too.
    #bc_file =  '/home/mm4238/vul_plaque_project/biochem_tr_Budof/1A_2012_LAD/BCnodeFacets.xml'
    #facet_domains = MeshFunction('size_t', mesh,bc_file )
    parabolic_flag = True
    inlet_flag_flow = True #if True flowrate is input, else avg velocity
    inlet_facet_ID = 1
    v_inlet_BC = numpy.interp(t_cycle, Time_array, v_IR_array)
    if (parabolic_flag):
      #Rotating the mesh so that the average normal vector aligned with z-axis passing the centroid
      d = mesh.geometry().dim()
      x = SpatialCoordinate(mesh)
      ds = Measure("ds")(subdomain_data=facet_domains)
      dsi = ds(inlet_facet_ID, domain=mesh, subdomain_data=facet_domains)
      # Compute area of boundary tesselation by integrating 1.0 over all facets
      A = assemble(Constant(1.0, name="one")*dsi)
      if MPI.rank(mpi_comm_world()) == 0:
              print 'Inlet area:', A
      if (inlet_flag_flow):
          v_inlet_BC = v_inlet_BC*1e3 / A
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
          print 'Normal vector:', normal

      bc_u2 = Expression("n0 * u * (1 - (pow(cent0-x[0], 2) + pow(cent1-x[1], 2) ) / r)",n0=normal[2], r=inlet_radius2, u=-2.0*v_inlet_BC, cent0=center[0],cent1=center[1],cent2=center[2], degree=3)

    if (parabolic_flag):

        #V_inletBC_x = Function(V,'/scratch/mm4238/utilities/BC_xml/bctxml_x.xml')
        #V_inletBC_y = Function(V,'/scratch/mm4238/utilities/BC_xml/bctxml_y.xml')
        #V_inletBC_z = Function(V,'/scratch/mm4238/utilities/BC_xml/bctxml_z.xml')
        bc_in_1  = DirichletBC(V, 0. , facet_domains,inlet_facet_ID) #inlet for u_x
        bc_in_2  = DirichletBC(V, 0. , facet_domains,inlet_facet_ID) #inlet for u_y
        bc_in_3  = DirichletBC(V, bc_u2 , facet_domains,inlet_facet_ID) #inlet for u_z
        #bc_in_1  = DirichletBC(V, bc_u0 , facet_domains,inlet_facet_ID) #inlet for u_x
        #bc_in_2  = DirichletBC(V, bc_u1 , facet_domains,inlet_facet_ID) #inlet for u_y
        #bc_in_3  = DirichletBC(V, bc_u2 , facet_domains,inlet_facet_ID) #inlet for u_z
    else:
        inflow = Expression('-vel_IR',vel_IR=0,degree=3)
        inflow.vel_IR = v_inlet_BC
        bc_in_1  = DirichletBC(V, 0. , facet_domains,inlet_facet_ID) #inlet for u_x
        bc_in_2  = DirichletBC(V, 0. , facet_domains,inlet_facet_ID) #inlet for u_y
        bc_in_3  = DirichletBC(V,  inflow , facet_domains,inlet_facet_ID) #inlet for u_z
    bc_wall_1  = DirichletBC(V, 0. , facet_domains,2) #wall for u_x
    bc_wall_2  = DirichletBC(V, 0. , facet_domains,2) #wall for u_y
    bc_wall_3  = DirichletBC(V, 0. , facet_domains,2) #wall for u_z
    bcp1  = DirichletBC(Q, 0. , facet_domains,3) #outlet p
    bcp2  = DirichletBC(Q, 0. , facet_domains,4) #outlet p
    bcp3  = DirichletBC(Q, 0. , facet_domains,5) #outlet p
    bcp4  = DirichletBC(Q, 0. , facet_domains,6) #outlet p
    bcp5  = DirichletBC(Q, 0. , facet_domains,7) #outlet p
    bcp6  = DirichletBC(Q, 0. , facet_domains,8) #outlet p
    bcp7  = DirichletBC(Q, 0. , facet_domains,9) #outlet p
    bcp8  = DirichletBC(Q, 0. , facet_domains,10) #outlet p
    bcp9  = DirichletBC(Q, 0. , facet_domains,11) #outlet p
    bcp10  = DirichletBC(Q, 0. , facet_domains,12) #outlet p
    bcp11  = DirichletBC(Q, 0. , facet_domains,13) #outlet p
    bcp12  = DirichletBC(Q, 0. , facet_domains,14) #outlet p
    bcp13  = DirichletBC(Q, 0. , facet_domains,15) #outlet p
    bcp14  = DirichletBC(Q, 0. , facet_domains,16) #outlet p
    bcp15  = DirichletBC(Q, 0. , facet_domains,17) #outlet p
    bcp16  = DirichletBC(Q, 0. , facet_domains,18) #outlet p
    bcp17  = DirichletBC(Q, 0. , facet_domains,19) #outlet p
    bcp18  = DirichletBC(Q, 0. , facet_domains,20) #outlet p
    bcp19  = DirichletBC(Q, 0. , facet_domains,21) #outlet p
    bcp20  = DirichletBC(Q, 0. , facet_domains,22) #outlet p
    bcp21  = DirichletBC(Q, 0. , facet_domains,23) #outlet p
    bcp22  = DirichletBC(Q, 0. , facet_domains,24) #outlet p
    bcp23  = DirichletBC(Q, 0. , facet_domains,25) #outlet p
    bcp24  = DirichletBC(Q, 0. , facet_domains,26) #outlet p
    bcp25  = DirichletBC(Q, 0. , facet_domains,27) #outlet p
    bcp26  = DirichletBC(Q, 0. , facet_domains,28) #outlet p
    bcp27  = DirichletBC(Q, 0. , facet_domains,29) #outlet p
    bcp28  = DirichletBC(Q, 0. , facet_domains,30) #outlet p
    bcp29  = DirichletBC(Q, 0. , facet_domains,31) #outlet p
    bcp30  = DirichletBC(Q, 0. , facet_domains,32) #outlet p
    bcp31  = DirichletBC(Q, 0. , facet_domains,33) #outlet p
    if MPI.rank(mpi_comm_world()) == 0:
      print '-----t = ', t
      print '-----t_cycle = ', t_cycle
    return dict(u0=[bc_in_1, bc_wall_1],
                u1=[bc_in_2, bc_wall_2],
                u2=[bc_in_3, bc_wall_3],
                p=[bcp1, bcp2, bcp3, bcp4, bcp5, bcp6, bcp7, bcp8, bcp9, bcp10, bcp11, bcp12, bcp13, bcp14, bcp15, bcp16, bcp17, bcp18, bcp19, bcp20, bcp21, bcp22, bcp23, bcp24, bcp25, bcp26, bcp27, bcp28, bcp29, bcp30, bcp31])
    #return dict(u0=[bc_in_1, bc_wall_1],
    #           u1=[bc_in_2, bc_wall_2],
    #           u2=[bc_in_3, bc_wall_3],
    #           p =[]) # no pressure BC

def pre_solve_hook(mesh, facet_domains, velocity_degree, u_,
                   AssignedVectorFunction, **NS_namespace):
    #bc_file =   '/scratch/aa3878/18_oasis/BCnodeFacets.xml'
    #bc_file =  '/home/mm4238/vul_plaque_project/biochem_tr_Budof/1A_2012_LAD/BCnodeFacets.xml'
    #facet_domains = MeshFunction('size_t', mesh,bc_file )
    #normal
    n_normal = FacetNormal(mesh)
    
    #Time depndent inlet BC
    #read BC from IA05_BC.py
    Time_array = prostate_artery_BC.time_BC()
    v_IR_array = prostate_artery_BC.Vel_IR_BC()
    Time_last = Time_array[-1]
    
    return dict(uv=AssignedVectorFunction(u_),  facet_domains= facet_domains, n_normal=n_normal,Time_array=Time_array,v_IR_array=v_IR_array,Time_last= Time_last )


def temporal_hook(tstep, uv, p_, plot_interval, **NS_namespace):
    if(0):
     if tstep % plot_interval == 0:
        uv()
        plot(uv, title='Velocity')
        plot(p_, title='Pressure')


def theend_hook(p_, uv, **NS_namespace):
   if(0):
    uv()
    plot(uv, title='Velocity')
    plot(p_, title='Pressure')
