## About Oasis_geomRot_ResTuning Package

This package is a modified version of Oasis software package to solve Navier-Stokes equations in arbitrary geometries. 

It is developed in the Cardiovascular Biomechanics Lab, NAU, by Mostafa Mahmoudi under direction of Dr. Amirhossein Arzani.

The results from these codes are published in:

Mahmoudi, M., Farghadan, A., McConnell, D. R., Barker, A. J., Wentzel, J. J., Budoff, M. J., & Arzani, A. (2021). The story of wall shear stress in coronary artery atherosclerosis: biochemical transport and mechanotransduction. Journal of Biomechanical Engineering, 143(4).


## Installation

No need to compile the codes. All the codes are written in Python 2.7 and are ready to use.

### Prerequisites

The following software is required:

- FEniCS 2017
- Python 2.7

### Test Case

A 3D patient-specific model of left anterior descending artery (LAD) is provided. The problem test case in the "problem/NSfracStep" directory is named "vul_plaque_3A_2016_LAD_geomRotation.py". This file read the inflow boundary condition file, "vul_plaque_3A_2016_LAD_BC", as well as the NSfracStep file named "NSfracStep_rotation_Res_LAD.py" in the main directory. The solver file associated with this problem is under "solvers/NSfracStep" directory with the name "IPCS_ABCN_LAD.py". 

Google Drive link to the mesh files: https://drive.google.com/drive/folders/15ttuJ0blJZ5SiGVaHm7v686KjvwzGZf1?usp=sharing

### How to cite?

Please cite the main paper for which this package was developed:

Mahmoudi, M., Farghadan, A., McConnell, D. R., Barker, A. J., Wentzel, J. J., Budoff, M. J., & Arzani, A. (2021). The story of wall shear stress in coronary artery atherosclerosis: biochemical transport and mechanotransduction. Journal of Biomechanical Engineering, 143(4).
