## About Biotransport Package

This package contains the codes to simulate the near-wall transport of NO, LDL, ATP, O2, Monocytes, and MCP-1 in coronary arteries.

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

A 3D patient-specific model of left anterior descending artery (LAD) is provided in the Test directory. All the codes will need blood flow velocities and WSS vectorial fields in vtu or h5 format. Under Test directory, all required files are included for the test case. To use h5 file format, the user need to comment out the section regarding reading vtu files and uncomment the section related to reading h5 files. Using h5 file format decreases the computation cost.

### How to cite?

Please cite the main paper for which this package was developed:

Mahmoudi, M., Farghadan, A., McConnell, D. R., Barker, A. J., Wentzel, J. J., Budoff, M. J., & Arzani, A. (2021). The story of wall shear stress in coronary artery atherosclerosis: biochemical transport and mechanotransduction. Journal of Biomechanical Engineering, 143(4).
