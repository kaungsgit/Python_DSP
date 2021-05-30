Scalable Parametric Sweep Framework By Kaung Myat San Oo (kaung.sanoo@analog.com)

generic_sweep.py: Main script for running the parametric sweep code.
- Run this file to perform parametric sweep with the specified setup file.
setup_files: Contains sweep setup files (_swp_setup.py), where parameters to be swept are defined.
parameter_classes.py: Parameter classes with set_param methods are located here. Add more parameter classes to suit your eval needs.
- This is the file that you and your team will contribute towards to create a collection of parameter classes that everyone can use to perform their sweeps.
datalogs: Output .csv data files will appear under here.