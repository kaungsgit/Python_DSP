Scalable Parametric Sweep Framework By Kaung Myat San Oo (kaung.sanoo@analog.com)

Main script for running the parametric sweep code is generic_sweep.py. 
- Run this file to perform parametric sweep with the specified setup file.
Sweep setup files, where what parameter and what values to be swept are defined, are located under setupFiles.
Parameter classes with set_param methods are in parameter_classes.py. Add more parameter classes to suit your eval needs.
.csv data files will appear under the folder datalogs.