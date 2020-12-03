# main file
""" this exercise proves that variable a in scriptA behaves like a true global variable
that retains its value (updated by other scripts)"""

import scriptA_global_var as swp_gbl
import scriptB_write as param_classes
import scriptD_read

print(swp_gbl.a)

import swp_setup_files.scriptE_write as swp_setup

print(swp_gbl.a)

param_classes.change_value_of_a()

print(swp_gbl.a)

scriptD_read.print_scriptA_value()

print(swp_gbl.a)

param_classes.dummy_func()
