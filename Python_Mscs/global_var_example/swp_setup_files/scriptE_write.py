import scriptA_global_var as swp_gbl

# import scriptA as swp_gbl
# import Python_Mscs.global_var_example.scriptA as swp_gbl
""" The two above import statements are importing the same file but since they're called differently,
scriptA will be imported TWICE! This is undesirable especially the scriptA is intended to be a global variable module.
"""

# scriptE changes a from scriptA
swp_gbl.a['val2'] = 'val2 added in scriptE'
