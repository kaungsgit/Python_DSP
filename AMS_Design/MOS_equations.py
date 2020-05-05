import numpy as np
import sympy as sp
from engineering_notation import EngNumber as en

''' ************************************ calculate NMOS ID, VGS, or VDS ************************************'''
a = dict()
a['VGS_n_val'] = '?'
a['VDS_n_val'] = 2.5
a['ID_n_val'] = 20e-6

# Baker parameters
Kp_n = 120e-6
W_n = 10
L_n = 2
VTH_n = 0.8
lmda_n = 0.01

# # McNeill parameters
# Kp_n = 7e-6
# W_n = 350
# L_n = 10
# VTH_n = 1.5
# lmda_n = 0.03

ID_n, VGS_n, VDS_n = sp.symbols('ID_n, VGS_n, VDS_n')

h = Kp_n / 2 * W_n / L_n * (VGS_n - VTH_n) ** 2 * (1 + lmda_n * VDS_n) - ID_n

x = [(ID_n, a['ID_n_val']), (VGS_n, a['VGS_n_val']), (VDS_n, a['VDS_n_val'])]

for i in x:
    sym, val = i
    if val == '?':
        x.remove(i)
        calculated_para = i

output_val_n = float(max(sp.solve(h.subs(x))))

print(f'{calculated_para[0]} is {en(output_val_n)}')

for k, v in a.items():
    if v == '?':
        a[k] = output_val_n

Rout_n = 1 / (lmda_n * a['ID_n_val'])
print(f'Rout_n is {en(Rout_n)}')
gm_n = 2 * a['ID_n_val'] / (a['VGS_n_val'] - VTH_n)
print(f'gm is {en(gm_n)}\n')

''' ************************************ calculate PMOS ID, VGS, or VDS ************************************'''
a = dict()
a['VGS_p_val'] = '?'
a['VDS_p_val'] = 2.5
a['ID_p_val'] = 20e-6

# Baker parameters
Kp_p = 40e-6
W_p = 30
L_p = 2
VTH_p = -0.9
lmda_p = 0.0125

# # McNeill parameters
# Kp_p = 2.5e-6
# W_p = 900
# L_p = 10
# VTH_p = -1.5
# lmda_p = 0.05

ID_p, VGS_p, VDS_p = sp.symbols('ID_p, VGS_p, VDS_p')

h = Kp_p / 2 * W_p / L_p * (VGS_p - VTH_p) ** 2 * (1 + lmda_p * VDS_p) - ID_p

x = [(ID_p, a['ID_p_val']), (VGS_p, a['VGS_p_val']), (VDS_p, a['VDS_p_val'])]

for i in x:
    sym, val = i
    if val == '?':
        x.remove(i)
        calculated_para = i

output_val_p = float(min(sp.solve(h.subs(x))))

print(f'{calculated_para[0]} is {en(output_val_p)}')

for k, v in a.items():
    if v == '?':
        a[k] = output_val_p

Rout_p = 1 / (lmda_p * a['ID_p_val'])
print(f'Rout_p is {en(Rout_p)}')
gm_p = 2 * a['ID_p_val'] / abs((a['VGS_p_val'] - VTH_p))
print(f'gm is {en(gm_p)}\n')
