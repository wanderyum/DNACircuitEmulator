from reaction import reaction
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import os
import sys
os.chdir(os.path.dirname(sys.argv[0]))


r = reaction()

# equ. 1
r.add_equation('fs+e->f1f2s+e', 'k0')
r.add_equation('fps+e->f1pf2ps+e', 'k0')
# equ. 2
r.add_equation('f1f2s+i->if2s+f1', 'k1')
r.add_equation('f1pf2ps+i->if2ps+f1p', 'k1')
# equ. 3
r.add_equation('if2s+f->fs+i+f2', 'k2')
r.add_equation('if2ps+f->fs+i+f2p', 'k2')

# reverse reaction of equ. 2
r.add_equation('if2s+f1->f1f2s+i', 'kr')
r.add_equation('if2ps+f1p->f1pf2ps+i', 'kr')

# additional equ.
r.add_equation('f1p+if2s->f1pf2s+i', 'kr')
r.add_equation('f1+if2ps->f1f2ps+i', 'kr')

# the forward reaction of additional equ.
#r.add_equation('f1pf2s+i->f1p+if2s', 'k1')
#r.add_equation('f1f2ps+i->f1+if2ps', 'k1')

'''
r.set_rate_constant('k0', 2.24e-9)
r.set_rate_constant('k1', 9.25e3)
r.set_rate_constant('k2', 4.04e2)
r.set_rate_constant('kr', 0.01*9.25e3)
'''

ini = []
# 1
ini.append(r.init({'fps':0.4e-6, 'e': 2e5, 'i': 0.4e-6, 'f': 0.6e-6}))
# 2-4
ini.append(r.init({'fps':0.1e-6, 'e': 2e5, 'i': 0.4e-6, 'f': 0.6e-6}))
ini.append(r.init({'fps':0.2e-6, 'e': 2e5, 'i': 0.4e-6, 'f': 0.6e-6}))
ini.append(r.init({'fps':0.6e-6, 'e': 2e5, 'i': 0.4e-6, 'f': 0.6e-6}))
# 5-7
ini.append(r.init({'fps':0.4e-6, 'e': 2e5, 'i': 0.4e-6, 'f': 0.3e-6}))
ini.append(r.init({'fps':0.4e-6, 'e': 2e5, 'i': 0.4e-6, 'f': 1.2e-6}))
ini.append(r.init({'fps':0.4e-6, 'e': 2e5, 'i': 0.4e-6, 'f': 1.8e-6}))
# 8-10
ini.append(r.init({'fps':0.4e-6, 'e': 2e5, 'i': 0.1e-6, 'f': 0.6e-6}))
ini.append(r.init({'fps':0.4e-6, 'e': 2e5, 'i': 0.2e-6, 'f': 0.6e-6}))
ini.append(r.init({'fps':0.4e-6, 'e': 2e5, 'i': 0.8e-6, 'f': 0.6e-6}))
# 11-13
ini.append(r.init({'fps':0.4e-6, 'e': 1e5, 'i': 0.4e-6, 'f': 0.6e-6}))
ini.append(r.init({'fps':0.4e-6, 'e': 3e5, 'i': 0.4e-6, 'f': 0.6e-6}))
ini.append(r.init({'fps':0.4e-6, 'e': 4e5, 'i': 0.4e-6, 'f': 0.6e-6}))

df = pd.read_csv('feature_data.csv')
data = np.array(df.iloc[:])
#print(data.shape)
p0 = {'k0':1e-9*240, 'k1':1e4*240, 'k2':1e2*240, 'kr':1e2*240}
targets = ['f1', 'f1f2ps']

scale = 0.6e-6 / 3930

#plsq = r.fit_result(inis=ini, p0=p0, targets=targets, y=data*scale)
#print(plsq)

'''
查看结果
[2.86772147e-07 4.14935728e+06 8.13362535e+04 6.16434189e+03]
'''
r.set_rate_constant('k0', 2.86772147e-07/240)
r.set_rate_constant('k1', 4.14935728e+06/240)
r.set_rate_constant('k2', 8.13362535e+04/240)
r.set_rate_constant('kr', 6.16434189e+03/240)

r.react_and_save_results(ini, targets, 60*12, 60)


labels = list(df.columns)
print(labels)
tit = 'Example'
r.plot_data(r.results, labels=labels, title=tit, figsize=(8,6), dpi=200, target=targets)
