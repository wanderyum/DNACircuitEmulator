import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import leastsq

class reaction():
    def __init__(self, plot_time_unit='minute', plot_conc_unit='uM'):
        self.equ = []
        self.rate_cons = {}
        self.reactant = {}
        self.rate = []              # 用于存储反应速率序列
        self.results = []
        self.factor = 1
        self.def_conc_factors()
        self.def_time_factors()
        self.plot_time_unit = plot_time_unit
        self.plot_conc_unit = plot_conc_unit

    def def_conc_factors(self):
        self.conc_factors = {}
        self.conc_factors['M'] = 1
        self.conc_factors['mM'] = 1e-3
        self.conc_factors['uM'] = 1e-6
        self.conc_factors['nM'] = 1e-9

    def def_time_factors(self):
        self.time_factors = {}
        self.time_factors['second'] = 1
        self.time_factors['minute'] = 60
        self.time_factors['hour'] = 60 * 60

    def set_plot_time_unit(self, s):
        self.plot_time_unit = s

    def add_equation(self, equ, rate_constant):
        self.equ.append([equ, rate_constant])

    def set_rate_constant(self, rate_constant, value):
        self.rate_cons[rate_constant] = value

    def take_reactions(self, ini, steps, factor=1):
        self.factor = factor
        self.count_reactants()
        self.set_ini(ini, steps)
        for i in range(steps-1):
            self.reaction_init(i)
            self.take_one_step_reactions(i, factor=factor)

    def count_reactants(self):
        '''
        自动推导所需的反应物并初始化反应物变化序列，以及反应速率序列。
        '''
        l = []
        self.rate = []
        for item in self.equ:
            tmp = item[0].split('->')
            for i in tmp:
                l += i.split('+')
            self.rate.append(None)
            if item[1] not in self.rate_cons:
                self.rate_cons[item[1]] = None
        s = set(l)
        for item in s:
            self.reactant[item] = None

    def show_init_gen(self):
        print('def ')
        for item in self.reactant:
            print('ini[\'{}\'] = 0'.format(item))

    def gen_ini_func(self):
        with open('gen_ini.py', 'w') as f:
            print('def gen_ini(D):\n    ini = {}', file=f)
            for key in self.reactant:
                print('    ini[\'{}\'] = 0'.format(key), file=f)
            print('    for key in D:\n        ini[key] = D[key]\n    return ini', file=f)

    def init(self, D):
        '''
        用于产生一条初始浓度设置。除用户指定物质浓度外, 其它都置0。
        输入:
        D: 字典, 用户指定浓度。
        输出:
        ini: 字典, 包含所有物质浓度的一条初始浓度设置。
        '''
        
        if len(self.reactant) == 0:
            self.count_reactants()
        ini = {}
        for key in self.reactant:
            ini[key] = 0
        for key in D:
            ini[key] = D[key]
        return ini

    def set_ini(self, ini, steps):
        for item in self.reactant:
            self.reactant[item] = np.linspace(0, 0, steps)
        for item in ini:
            self.reactant[item][0] = ini[item]
        for i in range(len(self.rate)):
            self.rate[i] = np.linspace(0, 0, steps-1)

    def reaction_init(self, present_step):
        for item in self.reactant:
            self.reactant[item][present_step+1] = self.reactant[item][present_step]

    def take_one_step_reactions(self, present_step, factor):
        '''
        用以模拟一个时间点上的反应。反应结果直接在self.reactant(反应物变化)和self.rate(反应速率变化)中记录。
        输入:
        present_step:   整数, 当前时间点
        factor:         正数, 反应速率常数的系数, 例如相邻时间点间隔1分钟则factor为60
        返回:
        无
        '''
        k = 0
        for eq in self.equ:
            tmp = eq[0].split('->')
            reactants = tmp[0].split('+')
            products = tmp[1].split('+')
            cons = self.rate_cons[eq[1]]
            rate = cons * factor
            for item in reactants:
                rate *= self.reactant[item][present_step]
            self.rate[k][present_step] = rate
            for item in reactants:
                self.reactant[item][present_step+1] = self.reactant[item][present_step+1]-rate
            for item in products:
                self.reactant[item][present_step+1] = self.reactant[item][present_step+1]+rate
            k += 1

    def react_and_save_results(self, ini, targets, steps, factor=1):
        self.results = []
        for item in ini:
            self.take_reactions(item, steps=steps, factor=factor)
            result_tmp = self.reactant[targets[0]]
            for tar in targets[1:]:
                result_tmp += self.reactant[tar]
            self.results.append(result_tmp)
        return self.results

    def fit_result(self, inis, p0, targets, y):
        '''
        inis:   列表, 里面是初始条件，每个初始条件为一个字典。
        p0:     字典, 猜测的参数的初始值。
        targets:列表, 里面是代表要测量量的字符串。
        y:      ndarray, 实际荧光数值
        '''
        self.count_reactants()
        self.data_length = y.shape[0]
        p0 = self.dic2list(p0)
        self.targets = targets
        self.inis = inis
        plsq = leastsq(self.error_function, x0=p0, args=y)[0]
        return plsq

    def dic2list(self, D):
        res = []
        for item in sorted(D.keys()):
            res.append(D[item])
        return res

    def error_function(self, p, data):
        tmp = sorted(self.rate_cons.keys())
        for i in range(len(tmp)):
            self.rate_cons[tmp[i]] = p[i]
        res = np.array(self.react_and_save_results(ini=self.inis, targets=self.targets, steps=self.data_length, factor=1))
        mat_tmp = np.square(data.T-res)
        ret = np.sum(mat_tmp, axis=0)
        return ret

    def display_settings(self):
        print('Reactions:')
        for item in self.equ:
            print('{:^25}\tconstant:{}'.format(item[0], item[1]))

    def plot_datum(self, target, figsize=(16, 9)):
        plt.figure(figsize=figsize)
        l = self.reactant[target].shape[0]
        x = np.linspace(0, l-1, l)
        plt.plot(x, self.reactant[target])
        plt.show()

    def plot_data(self, target_arr, figsize=(16, 9), dpi=150, labels=None, title=None, target=None):
        plt.figure(figsize=figsize, dpi=dpi)
        #print(target_arr)
        l = target_arr[0].shape[0]
        x = np.linspace(0, l-1, l) * self.factor / self.time_factors[self.plot_time_unit]
        if labels:
            for i in range(len(target_arr)):
                plt.plot(x, target_arr[i]/self.conc_factors[self.plot_conc_unit], label=labels[i])
        else:
            for datum in target_arr:
                plt.plot(x, datum/self.conc_factors[self.plot_conc_unit])
        if title:
            plt.title(title)

        if target:
            tmp = '['
            if type(target) == type([1]):
                tmp += target[0]
                for item in target[1:]:
                    tmp = tmp + '+' + item
            else:
                tmp += target
            tmp += ']'
            plt.ylabel('Concentration of {} ({})'.format(tmp, self.plot_conc_unit))
            #plt.ylabel('Concentration of [F1]+[F1:F2\':S] ($\mu$M)')
        else:
            plt.ylabel('Concentration ({})'.format(self.plot_conc_unit))
        plt.xlabel('Time: ({})'.format(self.plot_time_unit))
        plt.legend()
        plt.tight_layout()
        plt.show()

if __name__ == '__main__':
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

    r.count_reactants()
    r.set_rate_constant('k0', 2.24e-9)
    r.set_rate_constant('k1', 9.25e3)
    r.set_rate_constant('k2', 4.04e2)
