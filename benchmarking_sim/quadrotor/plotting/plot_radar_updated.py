import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns # improves plot aesthetics


def _invert(x, limits):
    """inverts a value x on a scale from
    limits[0] to limits[1]"""
    return limits[1] - (x - limits[0])

def _scale_data(data, ranges):
    """scales data[1:] to ranges[0],
    inverts if the scale is reversed"""
    for d, (y1, y2) in zip(data[1:], ranges[1:]):
        assert (y1 <= d <= y2) or (y2 <= d <= y1)
    x1, x2 = ranges[0]
    d = data[0]
    if x1 > x2:
        d = _invert(d, (x1, x2))
        x1, x2 = x2, x1
    sdata = [d]
    for d, (y1, y2) in zip(data[1:], ranges[1:]):
        if y1 > y2:
            d = _invert(d, (y1, y2))
            y1, y2 = y2, y1
        sdata.append((d-y1) / (y2-y1) 
                     * (x2 - x1) + x1)
    return sdata

class ComplexRadar():
    def __init__(self, fig, variables, ranges,
                 n_ordinate_levels=6):
        angles = np.arange(0, 360, 360./len(variables))

        axes = [fig.add_axes([0.1,0.1,0.9,0.9],polar=True,
                label = "axes{}".format(i)) 
                for i in range(len(variables))]
        l, text = axes[0].set_thetagrids(angles, 
                                         labels=variables)
        [txt.set_rotation(angle-90) for txt, angle 
             in zip(text, angles)]
        for ax in axes[1:]:
            ax.patch.set_visible(False)
            ax.grid("off")
            ax.xaxis.set_visible(False)
        for i, ax in enumerate(axes):
            grid = np.linspace(*ranges[i], 
                               num=n_ordinate_levels)
            gridlabel = ["{}".format(round(x,2)) 
                         for x in grid]
            if ranges[i][0] > ranges[i][1]:
                grid = grid[::-1] # hack to invert grid
                          # gridlabels aren't reversed
            gridlabel[-1] = "" # clean up origin
            gridlabel[0] = ""
            ax.set_rgrids(grid, labels=gridlabel,
                         angle=angles[i])
            #ax.spines["polar"].set_visible(False)
            ax.set_ylim(*ranges[i])
        # variables for plotting
        self.angle = np.deg2rad(np.r_[angles, angles[0]]) + np.pi/3
        self.ranges = ranges
        self.ax = axes[0]
    def plot(self, data, *args, **kw):
        sdata = _scale_data(data, self.ranges)
        self.ax.plot(self.angle, np.r_[sdata, sdata[0]], *args, **kw)
    def fill(self, data, *args, **kw):
        sdata = _scale_data(data, self.ranges)
        self.ax.fill(self.angle, np.r_[sdata, sdata[0]], *args, **kw)

# example data
variables = ("Generalization performance", "Performance", "Inference time", 
            "Model complexity", "Sampling complexity", "Robustness",)

data_gpmpc = (0.03876024, 0.05573254, 0.0090775150246974736, 3, int(660), int(70))
data_lmpc = (0.1424, 0.0768, 0.00024354409288477016, 2, int(1), int(0))
data_mpc = (0.1717, 0.0868, 0.0001976909460844817, 3, int(1), int(50))
data_ppo = (0.11634496692276783, 0.06775482275993325, 0.0011251235, 1, int(1.85*1e5), int(5))
data_sac = (0.2517, 0.0314, 0.00020738168999000832, 1, int(0.85*1e5), int(15))
data_td3 = (0.026798393095810013, 0.05096096290371684, 0.0061547613, 1, int(4e5), int(10))

data_list = [data_gpmpc, data_lmpc, data_mpc, data_ppo, data_sac, data_td3]
# get the max and min values of each variable
max_values = [np.max([data[i] for data in data_list]) for i in range(len(variables))]
min_values = [np.min([data[i] for data in data_list]) for i in range(len(variables))]

print('max_values:', max_values)
print('min_values:', min_values)
ranges = [(min_values[i], max_values[i]) for i in range(len(variables))]
# reverse the model complexity range
ranges[2] = (max_values[2], min_values[2])

data = data_gpmpc
# data = (1.76, 1.1, 1.2, 
#         4.4, 3.4, 86.8,)
# if the range is reversed then the scale will be inverted
# ranges = [(0.1, 2.3), (1.5, 0.3), (1.3, 0.5),
#          (1.7, 4.5), (1.5, 3.7), (70, 87),] 

# data_2 = (1.1, 1, 1.3,
#           3.2, 3., 85.7,)         
# plotting
fig1 = plt.figure(figsize=(7, 5),)
radar = ComplexRadar(fig1, variables, ranges)
radar.plot(data)
radar.fill(data, alpha=0.2)
# radar.plot(data_2)
# radar.fill(data_2, alpha=0.2)


fig1.savefig('radar_updated.png')