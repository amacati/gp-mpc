#!/usr/bin/python

"""
Installation: Get the *Miniconda* Python Distribution - not the Python distrubution from python.org!
- https://conda.io/miniconda.html
Then install modules:
- `cd ~/miniconda3/bin`
- `./conda install numpy pandas matplotlib`
Original source:
- https://stackoverflow.com/questions/24659005/radar-chart-with-multiple-scales-on-multiple-axes
- That code has problems with 5+ axes though
"""

import numpy as np
import matplotlib.pyplot as plt

# Optionally use different styles for the graph
# Gallery: http://tonysyu.github.io/raw_content/matplotlib-style-gallery/gallery.html
# import matplotlib
# matplotlib.style.use('dark_background')  # interesting: 'bmh' / 'ggplot' / 'dark_background'


class Radar(object):
    def __init__(self, figure, title, labels, rect=None):
        if rect is None:
            rect = [0.05, 0.05, 0.9, 0.9]

        self.n = len(title)
        self.angles = np.arange(0, 360, 360.0/self.n)

        self.axes = [figure.add_axes(rect, projection='polar', label='axes%d' % i) for i in range(self.n)]

        self.ax = self.axes[0]
        self.ax.set_thetagrids(self.angles, labels=title, fontsize=14)

        for ax in self.axes[1:]:
            ax.patch.set_visible(False)
            ax.grid(False)
            ax.xaxis.set_visible(False)

        # performance
        # self.axes[0].set_rgrids([0.001, 0.01, 0.1], angle=0, labels=labels[0])
        self.axes[0].set_rgrids([0.001, 0.01, 0.1], angle=0,)
        self.axes[0].set_yscale('symlog', linthresh=0.01)
        self.axes[0].spines['polar'].set_visible(False)
        # self.axes[0].set_ylim(0, 0.1)
        
        # generalization performance
        # self.axes[1].set_rgrids([0.001, 0.01, 0.1], angle=72, labels=labels[1])
        self.axes[1].set_yscale('symlog', linthresh=0.01)
        self.axes[1].set_rgrids([0.001, 0.01, 0.1], angle=72,)
        self.axes[1].spines['polar'].set_visible(False)
        # self.axes[1].set_ylim(0, 0.1)
        # max inference time
        self.axes[2].set_rgrids([0.00001, 0.005, 0.01], angle=144, labels=labels[2])
        self.axes[2].set_yscale('symlog', linthresh=0.01)
        self.axes[2].spines['polar'].set_visible(False)
        # self.axes[2].set_ylim(0, 0.02)
        # model complexity
        self.axes[3].set_rgrids([0, 1, 2, 3], angle=216, labels=labels[3])
        self.axes[3].spines['polar'].set_visible(False)
        # self.axes[3].set_ylim(0, 3)
        # sampling complexity
        self.axes[4].set_rgrids([0, 20, 1000], angle=288, labels=labels[4])
        self.axes[4].set_yscale('symlog', linthresh=0.01)
        self.axes[4].spines['polar'].set_visible(False)
        # self.axes[4].set_ylim(1, 1e5)

        # for ax, angle, label in zip(self.axes, self.angles, labels):
        #     ax.set_yscale('symlog', linthresh=0.01)
        #     ax.set_rgrids(np.linspace(0.0001, 0.05, 5), angle=angle, labels=label)
        #     ax.spines['polar'].set_visible(False)
        #     ax.set_ylim(0.0001, 0.05)       

    def plot(self, values, *args, **kw):
        angle = np.deg2rad(np.r_[self.angles, self.angles[0]])
        values = np.r_[values, values[0]]
        self.ax.plot(angle, values, *args, **kw)


if __name__ == '__main__':
    fig = plt.figure(figsize=(10, 8))

    # tit = list('ABCDEFGHIJKJ')  # 12x
    tit = ['Performance','Generalization performance','Inference time',
              'Model complexity', 'Sampling complexity']

    lab = [
        # list('abcde'),
        # list('12345'),
        # list('uvwxy'),
        # ['one', 'two', 'three', 'four', 'five'],
        # list('jklmn'),
        list('abc'),
        list('123'),
        list('uvw'),
        ['', 'Model-free', 'Linear Model', 'Nonlinear Model',],
        list('jkl'),
    ]

    gpmpc_res = [
        3.94646538e-02, 
        0.024646868904967967, 
        0.016518109804624086, 
        1e5, 
        500, 
    ]
    
    rl_res = [
        0.03,
        0.1,
        0.0001351369751824273,
        1e-1,
        100 * 1e3,
    ]

    # rect = [0.05, 0.05, 0.9, 0.9]
    rect = [0.1, 0.05, 0.75, 0.9]
    radar = Radar(fig, tit, lab, rect=rect)

    radar.plot(gpmpc_res,  '-', lw=2, color='b', alpha=0.4, label='GP-MPC')
    radar.plot(rl_res, '-', lw=2, color='r', alpha=0.4, label='RL')
    # radar.plot([3, 4, 3, 1, 2,], '-', lw=2, color='g', alpha=0.4, label='third')
    radar.ax.legend()
    fig.subplots_adjust(left=0.2, bottom=0.2, right=0.9, top=0.9, wspace=0.4, hspace=0.4)

    fig.savefig('result_radar.png')