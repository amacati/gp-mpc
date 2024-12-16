# plot a empty figure with text "placeholder" in the middle for a given size

import matplotlib.pyplot as plt

def plot_placeholder(size=(8, 4), text='placeholder'):
    fig, ax = plt.subplots(figsize=size)
    ax.text(0.5, 0.5, text, horizontalalignment='center', verticalalignment='center', fontsize=20)
    ax.axis('off')
    fig.savefig(f'placeholder_{size[0]}x{size[1]}.png')
    print(f'placeholder_{size[0]}x{size[1]}.png saved')

plot_placeholder()