from visdom import Visdom
import numpy as np
import glob
import os
import argparse
from load import load_data, load
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
matplotlib.rcParams.update({'font.size': 8})
from PIL import Image
import itertools
color_defaults = [
    '#1f77b4',  # muted blue
    '#ff7f0e',  # safety orange
    '#2ca02c',  # cooked asparagus green
    '#d62728',  # brick red
    '#9467bd',  # muted purple
    '#8c564b',  # chestnut brown
    '#e377c2',  # raspberry yogurt pink
    '#7f7f7f',  # middle gray
    '#bcbd22',  # curry yellow-green
    '#17becf'  # blue-teal
]

parser = argparse.ArgumentParser(description="Run commands")

parser.add_argument('--folder', type=str, default="/misc/vlgscratch2/FergusGroup/mansimov/a2c/")
parser.add_argument('--game', type=str, default="learntorun")
parser.add_argument('--smooth', type=int, default=1,
                    help="Smooth with openai smoothing", required=False)
parser.add_argument('--bin_size', type=int, default=100,
                    help="bin size for average", required=False)
parser.add_argument('--outfile', type=str, default="./figure.pdf",
                    help='outfile', required=False)

"""
flags = [['-lr0.007', '-lr0.0007'],
        ['-max_grad_norm10.0'],
        ['-nsteps5', '-nsteps20']]
"""
"""
flags = [['-lr0.0007'],
        ['-max_grad_norm10.0', '-max_grad_norm1.0'],
        ['-nsteps5', '-nsteps20']]
"""
flags = [['-lr7e-06'],#['-lr0.0007'],
        ['-max_grad_norm10.0', '-max_grad_norm1.0'],
        ['-nsteps5', '-nsteps20', '-nsteps100']]

if __name__ == "__main__":

    args = parser.parse_args()

    seeds = [1,2,3]
    options = itertools.product(*flags)

    i = 0
    for option in options:
        tmpx, tmpy = [], []
        head_folder = "/misc/vlgscratch2/FergusGroup/mansimov/a2c-learntorun/learntorun{}".format(''.join(option))
        for seed in seeds:
            indir = os.path.join(head_folder, "seed-{}".format(seed))
            print (indir)

            tx, ty = load_data(indir, args.smooth, args.bin_size)
            tmpx.append(tx)
            tmpy.append(ty)

        length = min([len(t) for t in tmpx])
        for j in range(len(tmpx)):
            tmpx[j] = tmpx[j][:length]
            tmpy[j] = tmpy[j][:length]

        x = np.mean(np.array(tmpx), axis=0)
        y_mean = np.mean(np.array(tmpy), axis=0)
        y_std = np.std(np.array(tmpy), axis=0)

        lines = []
        color = color_defaults[i]
        y_upper = y_mean + y_std
        y_lower = y_mean - y_std
        plt.fill_between(
            x, list(y_lower), list(y_upper), interpolate=True, facecolor=color, linewidth=0.0, alpha=0.3
        )
        #line = plt.plot(x, list(y_mean), label=indir.split('/')[-1], color=color)
        line = plt.plot(x, list(y_mean), label=''.join(option), color=color)

        lines.append(line[0])

        i = i + 1

    plt.xticks([1e6, 5e6, 10e6, 20e6, 40e6], ["1M", "5M", "10M", "20M", "40M"])
    plt.xlabel('Samples')
    plt.ylabel('Rewards')

    plt.xlim(0, 1e6)
    #plt.ylim(-100., 0.)

    plt.title(args.game)
    plt.legend(loc=4)
    plt.show()
    plt.draw()
    plt.savefig('figure.jpg', format='jpg', dpi=100)

    print ('Saved figure... Showing in Visdom')
    # Show it in visdom
    viz = Visdom()
    image = Image.open('figure.jpg')
    image.load()
    image = np.asarray(image)
    image = np.transpose(image, (2, 0, 1))
    #print (image.shape)
    viz.image(image)
