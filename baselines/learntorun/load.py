import json
import glob
import os
import numpy as np
from scipy.signal import medfilt
import sys

def smooth_reward_curve(x, y):
    # Halfwidth of our smoothing convolution
    halfwidth = min(31, int(np.ceil(len(x) / 30)))
    k = halfwidth
    xsmoo = x[k:-k]
    ysmoo = np.convolve(y, np.ones(2 * k + 1), mode='valid') / \
        np.convolve(np.ones_like(y), np.ones(2 * k + 1), mode='valid')
    downsample = max(int(np.floor(len(xsmoo) / 1e3)), 1)
    return xsmoo[::downsample], ysmoo[::downsample]


def fix_point(x, y, interval):
    np.insert(x, 0, 0)
    np.insert(y, 0, 0)

    fx, fy = [], []
    pointer = 0
    ninterval = int(max(x) / interval + 1)

    for i in range(ninterval):
        tmpx = interval * i

        while pointer + 1 < len(x) and tmpx > x[pointer + 1]:
            pointer += 1

        if pointer + 1 < len(x):
            alpha = (y[pointer + 1] - y[pointer]) / \
                (x[pointer + 1] - x[pointer])
            tmpy = y[pointer] + alpha * (tmpx - x[pointer])
            fx.append(tmpx)
            fy.append(tmpy)

    return fx, fy

def load_data(indir, smooth, bin_size):
    datas = []
    infiles = glob.glob(os.path.join(indir, '*monitor.json'))

    for inf in infiles:
        with open(inf, 'r') as f:
            t_start = float(json.loads(f.readline())['t_start'])
            for line in f:
                tmp = json.loads(line)
                t_time = float(tmp['t']) + t_start
                tmp = [t_time, int(tmp['l']), float(tmp['r'])]
                datas.append(tmp)

    datas = sorted(datas, key=lambda d_entry: d_entry[0])
    result = []
    timesteps = 0
    for i in range(len(datas)):
        result.append([timesteps, datas[i][-1]])
        timesteps += datas[i][1]

    x, y = np.array(result)[:, 0], np.array(result)[:, 1]

    if smooth == 1:
        x, y = smooth_reward_curve(x, y)

    if smooth == 2:
        y = medfilt(y, kernel_size=9)

    x, y = fix_point(x, y, bin_size)
    return [x, y]

def load(indir, subdir, smooth, bin_size):
    all_dirs = glob.glob(os.path.join(indir, '*/'))
    dirs = []
    for d in all_dirs:
        if subdir in d:
            dirs.append(d)

    result = []
    tmpx, tmpy = [], []

    for i in range(len(dirs)):
        indir = dirs[i]

        tx, ty = load_data(indir, smooth, bin_size)
        tmpx.append(tx)
        tmpy.append(ty)

    if len(tmpx) > 1:
        length = min([len(t) for t in tmpx])
        for j in range(len(tmpx)):
            tmpx[j] = tmpx[j][:length]
            tmpy[j] = tmpy[j][:length]

        x = np.mean(np.array(tmpx), axis=0)
        y_mean = np.mean(np.array(tmpy), axis=0)
        y_std = np.std(np.array(tmpy), axis=0)
    else:
        x = np.array(tmpx).reshape(-1)
        y_mean = np.array(tmpy).reshape(-1)
        y_std = np.zeros(len(y_mean))

    return x, y_mean, y_std
