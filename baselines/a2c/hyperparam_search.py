#!/usr/bin/env python2

import numpy as np
import itertools
import os
import subprocess
import time
import sys

flags = [['--nsteps 5', '--nsteps 20'],\
        ['--lr 7e-4'],
        ['--max_grad_norm 1','--max_grad_norm 10']]

env = "Reacher-v1"

if __name__ == "__main__":
    launched = 0
    seeds = [1,2,3]
    options = itertools.product(*flags)

    for option in options:
        for seed in seeds:
            current_script = "cd /home/mansimov/projects/learn-to-run/"
            current_script += "/n"
            current_script = 'CUDA_VISIBLE_DEVICES=3 python /home/mansimov/projects/learn-to-run/baselines/a2c/run_mujoco.py --env {} --seed {} --million_timesteps 4 {}'.format(env, seed, ' '.join(option))

            # write current_script to the sh file
            script_path = "/tmp/a2c_mujoco.sh"
            with open(script_path, 'w') as fw:
                fw.write(current_script)
                st = os.stat(script_path)
                os.chmod(script_path, st.st_mode | 0o111)


            cmd = "sh {}".format(script_path)
            p = subprocess.Popen(cmd.split(), stdin=subprocess.PIPE, stdout=sys.stdout)
            launched += 1
            time.sleep(0.05)

    print ("LAUNCHED {} NUMBER OF JOBS".format(launched))
