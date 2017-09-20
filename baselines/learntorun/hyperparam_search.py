#!/usr/bin/env python2

import numpy as np
import itertools
import os
import subprocess
import time
import sys

flags = [['--nsteps 5', '--nsteps 20', '--nsteps 100'],\
        ['--lr 7e-4', '--lr 7e-5', '--lr 7e-6'],
        ['--max_grad_norm 1', '--max_grad_norm 10']]


if __name__ == "__main__":
    launched = 0
    seeds = [1,2,3]
    options = itertools.product(*flags)
    device = 1
    for option in options:
        for seed in seeds:
            """
            # hardcoded
            if launched % 6 == 0 and launched > 0:
                device = device + 1
            # GPU 1 not working on vine14
            if device == 1:
                device = 2
            """
            if launched % 9 == 0 and launched > 0:
                device = device + 1
            # hardcoded
            if device == 3:
                device = 4

            current_script = "cd /home/mansimov/projects/learn-to-run/"
            current_script += "/n"
            current_script = 'CUDA_VISIBLE_DEVICES={} python /home/mansimov/projects/learn-to-run/baselines/learntorun/run.py --seed {} --million_timesteps 40 {}'.format(device, seed, ' '.join(option))


            # write current_script to the sh file
            script_path = "/tmp/a2c_learntorun.sh"
            with open(script_path, 'w') as fw:
                fw.write(current_script)
                st = os.stat(script_path)
                os.chmod(script_path, st.st_mode | 0o111)


            cmd = "sh {}".format(script_path)
            p = subprocess.Popen(cmd.split(), stdin=subprocess.PIPE, stdout=sys.stdout)
            launched += 1
            time.sleep(0.05)

    print ("LAUNCHED {} NUMBER OF JOBS".format(launched))
