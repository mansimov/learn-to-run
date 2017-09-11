# Learn to Run

![alt text](https://images.contentful.com/7h71s48744nc/ttbuWX4JtsGcpN25BIz41u/d6eb57c5801370b80e356e06f297f1af/Forrest-Gump-large.jpg "Logo Title Text 1")

Follow instructions to install opensim-rl

```
https://github.com/stanfordnmbl/osim-rl
```

Make sure to look at below link in case of errors

```
https://github.com/stanfordnmbl/osim-rl#frequently-asked-questions
```

The code is directly taken from OpenAI Baselines and changed to work with python 2.7 (might be some problems still)

After run (make sure to fix some paths)

```
CUDA_VISIBLE_DEVICES=0 python baselines/learntorun/run.py```

**TODO (for now)**

* Add separate evaluation only on level 2 difficulty as it will be evaluated in the competition
* Hack and play around with reward function
* Add ACKTR for better on-policy sample efficiency
* Add ACKTR-LSTM for better results ?
* Add PPO loss for off-policy samples
