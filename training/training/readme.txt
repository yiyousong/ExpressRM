run ExpressRM.py -h for help
check SLURM_3090.sh for the commands I used to train the models
only the comments in ExpressRM.py are curated. the rest can be outdated

you should only need to call ExpressRM.py to train a model
for example: nohup python ExpressRM.py --featurelist 1,1,1,1,1 --epoch 1000 &