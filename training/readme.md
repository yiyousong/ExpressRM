first prepare the data using the scripts in preparation folder then run ExpressRM.py to train

run ExpressRM.py -h for help
only the comments in ExpressRM.py are curated. the rest can be outdated

you should only need to call ExpressRM.py to train a model. for example: 
nohup python ExpressRM.py --featurelist 1,1,1,1,1 --minepoch 100 --maxepoch 200 --testlist 0,1,2,3 &
