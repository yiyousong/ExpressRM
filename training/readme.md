Before utilizing the scripts in the /training directory to train the model, ensure that the data and processes outlined in the /data_preparation folder have been properly prepared.

Only ExpressRM.py needs to be called to train a model. for example: 
nohup python ExpressRM.py --featurelist 1,1,1,1,1 --minepoch 100 --maxepoch 200 --testlist 0,1,2,3 &

Run ExpressRM.py -h for help.

