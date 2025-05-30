Before using the scripts in the `/training` directory to train the model, ensure that the data and processes specified in the `/data_preparation` directory have been properly set up.

To train a model, run the following command:
```bash
nohup python ExpressRM.py --featurelist 1,1,1,1,1 --minepoch 100 --maxepoch 200 --testlist 0,1,2,3 &
```
For additional information and usage options, run:
```bash
python ExpressRM.py -h
```