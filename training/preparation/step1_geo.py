import os
import time
### change this folder!!! ###
### folder for the transcriptomes ###
gtffolder='/home/yiyou/tissue/transcriptome/'

### looping is done outside the script for parallel processing ###
os.system('nohup Rscript geo_commandarg.R ref &')
for file in os.listdir(gtffolder):
    os.system('nohup Rscript geo_commandarg.R %s &'%(file[:-4]))
    ### waits for 10 seconds to even resource needs ###
    time.sleep(10)
