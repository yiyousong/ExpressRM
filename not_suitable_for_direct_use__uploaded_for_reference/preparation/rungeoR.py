import os
import time
os.system('nohup Rscript geo_commandarg.R 0 &')
for file in os.listdir('/home/yiyou/tissue/transcriptome/'):
    os.system('nohup Rscript geo_commandarg.R %s &'%(file[:-4]))
    time.sleep(10)
#import pandas as pd
#for file in os.listdir('/home/yiyou/tissue/geo/'):
#    pd.read_csv('/home/yiyou/tissue/geo/%s'% (file)).iloc[:,6:].to_csv('/home/yiyou/tissue/geo/%s'% (file))
#    os.system('mv /home/yiyou/tissue/geo/%s /home/yiyou/tissue/geo/%s.csv' % (file,file[:-8]))