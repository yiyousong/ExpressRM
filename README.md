# ExpressRM
code for ExpressRM  

for prediction usage check the test folder

the reference genome hg38.knownGene.gtf.gz can be downloaded from https://hgdownload.soe.ucsc.edu/goldenPath/hg38/bigZips/genes/ , you should unzip the file to test folder.  
the model and default site_file can be downloaded from __webiste__placeholder__, you should also download it to test folder.  
if you plan to provide the sites, the site file should contained single-base sites in grange format stored as .rds file  

to predict, run:  
test.py -b [bam1,bam2,...bamN] -o [output_folder] -s [site_grange.rds] -m [path_to_model.ckpt] --refgene [hg38.refGene.gtf]  

the results will be [output_folder]/prediction.csv and [output_folder]/siteprediction.csv  

For training check the training folder. 

the uploaded_for_reference folder contains the code I used to train the model and cannot be directly run on another computer, running them will result in error for certain.
