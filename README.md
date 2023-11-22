# ExpressRM
code for ExpressRM  
the reference genome  hg38.knownGene.gtf.gz can be downloaded from https://hgdownload.soe.ucsc.edu/goldenPath/hg38/bigZips/genes/ , you should unzip to script folder  
the model and default site_file can be downloaded from __webiste__placeholder__, you should also download it to script folder.  
if you plan to provide the sites, the site file should contained single-base sites in grange format stored as .rds file  
to run the test run test.py -b [bam1,bam2,...bamN] -o [output_folder] -s [site_grange.rds] -m [path_to_model.ckpt] --refgene [hg38.refGene.gtf]  
the results will be [output_folder]/prediction.csv and [output_folder]/siteprediction.csv  
