usage:
test.py -b [bam1,bam2,...bamN] -o [output_folder] -s [site_grange.rds] -m [path_to_model.ckpt] --refgene [hg38.refGene.gtf]
--knowngene [hg38.knowngene.gtf]
the results will be [output_folder]/prediction.csv and [output_folder]/siteprediction.csv
Note that refgene.gtf is used for gene expression and knowngene.gtf is used for geographic encoding. For gene expression, we want only important protein coding genes. For geographic encodings, we want to use all available mRNA coding transcripts 

