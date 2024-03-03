before data preprocessing you should have:
the site in grange format with labels attached(.rds) (this means the file we provided)
(you should remove the label section in the step1.R script for a different input file. remeber to rename the labelfile in step2.py to match your label file )

gene expression (.tab)
gene structure (.gtf) 
(the (.tab) files and (.gtf)  files should be in two seperate folder containing nothing else, no matter how many transcriptomes you use) 

reference gene structure (.gtf) 
(this file should NOT be together with other transcriptomes as the preparation script loops through the folder)

(if you do not have the (.tab) and (.gtf) files, you can use stringtie -e -G mode for gene structure and stringtie  -A mode for gene expression, as this step is not included in our scripts.)

you also need to change the path used in all four scripts to match your data.
the path that needs to be change are located at the head of the files

default location:

main_folder
|gene_expression_raw/
|transcriptomes/
|m6A_hg38_tissue_selected.rds
|hg38.refGene.gtf
|hg38.knownGene.gtf
