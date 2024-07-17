The following files should be prepared:
- the target RNA sites stored in grange format with condition-specific labels attached (.rds) (See an example http://www.rnamd.org/ExpressRM/index.html)
(If you choose to provide a separate label file and a .rds file without label, you need to comment out the label section in 'sequence%geneexp.R'. In this case, you will need to provide a separate 'selectedlabel.csv' under the main folder.)
- two hg38 assembly (.gtf) 
- a folder containing gene expression profile  (.tab)
- a folder containing condition-specific transcriptome assembly (.gtf)

default location:

main_folder
|gene_expression_raw/*.tab
|transcriptomes/*.gtf
|m6A_hg38_tissue_selected.rds
|hg38.refGene.gtf
|hg38.knownGene.gtf
|(if using seperate label file) selectedlabel.csv
