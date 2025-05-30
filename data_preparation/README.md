### Required Files
Before running the scripts, make sure to prepare the following data files. Other data objects will be generated using scripts in this directory:

- **Target RNA Sites with Condition-Specific Labels (in .rds format)**:  You can download the file [m6A_hg38_tissue_selected.rds](http://www.rnamd.org/ExpressRM/index.html) and place it in the ./data/input within the main directory. This file is a GRanges object contains labels. If you want to use your own data, customize it with labels in the same format as the default file.

- **Human Genome (hg38) Assembly (in .gtf format)**: This file contains gene annotations for the human genome (hg38). You can download the file from [UCSC Genome Browser](https://hgdownload.soe.ucsc.edu/goldenPath/hg38/bigZips/genes/). We recommend placing it in the ./data/hg38.

- **Condition-Specific Transcriptome Assembly Files (in .bam and .gtf formats)**: Use the `process_rna_seq.sh` script to generate the .bam file, and place it in the ./data/input. Use the `stringtie_commands_for_condition_data.sh` script to generate the .gtf file, and place it in the ./data/transcriptomes.

---
### Directory Structure
The following directory structure should be in place once the data preparation is complete.
```
main_folder
|-- data
|   |-- input
|   |   |-- m6A_hg38_tissue_selected.rds #RNA site data with labels
|   |   |-- sorted.bam  #Aligned BAM file
| 
|   |-- hg38
|   |   |-- hg38.refGene.gtf #hg38 gene annotation file
|
|   |-- transcriptomes
|   |   |--transcriptome.gtf #Condition-specific transcriptome assembly
|
|   |--gene_expression
|   |   |--lg2hosting_expression.csv
|   |   |--lg2geneexp.csv
|   |   |--lg2geneexp.pt
|   |   |--genelocexp_.pt
|   |   |--genelocexp_sitetest.pt
|   |   |--genelocexp_test.pt
|
|   |-- sequence
|   |   |--sequence.pt
|   |   |--sequence_sitetest.pt
|   |   |--sequence_tissuetest.pt
|   |   |--sequence_test.pt
|   |   |--sequence.npy
|
|   |-- geo
|   |   |--geo.csv
|
|-- test
|   |-- label.csv
|   |-- testidx.npy
|   |-- label_sitetest.pt
|   |-- tissue_specificityidx.npy
|   |-- testidx_balanced.npy
|   |-- sitetestidx.npy
|
|-- train
|   |-- label.csv
|   |-- testidx.npy
|   |-- label_sitetest.pt
|   |-- tissue_specificityidx.npy
|   |-- testidx_balanced.npy
|   |-- sitetestidx.npy
|
|-- model                                                      
|   |-- model.ckpt #http://www.rnamd.org/ExpressRM/index.html
|
|-- ExpressRM #Directory where ExpressRM scripts should be placed
|   |--(Download the scripts directly from the GitHub repository and put them in this directory)
```
---
