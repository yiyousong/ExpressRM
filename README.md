##  ExpressRM
1.	**Description**  
We presented ExpressRM, a zero-shot leaning framework for learning RNA modification in unobserved condition types. It enabled the training and validation of models for human m6A in any tissue or cell types of interest without doing m6A profiling experiments.

2.	**Requirements**  
Please make sure the following packages are installed in the Python environment:  
python = 3.9.0  
numpy = 1.24.3  
pandas = 1.5.3    
scikit-learn = 1.2.2
biopython = 1.78
pytorch = 2.0.1   
pytorch_cuda = 11.7
pytorch_lightning  = 2.0.3  

3.	**Usage instruction** 
* ExpressRM is used for the study of single-base RNA modification sites in a new condition. 
* The well-trained model and benchmarking datasets can be downloaded from http://www.rnamd.org/ExpressRM/index.html. 
* Files in /data_preparation are provided for feature generation. 
* Files in /scripts_analysis are uploaded for review purpose only.  
* The hg38 reference genome assembly used in this project is available at https://hgdownload.soe.ucsc.edu/goldenPath/hg38/bigZips/genes/. The condition-specific transcriptome assembly annotation file is derived from the BAM files of each condition, accessible via https://doi.org/10.1016/j.molcel.2019.09.032.
* The test.py located in the /test directory is provided for the testing purpose, which requires input files including target RNA sites in `.rds` format, RNA-seq files in `BAM` format, and the hg38 refGene assembly file in`GTF` format. It generates two prediction files in `.csv` format as output.

