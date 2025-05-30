# The BAM file of each condition can be accessed via https://doi.org/10.1016/j.molcel.2019.09.032 
# The condition-specific annotation file is the transcriptome assembly (GTF files) of each condition. It can be extracted using stringtie.  A sample command is shown below.

stringtie  ./data/input/sorted.bam -o ./data/transcriptomes/transcriptome.gtf 

# The gene expression of each condition can be extracted using stringtie. A sample command is shown below.
# The annotation files of human genome assembly hg38 can be accessed via https://hgdownload.soe.ucsc.edu/goldenPath/hg38/bigZips/genes/ 

stringtie  ./data/input/sorted.bam -e -G hg38.refGene.gtf -A ./data/gene_expression/geneexp.tab

