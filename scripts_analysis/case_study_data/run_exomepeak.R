if (!require("GenomicRanges")) install.packages("GenomicRanges")
if (!require("exomePeak2")) install.packages("exomePeak2")

library(GenomicRanges)
library(exomePeak2)

bam_input <- " ./data/input/CRR072990_sorted.bam"
bam_ip <- " ./data/input/CRR072991_sorted.bam"
gene_anno_gtf <- " ./data/input/hg38.refGene.gtf" #download from https://hgdownload.soe.ucsc.edu/goldenPath/hg38/bigZips/genes/

genome <- "hg38"

exomePeak2_output <- exomePeak2(bam_ip = bam_ip,
                                bam_input = bam_input,
                                gff = gene_anno_gtf,
                                genome = genome)

saveRDS(exomePeak2_output$peaks, file = " ./data/input/peaks.rds")