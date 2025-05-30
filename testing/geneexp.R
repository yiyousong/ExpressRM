library(GenomicRanges)
library(stringr)
library(plyr)
library(rtracklayer)
library(Biostrings)

gene_exp_list <- c()
header_list <- c()

# define directory and load tissue list
filelist <- list.files('./data/transcriptomes/')
tissuelist <- str_sub(filelist, 1, -5)
write(tissuelist, './data/tissuename.txt')

# import reference gene annotations
refg <- import('./data/hg38/hg38.refGene.gtf')
transcripts <- refg[refg$type == 'transcript']

# sequence data
seq2001 <- readRDS('./data/input/m6A_hg38_tissue_selected.rds')
writeXStringSet(seq2001$refseq_2001, './data/selected.fasta')

# label data
label <- data.frame(seq2001@elementMetadata[2:39])
colnames(label) <- c('Site-label', colnames(seq2001@elementMetadata)[3:39])
label[1] <- (label[1] == 'P')
label <- label * (data.frame(seq2001@elementMetadata[2]) == 'P')
label <- label[, c(order(colnames(label)[2:38]) + 1, 1)]
write.table(label, './data/selectedlabel.csv', sep = ',', row.names = FALSE)

# define file prefixes for output
outprefix <- './data/gene_expression/'
folderprefix <- './data/gene_expression_raw/'

# read the gene expression matrix for the first tissue
genetab_path <- paste0(folderprefix, tissuelist[1], '.tab')
gene_exp_table <- read.table(genetab_path, header = 1, sep = '\t')

# initialize gene expression matrix
genename <- sort(unique(gene_exp_table$Gene.ID))
genename2idx <- data.frame(1:length(genename))
rownames(genename2idx) <- genename
geneexp <- matrix(0, nrow = length(genename), ncol = 38)
colnames(geneexp) <- c('GeneName', tissuelist)
geneexp[, 1] <- genename

# initialize location expression list
locexplist <- data.frame(matrix(0, nrow = length(seq2001), ncol = 37))
colnames(locexplist) <- tissuelist

# loop through all tissues
for (i in 1:37) {
  genetab_path <- paste0(folderprefix, tissuelist[i], '.tab')
  gene_exp_table <- read.table(genetab_path, header = 1, sep = '\t')
  colnames(gene_exp_table)[3] <- 'Seqnames'
  
  df <- data.frame(gene_exp_table$Gene.ID, gene_exp_table$TPM)
  df <- aggregate(gene_exp_table.TPM ~ gene_exp_table.Gene.ID, df, sum) 
  geneexp[genename2idx[df[, 1], ], i + 1] <- df[, 2]
  
  gene_exp_grange <- makeGRangesFromDataFrame(gene_exp_table, keep.extra.columns = TRUE)
  
  ovlp <- findOverlaps(seq2001, gene_exp_grange)
  rowidx <- ovlp@from
  repgene <- gene_exp_grange[ovlp@to]
  
  locexp <- data.frame('rowidx' = rowidx, 'level' = repgene$TPM)
  locexp <- aggregate(locexp$level ~ locexp$rowidx, locexp, sum) 
  locexplist[locexp[, 1], i] <- locexp[, 2]
}

# Log2 transformation for expression data
locexplist <- log2(locexplist + 1)
write.table(locexplist, './data/gene_expression/lg2hosting_expression.csv', sep = ',', row.names = FALSE)

geneexp2 <- geneexp[, 2:38]
geneexp2 <- as.double(geneexp2)
geneexp2 <- log2(geneexp2 + 1)
geneexp[, 2:38] <- geneexp2

write.table(geneexp, './data/gene_expression/lg2geneexp.csv', sep = ',', col.names = TRUE, row.names = FALSE)
