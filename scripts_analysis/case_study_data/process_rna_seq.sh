#!/bin/bash
mkdir -p ./data/input
mkdir -p ./data/SRR
mkdir -p ./data/hg38

#build the hisat2 index for hg38
wget -P ./data/hg38 https://hgdownload.soe.ucsc.edu/goldenPath/hg38/bigZips/hg38.fa.gz
gunzip ./data/hg38/hg38.fa.gz

wget -P ./data/hg38 https://hgdownload.soe.ucsc.edu/goldenPath/hg38/bigZips/genes/hg38.refGene.gtf.gz
gunzip ./data/hg38/hg38.refGene.gtf.gz

hisat2-build -p 8 \
    --ss ./data/hg38/hg38.refGene.gtf \
    --exon ./data/hg38/hg38.refGene.gtf \
    ./data/hg38/hg38.fa \
    ./data/hg38/genome

# generate CRR072990_sorted.bam
wget -P ./data/SRR ftp://download.big.ac.cn/gsa/CRA001315/CRR072990/CRR072990_f1.fastq.gz
wget -P ./data/SRR ftp://download.big.ac.cn/gsa/CRA001315/CRR072990/CRR072990_r2.fastq.gz

fastqc -o ./data/SRR/ ./data/BAM/CRR072990_f1.fastq.gz ./data/SRR/CRR072990_r2.fastq.gz

unzip ./data/SRR/CRR072990_f1_fastqc.zip -d ./data/SRR/
unzip ./data/SRR/CRR072990_r2_fastqc.zip -d ./data/SRR/

trim_galore -o ./data/SRR/ --quality 20 --length 30 ./data/SRR/CRR072990_f1.fastq.gz
trim_galore -o ./data/SRR/ --quality 20 --length 30 ./data/SRR/CRR072990_r2.fastq.gz

hisat2 -x  ./data/hg38/genome -1 ./data/SRR/CRR072990_f1_trimmed.fq.gz -2 ./data/SRR/CRR072990_r2_trimmed.fq.gz -p 5 --summary-file ./data/SRR/CRR072990_align_summary.txt | samtools view -Su | samtools sort -o ./data/input/CRR072990_sorted.bam

samtools index ./data/input/CRR072990_sorted.bam


