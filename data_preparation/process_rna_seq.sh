#!/bin/bash
mkdir -p ./data/input
mkdir -p ./data/SRR
mkdir -p ./data/hg38

# download hg38 genome and annotation files
wget -P ./data/hg38 https://hgdownload.soe.ucsc.edu/goldenPath/hg38/bigZips/hg38.fa.gz
gunzip ./data/hg38/hg38.fa.gz

wget -P ./data/hg38 https://hgdownload.soe.ucsc.edu/goldenPath/hg38/bigZips/genes/hg38.refGene.gtf.gz
gunzip ./data/hg38/hg38.refGene.gtf.gz

# build the HISAT2 index for hg38
hisat2-build -p 8 \
    --ss ./data/hg38/hg38.refGene.gtf \
    --exon ./data/hg38/hg38.refGene.gtf \
    ./data/hg38/hg38.fa \
    ./data/hg38/genome

# process sample fastq files 
SAMPLE_NAME="your_sample_name"
fastqc -o ./data/SRR/ ./data/SRR/${SAMPLE_NAME}_f1.fastq.gz ./data/SRR/${SAMPLE_NAME}_r2.fastq.gz

# unzip fastqc results
unzip ./data/SRR/${SAMPLE_NAME}_f1_fastqc.zip -d ./data/SRR/
unzip ./data/SRR/${SAMPLE_NAME}_r2_fastqc.zip -d ./data/SRR/

trim_galore -o ./data/SRR/ --quality 20 --length 30 ./data/SRR/${SAMPLE_NAME}_f1.fastq.gz
trim_galore -o ./data/SRR/ --quality 20 --length 30 ./data/SRR/${SAMPLE_NAME}_r2.fastq.gz

hisat2 -x ./data/hg38/genome -1 ./data/SRR/${SAMPLE_NAME}_f1_trimmed.fq.gz -2 ./data/SRR/${SAMPLE_NAME}_r2_trimmed.fq.gz -p 5 --summary-file ./data/SRR/${SAMPLE_NAME}_align_summary.txt | samtools view -Su | samtools sort -o ./data/input/${SAMPLE_NAME}_sorted.bam

samtools index ./data/input/${SAMPLE_NAME}_sorted.bam


