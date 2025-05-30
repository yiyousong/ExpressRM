### Command Usage

```bash
test.py -b [sorted.bam1, sorted.bam2, ..., sorted.bamN] -o [output_folder] -s [site_grange.rds] -m [path_to_model.ckpt] --refgene [hg38.refGene.gtf] --knowngene [hg38.knowngene.gtf]
```

### Parameters:
- `-b [sorted.bam1, sorted.bam2, ..., sorted.bamN]`: Comma-separated list of sorted BAM files for the input data.
- `-o [output_folder]`: Path to the output directory where the results will be saved.
- `-s [site_grange.rds]`: Path to the target RNA site range RDS file.
- `-m [path_to_model.ckpt]`: Path to the pre-trained model checkpoint file.
- `--refgene [hg38.refGene.gtf]`: Path to the reference gene annotation GTF file, which is used for gene expression analysis). This file contains only important protein-coding genes.
- `--knowngene [hg38.knowngene.gtf]`: Path to the known gene annotation GTF file, which is used for geographic encoding. This file includes all available mRNA-coding transcripts.
