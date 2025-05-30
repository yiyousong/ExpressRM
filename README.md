##  ExpressRM: Zero-shot Learning Framework for RNA Modification Prediction

### **Description**  
ExpressRM is a zero-shot leaning framework for learning RNA modification in unobserved condition types. It enables training and validation of models for human m6A in any tissue or cell types of interest without requiring m6A profiling experiments.
[![DOI](https://zenodo.org/badge/722145690.svg)](https://doi.org/10.5281/zenodo.15226699)

---

### **Requirements**  
Please ensure that the following Python packages are installed:
```bash
# Python 3.x:
python = 3.9.0 
# Dependencies:
numpy = 1.24.3  
pandas = 1.5.3  
scikit-learn = 1.2.2  
biopython = 1.78  
pytorch = 2.0.1  
pytorch-cuda = 11.7  
pytorch-lightning = 2.0.3  
```

---

### **Usage Instructions**  

#### **Getting Started** 
- The **pre-trained model** and **benchmarking datasets** can be downloaded from: [http://www.rnamd.org/ExpressRM/index.html](http://www.rnamd.org/ExpressRM/index.html).  
- The **hg38 reference genome** assembly used in this project is available at: [UCSC Genome Browser](https://hgdownload.soe.ucsc.edu/goldenPath/hg38/bigZips/genes/).  
- The **condition-specific transcriptome annotation**  is derived from the BAM files, which can be accessed via: [DOI: 10.1016/j.molcel.2019.09.032](https://doi.org/10.1016/j.molcel.2019.09.032).  

#### **Directory Instruction**  
- **`/data_preparation`**: This directory contains scripts for feature generation. Users can use these files to preprocess and prepare your data for model training. 
- **`/scripts_analysis`**: This directory includes scripts for data analysis in this project. They were employed for case studies, model comparisons, and other supplementary analyses presented in the paper.
- **`/testing`**:  The test.py in this directory is provided for evaluating the model, which requires input files including target RNA sites in `.rds` format, RNA-seq files in `BAM` format, and hg38 refGene assembly file in`GTF` format. The script generates two prediction files in`.csv` format as output.
