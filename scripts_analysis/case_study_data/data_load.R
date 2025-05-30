if (!require("GenomicRanges")) install.packages("GenomicRanges")
library(GenomicRanges)

main <- function(input_dir = "./data/input", output_dir = "./test") {
  if (!dir.exists(output_dir)) dir.create(output_dir, recursive = TRUE)
  
  data <- list(
    peaks = readRDS(file.path(input_dir, "peak.rds")),  
    site_grange = readRDS(file.path(input_dir, "m6A_hg38_tissue_selected.rds"))
  )
  
  ovlp <- findOverlaps(data$site_grange, data$peaks)
  write.table(
    ovlp@from, 
    file.path(output_dir, "label.csv"),
    col.names = FALSE, 
    row.names = FALSE, 
    sep = ","
  )
}

args <- commandArgs(trailingOnly = TRUE)
main(
  input_dir = ifelse(length(args) > 0, args[1], "./data/input"),
  output_dir = ifelse(length(args) > 1, args[2], "./test")
)