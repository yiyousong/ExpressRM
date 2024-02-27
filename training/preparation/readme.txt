before data preprocessing you should have:
the site in grange format with labels attached(.rds) (this means the file we provided)
(you should rewrite the step1.R script for a different input file. start by checking the inputs needed for step2.py )
gene expression (.tab)
gene structure (.gtf) 
reference gene structure (.gtf) 
(if you do not have the (.tab) and (.gtf) files, you can use stringtie -e -G mode for gene structure and stringtie  -A mode for gene expression)
(this step is not included in the scripts.)
change the path used in all four scripts to match your data.

1.run step1.R and step1_geo.py first, the second creates many background processes.
the step1_geo.py will exit soon, but you should expect one day for the background processes to finish, use htop to check.
2.confirm the background processes have finished then run the step2.py 

