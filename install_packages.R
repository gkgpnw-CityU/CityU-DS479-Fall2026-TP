# Installs any of the project's R packages that are missing. Run once:
#   Rscript install_packages.R
pkgs <- c("tidyverse", "tm", "SnowballC", "caret", "glmnet", "naivebayes",
          "Matrix", "rmarkdown", "knitr")
missing <- setdiff(pkgs, rownames(installed.packages()))
if (length(missing)) install.packages(missing, repos = "https://cloud.r-project.org")
message("All packages available: ", paste(pkgs, collapse = ", "))
