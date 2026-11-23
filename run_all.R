# Runs the whole pipeline from raw data to the HTML report.
# Usage (from the project root):  Rscript run_all.R
# or open CityU-DS479-Fall2026-TP.Rproj in RStudio and source this file.

steps <- c(
  "R/01_load_clean.R",
  "R/02_features.R",
  "R/03_train.R",
  "R/04_evaluate.R"
)

for (s in steps) {
  t0 <- Sys.time()
  source(s, local = new.env())
  message(sprintf("-- %s finished in %.1f min\n", s,
                  as.numeric(difftime(Sys.time(), t0, units = "mins"))))
}

for (rmd in c("report/email_classifier.Rmd", "report/email_classifier_summary.Rmd")) {
  rmarkdown::render(rmd, knit_root_dir = getwd(), quiet = TRUE)
  message("Report written to ", sub("\\.Rmd$", ".html", rmd))
}
