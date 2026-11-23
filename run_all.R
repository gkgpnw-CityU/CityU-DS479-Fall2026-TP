# Runs the whole pipeline from raw data to the evaluation outputs.
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

message("Done. Results are in outputs/; the trained model is in models/model.rds")
