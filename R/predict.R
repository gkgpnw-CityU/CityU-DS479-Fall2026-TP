# classify_email(): score new emails with the trained model.
#
# Usage (from the project root, after running run_all.R once):
#   source("R/predict.R")
#   classify_email("Your account has been suspended. Click here to verify.")
#   classify_email(body = c(email1, email2), subject = c(subj1, subj2))
#
# Returns one row per email: the predicted class and the probability of each class.

source("R/utils.R")
suppressPackageStartupMessages(library(glmnet))

.model_cache <- new.env()

load_model <- function(path = file.path(PATHS$models, "model.rds")) {
  if (is.null(.model_cache$model)) {
    if (!file.exists(path)) stop("No trained model at ", path, " - run run_all.R first.")
    .model_cache$model <- readRDS(path)
  }
  .model_cache$model
}

classify_email <- function(body, subject = "", model = load_model()) {
  text   <- clean_text(paste(subject, body))
  counts <- make_dtm(preprocess_corpus(text), vocab = model$vocab)
  x      <- tfidf(counts, model$idf)
  probs  <- predict(model$lasso, x, s = model$lambda, type = "response")[, , 1, drop = FALSE]
  probs  <- matrix(probs, nrow = length(text), dimnames = list(NULL, dimnames(probs)[[2]]))
  probs  <- probs[, model$classes, drop = FALSE]

  tibble(
    predicted = factor(model$classes[max.col(probs, ties.method = "first")], levels = model$classes),
    p_legit     = probs[, "legit"],
    p_spam      = probs[, "spam"],
    p_malicious = probs[, "malicious"]
  )
}

# Demo when run directly:  Rscript R/predict.R
if (sys.nframe() == 0) {
  examples <- tribble(
    ~subject,                          ~body,
    "Team meeting moved to Thursday",  "Hi all, the project sync is moving to Thursday at 2pm in room 301. Please bring your status updates.",
    "Limited time offer!!!",           "Buy cheap watches and pills now, 80% off, free shipping, order today and save big.",
    "Action required: verify account", "Dear customer, your mailbox has been suspended due to unusual activity. Click the link below and log in to verify your account within 24 hours."
  )
  print(bind_cols(examples |> select(subject), classify_email(examples$body, examples$subject)))
}
