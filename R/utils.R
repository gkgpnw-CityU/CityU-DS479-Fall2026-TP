# Shared settings and helper functions used by every step of the pipeline
# and by classify_email(). Source this file from the project root.

suppressPackageStartupMessages({
  library(tidyverse)
  library(tm)
  library(SnowballC)
  library(Matrix)
})

if (!file.exists("run_all.R")) {
  stop("Run this project from its root folder (open the .Rproj file in RStudio, or setwd() to the repo root).")
}

CONFIG <- list(
  seed                = 479,
  test_frac           = 0.20,   # share of emails held out for testing
  train_cap_per_class = 6000,   # down-sample legit/spam to at most this many training emails
  max_words_per_email = 1500,   # truncate very long emails
  min_doc_freq        = 5,      # a word must appear in at least this many training emails
  max_vocab           = 5000    # keep the most common words up to this many
)

CLASSES <- c("legit", "spam", "malicious")

PATHS <- list(
  processed = "data/processed",
  models    = "models",
  outputs   = "outputs"
)
walk(PATHS, dir.create, showWarnings = FALSE, recursive = TRUE)

# Colors for the three classes (validated categorical palette, fixed order).
CLASS_COLORS <- c(legit = "#2a78d6", spam = "#eb6834", malicious = "#1baf7a")

# Words that identify which corpus an email came from rather than what it says
# (e.g. Enron employee names, the Nazario honeypot mailbox). Removing them keeps
# the model from learning the source instead of the content.
CORPUS_ARTIFACT_WORDS <- c(
  "enron", "ect", "hou", "kaminski", "vince", "nazario", "jose", "monkey"
)

# Step 1 of text cleaning (fast, string level). Applied identically to every
# source so that formatting differences between datasets don't leak the label.
# email_text.csv arrives already lowercased with punctuation removed, so every
# other source is brought to that same form.
clean_text <- function(x, max_words = CONFIG$max_words_per_email) {
  x <- iconv(coalesce(x, ""), from = "UTF-8", to = "UTF-8", sub = " ")
  x <- str_to_lower(x)
  x <- str_replace_all(x, "=\\?[^?\\s]+\\?[bq]\\?[^?\\s]*\\?=", " ") # MIME-encoded header text
  x <- str_replace_all(x, "\\S*monkey\\.org\\S*", " ")       # Nazario honeypot mailbox address
  x <- str_replace_all(x, regex("<(style|script)[^>]*>.*?</\\1>", dotall = TRUE), " ") # CSS / JS blocks
  x <- str_replace_all(x, "\\{[^{}]{0,2000}\\}", " ")        # stray CSS rules { ... }
  x <- str_replace_all(x, "<[^>]{0,500}>", " ")              # HTML tags
  x <- str_replace_all(x, "&[a-z#0-9]{2,8};", " ")           # HTML entities
  x <- str_replace_all(x, "\\b(escapenumber|escapelong)\\b", " numtoken ")
  x <- str_replace_all(x, "[0-9]+", " numtoken ")
  x <- str_replace_all(x, "[^a-z ]+", " ")                   # punctuation, symbols
  x <- str_squish(x)
  words <- str_split(x, " ")
  map_chr(words, ~ paste(head(.x, max_words), collapse = " "))
}

# Step 2 of text cleaning (tm + SnowballC): remove stop words and corpus
# artifacts, then stem. Returns a tm corpus.
preprocess_corpus <- function(text) {
  corpus <- VCorpus(VectorSource(text))
  corpus <- tm_map(corpus, removeWords, c(stopwords("en"), CORPUS_ARTIFACT_WORDS))
  corpus <- tm_map(corpus, stemDocument, language = "english")
  corpus <- tm_map(corpus, stripWhitespace)
  corpus
}

# Build a sparse document-term matrix of word counts. With `vocab`, columns are
# exactly that vocabulary in that order (used for test data and new emails).
make_dtm <- function(corpus, vocab = NULL) {
  ctrl <- list(wordLengths = c(2, 25))
  if (!is.null(vocab)) ctrl$dictionary <- vocab
  dtm <- DocumentTermMatrix(corpus, control = ctrl)
  if (is.null(vocab)) {
    return(sparseMatrix(i = dtm$i, j = dtm$j, x = dtm$v,
                        dims = dim(dtm), dimnames = list(NULL, Terms(dtm))))
  }
  # Map tm's columns onto the fixed vocabulary order (words outside it are dropped).
  col <- match(Terms(dtm), vocab)[dtm$j]
  keep <- !is.na(col)
  sparseMatrix(i = dtm$i[keep], j = col[keep], x = dtm$v[keep],
               dims = c(nrow(dtm), length(vocab)), dimnames = list(NULL, vocab))
}

# Per-class precision / recall / F1 plus overall accuracy and macro F1,
# computed with caret's confusionMatrix.
class_metrics <- function(pred, truth, model_name) {
  cm <- caret::confusionMatrix(factor(pred, levels = CLASSES),
                               factor(truth, levels = CLASSES), mode = "prec_recall")
  list(
    by_class = as_tibble(cm$byClass, rownames = "class") |>
      transmute(model = model_name, class = str_remove(class, "^Class: "),
                precision = Precision, recall = Recall, f1 = F1,
                support = as.integer(colSums(cm$table))),
    overall = tibble(model = model_name,
                     accuracy = cm$overall[["Accuracy"]],
                     macro_f1 = mean(cm$byClass[, "F1"], na.rm = TRUE),
                     malicious_recall = cm$byClass["Class: malicious", "Recall"]),
    table = cm$table
  )
}

# TF-IDF weighting: log-scaled term counts times inverse document frequency
# (learned from training data only), with each email scaled to unit length.
tfidf <- function(counts, idf) {
  x <- counts
  x@x <- log1p(x@x)
  x <- x %*% Diagonal(x = idf)
  norms <- sqrt(rowSums(x^2))
  norms[norms == 0] <- 1
  x <- Diagonal(x = 1 / norms) %*% x
  colnames(x) <- colnames(counts)
  as(x, "CsparseMatrix")
}
