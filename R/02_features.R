# Step 2: split into train/test, down-sample the large classes in the training
# set, and turn text into word-count matrices with a fixed vocabulary.
# Output: data/processed/features.rds

source("R/utils.R")
library(caret)
message("== Step 2: split and build features")

emails <- readRDS(file.path(PATHS$processed, "emails_clean.rds"))

# Stratified split: every class keeps the same share in train and test.
set.seed(CONFIG$seed)
train_idx <- createDataPartition(emails$class, p = 1 - CONFIG$test_frac, list = FALSE)[, 1]
train_full <- emails[train_idx, ]
test       <- emails[-train_idx, ]

# Down-sample legit and spam in TRAINING only, so the small malicious class
# isn't drowned out. The test set keeps its natural class mix.
set.seed(CONFIG$seed)
train <- train_full |>
  group_by(class) |>
  slice_sample(n = CONFIG$train_cap_per_class) |>
  ungroup()

message("Training emails: ", nrow(train), " | Test emails: ", nrow(test))
print(table(train = train$class))
print(table(test = test$class))

# Vocabulary is learned from the training set only.
train_counts_all <- make_dtm(preprocess_corpus(train$text))
doc_freq <- colSums(train_counts_all > 0)
vocab <- names(sort(doc_freq[doc_freq >= CONFIG$min_doc_freq], decreasing = TRUE))
vocab <- head(vocab, CONFIG$max_vocab)
message("Vocabulary size: ", length(vocab))

train_counts <- train_counts_all[, vocab]
test_counts  <- make_dtm(preprocess_corpus(test$text), vocab = vocab)

idf <- log(nrow(train_counts) / (1 + doc_freq[vocab]))

saveRDS(list(
  train_counts = train_counts, y_train = train$class, train_source = train$source,
  test_counts  = test_counts,  y_test  = test$class,  test_source  = test$source,
  test_text    = test$text,
  vocab = vocab, idf = idf
), file.path(PATHS$processed, "features.rds"))
