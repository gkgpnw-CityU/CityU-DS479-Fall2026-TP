# Step 3: train the models.
#   * Main model: multinomial LASSO logistic regression (glmnet) on TF-IDF
#     features, with class weights, lambda chosen by 5-fold cross-validation.
#   * Baseline: multinomial Naive Bayes (naivebayes) on raw word counts.
# Output: models/model.rds

source("R/utils.R")
library(glmnet)
library(naivebayes)
message("== Step 3: train models")

f <- readRDS(file.path(PATHS$processed, "features.rds"))
x_train <- tfidf(f$train_counts, f$idf)

# Class weights: each class contributes equally to the loss.
class_freq <- table(f$y_train)
w <- as.numeric(sum(class_freq) / (length(class_freq) * class_freq[as.character(f$y_train)]))

set.seed(CONFIG$seed)
t0 <- Sys.time()
lasso <- cv.glmnet(x_train, f$y_train, family = "multinomial", alpha = 1,
                   weights = w, nfolds = 5, type.measure = "deviance")
message(sprintf("glmnet trained in %.1f min; lambda.min = %.5f",
                as.numeric(difftime(Sys.time(), t0, units = "mins")), lasso$lambda.min))

nb <- multinomial_naive_bayes(x = f$train_counts, y = f$y_train, laplace = 1)
message("Naive Bayes trained")

model <- list(
  lasso = lasso, lambda = "lambda.min", nb = nb,
  vocab = f$vocab, idf = f$idf, classes = CLASSES, config = CONFIG,
  trained_on = Sys.Date()
)
saveRDS(model, file.path(PATHS$models, "model.rds"), compress = "xz")
