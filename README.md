# CityU-DS479-Fall2026-TP

A rudimentary R text classifier that labels an email as **legit**, **spam**, or **malicious**
(phishing) from its subject and body text alone. It is a course project, not a production
email-security system: it never looks at senders, links, or attachments.

**Full write-up:** [`report/email_classifier.html`](report/email_classifier.html), knitted from
[`report/email_classifier.Rmd`](report/email_classifier.Rmd).

**Plain-language summary** (method choice, splits, hyperparameters, evaluation):
[`report/email_classifier_summary.html`](report/email_classifier_summary.html), knitted from
[`report/email_classifier_summary.Rmd`](report/email_classifier_summary.Rmd).

## Results (held-out test set, 17,665 emails)

| Model | Accuracy | Macro F1 | Malicious recall | Malicious precision |
|---|---|---|---|---|
| **LASSO logistic regression** (main) | 96.4% | 91.1% | 95.0% | 68.7% |
| Naive Bayes (baseline) | 93.0% | 83.2% | 96.0% | 45.8% |

Both models catch about 95% of phishing emails. The LASSO model raises far fewer false
alarms, and most of its false alarms are spam rather than legitimate mail.

## Quick start

From the project root (or open `CityU-DS479-Fall2026-TP.Rproj` in RStudio):

```bash
Rscript install_packages.R
```

```bash
Rscript run_all.R
```

`run_all.R` runs the four steps below and knits both reports. It takes about 3 minutes.

To classify new emails with the trained model (`models/model.rds` is included in the repo):

```r
source("R/predict.R")
classify_email("Your mailbox is suspended. Click here to verify your account.",
               subject = "Action required")
```

## Pipeline

| Script | What it does |
|---|---|
| `R/utils.R` | Shared settings (`CONFIG`) and text-cleaning, document-term-matrix and TF-IDF helpers |
| `R/01_load_clean.R` | Loads the four datasets, maps them to 3 classes, cleans the text, removes duplicates |
| `R/02_features.R` | Stratified 80/20 split (caret), down-samples legit/spam in training, builds the vocabulary (tm + SnowballC) |
| `R/03_train.R` | Trains multinomial LASSO logistic regression (glmnet, class weights, 5-fold CV to pick lambda) and multinomial Naive Bayes (naivebayes); saves cross-validation (validation) results |
| `R/04_evaluate.R` | Confusion matrices, per-class precision/recall/F1, accuracy by source, top words, plots → `outputs/` |
| `R/predict.R` | `classify_email()` for scoring new emails |

## Data

| Folder | Source | Mapped to |
|---|---|---|
| `Enron-Spam/` | Enron-Spam (Metsis, Androutsopoulos & Paliouras, 2006), CSV version by M. Wiechmann | legit, spam |
| `lingspam/` | Ling-Spam (Androutsopoulos et al., 2000), `bare` version | legit, spam |
| `phishing-email-dataset/email_text.csv` | Appears to be the TREC 2007 public spam corpus (pre-cleaned) | legit, spam |
| `phishing-email-dataset/Nazario_5.csv` | Nazario phishing corpus 2015–2022 (label 1) plus legitimate filler (label 0) | malicious, legit |

The `phishing-email-dataset` files most likely come from the Kaggle "Phishing Email Dataset"
(Al-Subaiey et al., 2024, CC BY-SA 4.0). After cleaning there are 88,329 emails: 42,072 legit,
44,740 spam and 1,517 malicious.

**Leakage controls.** All sources are normalised to the same text format. The Nazario
`sender` and `urls` columns are not used, and the Nazario honeypot address, encoded header
text, CSS and corpus-specific names are stripped, so the model can't identify the class from
artifacts of the source dataset.

## Limitations

- "Malicious" means phishing judged by text only. Malware is usually carried in attachments
  or links, which are not inspected.
- All malicious examples come from one corpus. Legitimate mail comes mostly from Enron and
  technical or academic mailing lists, and most data predates 2008.

## Environment

Tested with R 4.3.3, tidyverse 2.0.0, tm 0.7-11, SnowballC 0.7.1, caret 6.0-94, glmnet 4.1-8,
naivebayes 1.0.0, Matrix 1.6-5, rmarkdown 2.25.
