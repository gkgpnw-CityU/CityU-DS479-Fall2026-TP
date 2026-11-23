# Step 1: load the four source datasets, map them to three classes
# (legit / spam / malicious), clean the text and remove duplicates.
# Output: data/processed/emails_clean.rds, outputs/data_summary.csv

source("R/utils.R")
message("== Step 1: load and clean")

# --- lingspam (bare version: no lemmatizer, no stop list) ------------------
ling_files <- list.files("lingspam/bare", pattern = "\\.txt$",
                         recursive = TRUE, full.names = TRUE)
lingspam <- tibble(
  file   = ling_files,
  text   = map_chr(ling_files, ~ paste(read_lines(.x), collapse = " ")),
  class  = if_else(str_starts(basename(ling_files), "spmsg"), "spam", "legit"),
  source = "lingspam"
) |>
  mutate(text = str_remove(text, "^Subject:")) |>
  select(text, class, source)

# --- Enron-Spam -------------------------------------------------------------
# (readr warns about an "embedded null" in 2 messages; both rows still load.)
enron <- read_csv("Enron-Spam/enron_spam_data.csv", show_col_types = FALSE,
                  col_types = cols(.default = col_character())) |>
  transmute(
    text   = paste(coalesce(Subject, ""), coalesce(Message, "")),
    class  = if_else(`Spam/Ham` == "spam", "spam", "legit"),
    source = "enron"
  )

# --- email_text.csv (TREC 2007 spam corpus; label 1 = spam, 0 = legit) ------
trec <- read_csv("phishing-email-dataset/email_text.csv", show_col_types = FALSE,
                 col_types = cols(.default = col_character())) |>
  transmute(
    text   = text,
    class  = if_else(label == "1", "spam", "legit"),
    source = "email_text"
  )

# --- Nazario_5.csv (label 1 = phishing -> malicious, 0 = legit) -------------
# The sender and urls columns are deliberately not used: they give away the
# label (honeypot mailbox address; urls is a 0/1 flag only for phishing rows).
nazario <- read_csv("phishing-email-dataset/Nazario_5.csv", show_col_types = FALSE,
                    col_types = cols(.default = col_character())) |>
  filter(!str_detect(coalesce(subject, ""), "FOLDER INTERNAL DATA"),
         !str_detect(coalesce(body, ""), "internal format of your mail folder")) |>
  transmute(
    text   = paste(coalesce(subject, ""), coalesce(body, "")),
    class  = if_else(label == "1", "malicious", "legit"),
    source = "nazario"
  )

raw <- bind_rows(lingspam, enron, trec, nazario)
message("Loaded ", nrow(raw), " emails")

# --- clean, drop near-empty emails, de-duplicate ----------------------------
emails <- raw |>
  mutate(text = clean_text(text),
         n_words = str_count(text, "\\S+")) |>
  filter(n_words >= 3)

# Identical cleaned text appearing under different classes is ambiguous: drop it.
# Otherwise keep one copy of each duplicate.
conflicts <- emails |> distinct(text, class) |> count(text) |> filter(n > 1)
emails <- emails |>
  anti_join(conflicts, by = "text") |>
  distinct(text, .keep_all = TRUE) |>
  mutate(class = factor(class, levels = CLASSES))

summary_tbl <- raw |>
  count(source, class, name = "raw") |>
  left_join(emails |> count(source, class, name = "after_cleaning") |>
              mutate(class = as.character(class)),
            by = c("source", "class")) |>
  mutate(after_cleaning = coalesce(after_cleaning, 0L))

print(summary_tbl)
message("Kept ", nrow(emails), " emails after cleaning and de-duplication")
print(table(emails$class))

saveRDS(emails, file.path(PATHS$processed, "emails_clean.rds"))
write_csv(summary_tbl, file.path(PATHS$outputs, "data_summary.csv"))
