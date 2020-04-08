# code_snippets

PLEASE FOLLOW INSTRUCTIONS on https://stanfordnlp.github.io/CoreNLP/download.html
to get latest version of stanford_corenlp

Different modules for miscellaneous tasks. In this way I avoid rewriting code.

dataprep: contains modules for common tasks such as reducing the number of levels of categorical variables with high cardinality, correcting typos in categorical (text) variables using fuzzy matching, etc...

feature_engineering: modules for feature engineering such as holiday based feature engineering, cosine and sine transformation for cyclical variables (e.g. month of year), etc...

prediction_engineering: modules that help to transform business problem into machine learning problem when data is not previously labeled. For example: transforming a time series to a supervised problem, labeling data for customer churn prediction.
