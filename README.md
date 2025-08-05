# Hate Speech Detection Using NLP

This project tackles the challenge of detecting hate speech on social media platforms using NLP and deep learning models. We compare the performance of two neural models: GRU and BERT, in classifying tweets into three categories:

- Hate Speech  
- Offensive Language  
- Neither

---

##  Dataset and Preprocessing

We used a labeled dataset of tweets and applied the following preprocessing steps:

- Text cleaning: removed URLs, mentions, hashtags, and non-alphabetic characters
- Lemmatization with WordNet
- POS tagging with extraction of:
  - Noun, Verb, and Adjective counts
  - POS Unigrams, Bigrams, Trigrams

The processed data was saved as `preprocessed_with_pos_features.csv`.

---

## ⚙️ Models Implemented

### GRU
- Custom embedding layer
- Dropout regularization
- Dense Softmax output
- Trained on lemmatized text

### BERT
- HuggingFace pretrained model
- Token classification
- Fine-tuned using Transformers library

---

##  Evaluation

| Model | Accuracy | Precision | Recall | F1-Score |
|-------|----------|-----------|--------|----------|
| GRU   | 0.87     | 0.85      | 0.86   | 0.85     |
| BERT  | 0.91     | 0.90      | 0.92   | 0.91     |

---

##  How to Run

```bash
# Clone the repository
git clone git@github.com-rimas:Rimas-Alhazmi/Hate-Speech-Detection-Using-NLP.git
cd Hate-Speech-Detection-Using-NLP

# Install required packages
pip install -r requirements.txt

# Run the notebook
jupyter notebook "NLP_Project R.ipynb"
