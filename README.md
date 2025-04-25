# Legal Text Classification Project

![Python](https://img.shields.io/badge/python-3.9%2B-blue)
![NLP](https://img.shields.io/badge/NLP-Legal%20Text-brightgreen)
![License](https://img.shields.io/badge/license-MIT-green)

## 📌 Introduction
A hybrid legal text classification system combining:
- Rule-based keyword matching
- Unsupervised machine learning (TF-IDF + K-means)
- Dimensionality reduction (UMAP)

**Motivation:** As a pre-law student, I developed this to solve the challenge of efficiently finding relevant legal cases for research across different domains (family law, criminal law, etc.).

## 📂 Dataset
**Source:** [Kaggle Legal Text Dataset](https://www.kaggle.com/datasets/shivamb/legal-citation-text-classification)  
**Contents:** 24,985 Australian legal cases with:
- Case ID (unique identifier)
- Outcome (judicial decision)
- Title (case name)
- Text (full narrative)

## 🛠️ Methodology

### 1. Preprocessing Pipeline
```python
# Sample preprocessing steps
text = text.lower()
text = re.sub(r'[^\w\s]', '', text)  # Remove punctuation
tokens = word_tokenize(text)          # Tokenization
lemmatized = [lemmatizer.lemmatize(t) for t in tokens]
clean_text = [t for t in lemmatized if t not in stopwords]
```
### 2. Hybrid Classification

```python
legal_categories = {
    'family': ['custody', 'divorce', 'marriage', ...],
    'criminal': ['theft', 'murder', 'fraud', ...],
    # 8 other domains...
}
```
**Unsupervised Learning:** 
- TF-IDF Vectorization (1-3 ngrams)
- UMAP Dimensionality Reduction (n_components=20)
- K--means Clustering (k=19 via silhouette score)

## 3. Performance Metrics

| Stage        | Silhouette Score | Calinski-Harabasz |
|--------------|------------------|-------------------|
| Initial      | 0.036            | 159.75            |
| After UMAP   | 0.36             | 7607.45           |

## 📊 Results

### Cluster Analysis

**Top Business Law Terms:**
```python
['case', 'court', 'agreement', 'party', 'contract', 'ltd']
['fca', 'decision', 'applicant', 'immigration', 'tribunal']
```

## 🔍 Key Findings

- Successfully categorized 60% of cases via initial keyword matching
- Cluster quality improved 10x after UMAP reduction
- Identified limitations in business/financial law separation

## 📝 Discussion

### Advantages
✅ **Efficient** - Quick categorization of obvious cases  
✅ **Transparent** - Clear reasoning for classifications  
✅ **Adaptable** - Expandable keyword dictionary  

### Limitations
⚠️ **Keyword dependence** - May miss niche terminology  
⚠️ **Cluster overlap** - Some domain boundaries unclear  
⚠️ **Resource constraints** - Limited BERT implementation  

## 🌟 Future Work

- Web interface for legal researchers
- Dynamic keyword expansion
- BERT fine-tuning with better hardware
- Hierarchical classification system

## 👩‍💻 Contributor

[Yoshita Aligina](https://github.com/yoshialigina)  
Rutgers University CS 2026  

## 📚 References

1. Nghiem et al. (2022) - Transformer-based legal classification (https://aclanthology.org/2022.lrec-1.504.pdf)
2. [UMAP Documentation](https://umap-learn.readthedocs.io/)
3. [HuggingFace Transformers](https://huggingface.co/docs/transformers)
4. https://www.geeksforgeeks.org/removing-stop-words-nltk-python/
5. https://stanfordnlp.github.io/CoreNLP/lemma.html
