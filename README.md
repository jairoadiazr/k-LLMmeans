# k-LLMmeans
**LLM-based Centroids for Text Clustering**

This repository contains the official implementation of **k-LLMmeans**, a text clustering algorithm that leverages large language models (LLMs) for dynamic centroid generation. 

📄 **Original Paper**: J. Diaz-Rodriguez. *"k-LLMmeans: Summaries as Centroids for Interpretable and Scalable LLM-Based Text Clustering"* [Available on arXiv](https://arxiv.org/abs/2502.09667)  

*We introduce k-LLMmeans, a novel modification of the k-means clustering algorithm that utilizes LLMs to generate textual summaries as cluster centroids, thereby capturing contextual and semantic nuances often lost when relying on purely numerical means of document embeddings. This modification preserves the properties of k-means while offering greater interpretability: the cluster centroid is represented by an LLM-generated summary, whose embedding guides cluster assignments. We also propose a mini-batch variant, enabling efficient online clustering for streaming text data and providing real-time interpretability of evolving cluster centroids. Through extensive simulations, we show that our methods outperform vanilla k-means on multiple metrics while incurring only modest LLM usage that does not scale with dataset size. Finally, We present a case study showcasing the interpretability of evolving cluster centroids in sequential text streams. As part of our evaluation, we compile a new dataset from StackExchange, offering a benchmark for text-stream clustering.*

## 📂 Repository Structure
- **`kLLMmeans.py`** – Core implementation containing the `kLLMmeans` and `miniBatchKLLMeans` functions. Change value for variable `YOUR_OPENAI_KEY` with your own key before running.
- **`data_loaders/`** – Contains scripts for loading and preprocessing all data.
- **`data_loaders/clean_stackexchange.csv`** – Contains the clean stackexchange dataset used in the paper (unzip first)
- **`processed_data/`** – Folder where processed datasets will be stored (must be generated first).
- **Notebooks:**
  - `offline_experiments.ipynb` – Reproduces offline experiments from the paper.
  - `sequential_experiments.ipynb` – Runs sequential experiments.
  - `case_study_AI.ipynb` – Conducts the AI-related case study.

## ⚡ Getting Started
1. **Preprocess Data**:  
   Before running the experiments, preprocess the datasets by executing:  
   - `data_loaders/preprocess_offline_data.ipynb`
   - `data_loaders/preprocess_stackexchange.ipynb`  

   This will generate the necessary files in the `processed_data/` folder.  
   _(If you'd like to skip this step, contact me, and I can provide it for you —see the paper for details.)_

2. **Run Experiments**:  
   Open the provided Jupyter notebooks to reproduce the results from the paper.

## 📜 Citation
If you use this code or any of the data provided in this repository in your research, please cite the official paper:  
[arXiv:2502.09667](https://arxiv.org/abs/2502.09667)

For the Stack Exchange dataset, please cite both:
- Our processed version: [arXiv:2502.09667](https://arxiv.org/abs/2502.09667)
- Original source: [Internet Archive](https://archive.org/download/stackexchange)

  _(If you'd like the raw dataset described in the paper, contact me, and I can provide it for you —see the paper for details.)_
---

For any inquiries, refer to the contact details in the paper.
