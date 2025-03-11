# k-LLMmeans
**LLM-based Centroids for Text Clustering**

This repository contains the official implementation of **k-LLMmeans**, a text clustering algorithm that leverages large language models (LLMs) for dynamic centroid generation. 

📄 **Original Paper**: [Available on arXiv](https://arxiv.org/abs/2502.09667)  

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

_(If you'd like to the raw dataset described in the paper, contact me, and I can provide it for you —see the paper for details.)_
---

For any inquiries, refer to the contact details in the paper.
