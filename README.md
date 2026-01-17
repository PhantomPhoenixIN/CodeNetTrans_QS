# CodeNetTrans-QS

Quality-Stratified, Execution-Grounded Synthetic Dataset and Training Framework for Code Translation

---

## Overview

CodeNetTrans-QS is the official research repository accompanying the paper:

**Quality as a Continuum: Execution-Grounded Stratification for Synthetic Code Translation**  
Deepak Naik M V, Swaminathan J

This repository provides a complete, end-to-end framework for constructing, training, and evaluating neural code translation models using execution-grounded, quality-stratified synthetic parallel data derived from Project CodeNet.

The central idea of this work is to treat translation quality as a continuum rather than a binary signal. Instead of discarding imperfect translations, the framework retains and organizes them based on progressively stronger execution-level correctness signals, enabling systematic analysis of learnability, generalization, and functional correctness.

---

## Key Contributions

- Execution-grounded synthetic parallel dataset for code translation
- Quality stratification using AST parsability, compilation success, and functional correctness
- Multi-teacher synthetic translation generation
- Curriculum learning over quality levels
- Fully automated and scalable pipeline
- Competitive-programming–level difficulty using Project CodeNet

---

## Dataset Statistics

The CodeNetTrans-QS dataset is constructed from Project CodeNet using an execution-grounded, quality-stratified pipeline.

| Statistic | Value |
|---------|------|
| Source Dataset | Project CodeNet |
| Input Languages | C, C++, C#, Python, Ruby, Kotlin, Swift |
| Target Language | Java |
| Total Problems | 3,260 |
| Train / Validation / Test (problems) | 2,831 / 113 / 316 |
| Total Translation Pairs | 22,856 |
| Teacher Models | StarCoder, QwenCoder, DeepSeek-Coder |
| Quality Levels | AST-parsable (Score 1), Compilable (Score 2), Functionally Correct (Score 3) |
| Dataset Type | Synthetic, parallel, execution-grounded, quality-stratified |

The dataset preserves translations across all quality levels, enabling controlled empirical analysis of how supervision quality influences learnability and executable correctness.

---

## Execution-Grounded Quality-Stratified Framework

<p align="center">
  <img src="execution_grounded_quality_stratified_pipeline.png" width="900">
</p>

<p align="center">
  <em>
    Execution-grounded quality-stratified code translation framework.
    Multilingual source programs from Project CodeNet are translated using
    multiple teacher models and stratified based on AST parsability,
    compilability, and functional correctness.
  </em>
</p>


## Key Results (Summary)

Using the proposed quality-stratified dataset, a CodeT5-base model trained with curriculum learning achieves strong execution-level performance.

| Model | Functional Correctness (%) |
|------|----------------------------|
| StarCoder (Teacher) | 19.88 |
| QwenCoder (Teacher) | 24.54 |
| CodeT5 (trained on Score-3 only) | 20.41 |
| CodeT5 (Curriculum: Score 1 → Score 2 → Score 3) | 24.62 |

Curriculum learning over quality levels yields a **20.6% relative improvement in functional correctness** compared to training exclusively on functionally correct translations. Detailed language-wise and ablation results are provided in the accompanying paper.

---

## Repository Structure

Github_CodeNetTrans-QS/
└── content/
├── datasets/
├── files/
├── models/
├── env/
└── run_on_hpc.py


Each directory corresponds directly to a stage of the methodology described in the paper.

---

## content/datasets/

This directory contains all dataset artifacts, including raw data, processed datasets, predictions, and evaluation outputs.

### Core Dataset Files

- `codenet_combined_translator_dataset.jsonl`  
  Main synthetic parallel dataset containing translations across all quality levels.

- `codenet_combined_translator_dataset_astfixed.jsonl`  
  AST-normalized version used to ensure stable parsing and evaluation.

- `codenet_single_solution.jsonl`  
  One accepted solution per problem per source language extracted from Project CodeNet.

- `codenet_single_solution_starcoder.jsonl`  
  Java translations generated using StarCoder.

- `codenet_single_solution_qwencoder.jsonl`  
  Java translations generated using QwenCoder.

- `codenet_single_solution_deepseekcoder.jsonl`  
  Java translations generated using DeepSeek-Coder.

Each JSONL entry includes:
- Problem ID
- Source language
- Source code
- Generated Java code
- Teacher model identifier
- Execution-grounded quality score

---

### Execution and Evaluation Artifacts

- `problem_tests.json`  
  Problem-specific test cases extracted from Project CodeNet.

- `complete_test_set_predictions_FFT/`  
  Predictions from full fine-tuning experiments.

- `complete_test_set_predictions_progressive_learning/`  
  Predictions from curriculum learning experiments.

- `complete_test_set_predictions_progressive_learning_reverse/`  
  Predictions from reverse curriculum ablation experiments.

---

### Raw Source Archive

- `Project_CodeNet.tar.gz`  
  Snapshot of Project CodeNet used for dataset construction.

---

## content/files/

This directory contains all Python scripts used for dataset construction, translation generation, training, inference, and evaluation.

### Dataset Construction and Preprocessing

- `data_collection_from_codenet.py`  
  Extracts accepted solutions from Project CodeNet.

- `problemleveldatasetsplitting.py`  
  Performs problem-level train/validation/test splitting to avoid data leakage.

- `sdg_starcoder.py`  
  Synthetic data generation using StarCoder.

- `sdg_qwencoder.py`  
  Synthetic data generation using QwenCoder.

- `sdg_deepseekcoder.py`  
  Synthetic data generation using DeepSeek-Coder.

---

### Translation and Training Modules

- `Translator_Module_Single_Teacher.py`  
  Training using a single teacher model.

- `Translation_Module_Single_Teacher_Curriculum.py`  
  Curriculum learning using a single teacher.

- `Translator_Module_Progressive_Learning.py`  
  Progressive multi-teacher curriculum learning.

- `Translator_Module_Progressive_Learning_Reverse.py`  
  Reverse-order curriculum ablation.

- `Translation_Module_FFT.py`  
  Full fine-tuning baseline without curriculum.

---

### Prediction and Evaluation

- `translator_module_predictions.py`  
  General inference pipeline.

- `translator_module_prediction_complete_test_set.py`  
  Evaluation on the full de-duplicated test set.

- `translator_module_prediction_progressive_learning.py`  
  Inference for curriculum-trained models.

- `translator_module_prediction_progressive_learning_reverse.py`  
  Inference for reverse curriculum models.

- `evaluating_translator_module_individual_scores.py`  
  Evaluation broken down by quality score.

- `evaluating_translator_module_progressive_learning.py`  
  Evaluation of curriculum learning effectiveness.

---

### Teacher Query Scripts

- `starcoderqs.py`  
  StarCoder query interface.

- `qwencoderqs.py`  
  QwenCoder query interface.

- `deepseekcoderqs.py`  
  DeepSeek-Coder query interface.

---

## content/models/

This directory contains pretrained and fine-tuned CodeT5 models used in the experiments.

models/
├── CodeT5_Translation_full_finetune/
├── CodeT5_Translation_Progressive_Learning/
├── CodeT5_Translation_Progressive_Learning_Reverse/
├── CodeT5_Translation_Deepseek/
└── CodeT5.zip


- `CodeT5_Translation_full_finetune`  
  Model trained on a single quality level.

- `CodeT5_Translation_Progressive_Learning`  
  Model trained using curriculum learning from low to high quality.

- `CodeT5_Translation_Progressive_Learning_Reverse`  
  Reverse curriculum ablation model.

- `CodeT5_Translation_Deepseek`  
  Teacher-specific comparison model.

- `CodeT5.zip`  
  Base pretrained CodeT5 model.

---

## Execution-Grounded Quality Levels

Each translation is assigned a quality score based on execution-level correctness:

- Score 1: AST-parsable Java code
- Score 2: Successfully compilable Java code
- Score 3: Functionally correct Java code that passes all test cases

This stratification is central to the experimental design and analysis.

---

## Running Experiments

The repository is designed for execution on HPC clusters.

python run_on_hpc.py


This script coordinates:
- Dataset loading
- Model training
- Distributed execution
- Checkpointing and logging

---

## Reproducibility

- Problem-level dataset splits prevent leakage
- De-duplicated test set ensures unbiased evaluation
- Deterministic execution-grounded scoring
- Fully automated pipeline from data generation to evaluation

---


---

## Contact

Deepak Naik M V  
PhD Scholar, Program Analysis and AI  
Email: deepaknaikmv01@gmail.com

Swaminathan J  
Senior Member, IEEE  

Department of Computer Science and Engineering  
Amrita Vishwa Vidyapeetham, India
