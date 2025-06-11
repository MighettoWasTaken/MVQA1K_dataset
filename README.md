# MVQA1k Dataset

This repository contains a reproducible pipeline for generating, filtering, and sampling medical visual question answering (VQA) data using the Med-R1 model and MiniGPT-Med. The output is a curated dataset of 1,000 diverse, high-quality image-question-answer-reasoning samples, designed to support downstream fine-tuning and evaluation of clinical multimodal models such as LLava-Med.

This work serves as an empirical testbed for the ideas presented in recent research on test-time scaling strategies in medical LLMs.

## Referenced Papers

- **S1**: *Simple Test-Time Scaling Improves Zero-Shot Reasoning with Large Language Models*  
  [arXiv:2501.19393](https://arxiv.org/abs/2501.19393)

- **M1**: *Unleash the Potential of Test-Time Scaling for Medical Reasoning with Large Language Models*  
  [arXiv:2504.00869](https://arxiv.org/abs/2504.00869)

## Pipeline Overview

1. **Difficulty Filtering with MiniGPT-Med**  
   - Dataset: [RadGenome/PMC-VQA on Hugging Face](https://huggingface.co/datasets/RadGenome/PMC-VQA)  
   - Model: [MiniGPT-Med GitHub](https://github.com/Vision-CAIR/MiniGPT-Med)  
   - Filtered dataset (top ~15% most difficult questions):  
     [Google Drive Link](https://drive.google.com/file/d/1ejXK73W0Siym0-wsj_M5XjPfD1OcaPC5/view?usp=drive_link)

2. **Reasoning Trace Generation with Med-R1**  
   - Model: [Med-R1 on Hugging Face](https://huggingface.co/yuxianglai117/Med-R1)  
   - Prompts include structured <think>...</think> and <answer>...</answer> tags for interpretability

3. **Correct Answer Filtering**  
   - Only examples where Med-R1’s prediction exactly matches the ground-truth `Answer_label` are retained

4. **Diversity Sampling via K-Means Clustering**  
   - TF-IDF embeddings computed over both question text and image identifiers  
   - Image vectors are down-weighted to prioritize semantic question diversity  
   - K-means used to cluster and sample 1,000 representative examples

## Goals: 

5. **Augment Llava-Med to generate wait tokens**  
   - Med-R1 traces are reformatted with wait-token prompting for LLava-Med compatibility

6. **Fine-tuning LLava-Med**  
   - Fine-tuned on the curated dataset to evaluate improvements in reasoning performance

7. **Performance Comparison**  
   - Evaluate performance improvements using the methodology 

## Data and Outputs

| Resource                         | Link |
|----------------------------------|------|
| Filtered dataset (11k hard cases)   | [Google Drive](https://drive.google.com/file/d/1ejXK73W0Siym0-wsj_M5XjPfD1OcaPC5/view?usp=drive_link)  
| Final dataset (1k sampled examples) | [Google Drive](https://drive.google.com/file/d/1WY51Yg18F1J8gD8FUlpnP72mSh3qXbxH/view?usp=sharing)  
| Final images ZIP (1k matched)      | [Google Drive](https://drive.google.com/file/d/11RpwkqAfg51pK3mHIyrJa0I-mX689X03/view?usp=sharing)
| `traced_partial.csv`              | Raw model outputs (including failed or incorrect predictions)  
| `final1k.csv`                     | Final filtered and sampled dataset with reasoning traces  
