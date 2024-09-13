# Effective Pre-Training of Audio Transformers for Sound Event Detection

In this repository, we publish pre-trained models and the code for the ICASSP'25 submission: **Effective Pre-Training of Audio Transformers for Sound Event Detection**.

In this paper, we propose a pre-training pipeline for audio spectrogram transformers for frame-level sound event detection tasks. On top of common pre-training steps, we add a meticulously desigined training routine on AudioSet frame-level annotations. For five transformers, we show that this additional pre-training step leads to substantial performance improvements on frame-level downstream tasks. We release all model checkpoints and hope that they will help researchers improving tasks that require high-quality frame-level representations. 

The codebase is **under construction**, the next steps involve:
* Upload all pre-trained checkpoints and model files [In Progress]
* Create a script that demonstrates how the pre-trained checkpoints can be loaded and used for inference [In Progress]
* Upload an arxiv version of the submitted paper [In Progress]
* Add a table outlining the external checkpoints used in this work
* Evaluation routine on the AudioSet frame-level annotations
* Upload the ensemble logits for the AudioSet Strong evaluation set
* Include a clean version of the AudioSet Strong training routine
* Include training routines on the downstream tasks
* Wrap this repository in an installable python package for easy use



