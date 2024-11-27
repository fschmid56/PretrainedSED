# Effective Pre-Training of Audio Transformers for Sound Event Detection

In this repository, we publish pre-trained models and code for the ICASSP'25 submission: [**Effective Pre-Training of Audio Transformers for Sound Event Detection**](https://arxiv.org/abs/2409.09546).

In this paper, we propose a pre-training pipeline for audio spectrogram transformers for frame-level sound event detection tasks. On top of common pre-training steps, we add a meticulously designed training routine on AudioSet frame-level annotations. For five transformers, we show that this additional pre-training step leads to substantial performance improvements on frame-level downstream tasks. We release all model checkpoints and hope that they will help researchers improve tasks that require high-quality frame-level representations. 

The codebase is **under construction**; the next steps involve:
* Upload all pre-trained checkpoints and model files [DONE]
* Create a script that demonstrates how the pre-trained checkpoints can be loaded and used for inference [DONE]
* Upload an arxiv version of the submitted paper [DONE]
* Add a table outlining the external checkpoints used in this work [DONE]
* Evaluation routine on the AudioSet frame-level annotations
* Upload the ensemble logits for the AudioSet Strong evaluation set
* Include a clean version of the AudioSet Strong training routine
* Include training routines on the downstream tasks
* Wrap this repository in an installable python package for easy use

## Setting up Environment

1. If needed, create a new environment with python 3.9 and activate it:

```bash
conda create -n ptsed python=3.9
conda activate ptsed
 ```

2. Install pytorch build that suits your system. For example:

```bash
pip3 install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118
# or for cuda >= 12.1
pip3 install torch torchvision torchaudio 
```

3. Install the requirements:

 ```bash
pip3 install -r requirements.txt
 ```

## Inference

The script [inference.py](inference.py) demonstrates how to load a pre-trained model and run sound event detection on an audio file
of arbitrary length.

 ```python
python inference.py --cuda --model_name="BEATs" --audio_file="test_files/752547__iscence__milan_metro_coming_in_station.wav"
 ```

The argument ```model_name``` specifies the transformer used for inference, and the corresponding pre-trained model checkpoint
is automatically downloaded and placed in the folder [resources](resources).

The argument ```audio_file``` specifies the path to a single audio file. There is one [example file](test_files/752547__iscence__milan_metro_coming_in_station.wav) included. 
More example files can be downloaded from the [GitHub release](https://github.com/fschmid56/PretrainedSED/releases/tag/v0.0.1).

## Model Checkpoints

The following is a list of checkpoints that we have created and worked with in our paper. For external checkpoints, we provide the download link. "Checkpoint Name" refers to the respective names in our [GitHub release](https://github.com/fschmid56/PretrainedSED/releases/tag/v0.0.1).

| Model      | Pre-Training | Checkpoint Name    | Download Link                                                                                                             | Reference                                                                        |
|------------|--------------|--------------------|---------------------------------------------------------------------------------------------------------------------------|----------------------------------------------------------------------------------|
| BEATs      | SSL          | BEATs_ssl.pt       | [here](https://1drv.ms/u/s!AqeByhGUtINrgcpxJUNDxg4eU0r-vA?e=qezPJ5)                                                       | [[1]](https://arxiv.org/pdf/2212.09058)                                          |
| BEATs      | Weak         | BEATs_weak.pt      | [here](https://1drv.ms/u/s!AqeByhGUtINrgcpke6_lRSZEKD5j2Q?e=A3FpOf)                                                       | [[1]](https://arxiv.org/pdf/2212.09058)                                          |
| BEATs      | Strong       | BEATs_strong_1.pt  | ours                                                                                                                      | [[1]](https://arxiv.org/pdf/2212.09058)                                          |
| ATST-Frame | SSL          | ATST-F_ssl.pt      | [here](https://drive.google.com/file/d/1bGJSZWlAIIJ6GL5Id5dW0PTB72DL-QDQ/view?usp=sharing)                                | [[2]](https://arxiv.org/pdf/2306.04186)                                          |
| ATST-Frame | Weak         | ATST-F_weak.pt     | [here](https://drive.google.com/file/d/1_xb0_n3UNbUG_pH1vLHTviLfsaSfCzxz/view?usp=drive_link)                             | [[2]](https://arxiv.org/pdf/2306.04186)                                          |
| ATST-Frame | Strong       | ATST-F_strong_1.pt | ours                                                                                                                      | [[2]](https://arxiv.org/pdf/2306.04186)                                          |
| fPaSST     | SSL          | fpasst_im.pt       | [here](https://dl.fbaipublicfiles.com/deit/deit_base_distilled_patch16_384-d0272ac0.pth)                                  | [[3]](https://arxiv.org/pdf/2110.05069), [[4]](https://arxiv.org/pdf/2407.12997) |
| fPaSST     | Weak         | fpasst_weak.pt     | ours                                                                                                                      | [[3]](https://arxiv.org/pdf/2110.05069), [[4]](https://arxiv.org/pdf/2407.12997) |
| fPaSST     | Strong       | fpasst_strong_1.pt | ours                                                                                                                      | [[3]](https://arxiv.org/pdf/2110.05069), [[4]](https://arxiv.org/pdf/2407.12997) |
| ASiT       | SSL          | ASIT_ssl.pt        | [here](https://drive.google.com/file/d/11eaOU40jonpYZ3u_XI-XUSSWclv8qeR7/view?usp=drive_link)                             | [[5]](https://arxiv.org/pdf/2211.13189)                                          |
| ASiT       | Weak         | ASIT_weak.pt       | ours                                                                                                                      | [[5]](https://arxiv.org/pdf/2211.13189)                                          |
| ASiT       | Strong       | ASIT_strong_1.pt   | ours                                                                                                                      | [[5]](https://arxiv.org/pdf/2211.13189)                                          |
| M2D        | SSL          | M2D_ssl.pt         | [here](https://github.com/nttcslab/m2d/releases/download/v0.3.0/m2d_clap_vit_base-80x1001p16x16-240128_AS-FT_enconly.zip) | [[6]](https://arxiv.org/pdf/2406.02032)                                          |
| M2D        | Weak         | M2D_weak.pt        | [here](https://github.com/nttcslab/m2d/releases/download/v0.3.0/m2d_clap_vit_base-80x1001p16x16-240128_AS-FT_enconly.zip) | [[6]](https://arxiv.org/pdf/2406.02032)                                          |
| M2D        | Strong       | M2D_strong_1.pt    | ours                                                                                                                      | [[6]](https://arxiv.org/pdf/2406.02032)                                          |


## Results & Ablation Studies

This section presents the main results reported [in the paper](https://arxiv.org/pdf/2409.09546), along with additional ablation studies, including teacher model performances, comparisons of different sequence models, and evaluations using the DESED baseline system setup. The additional ablation studies have been requested by ICASSP`25 reviewers.

* All results represent averages over three independent runs.
* For AudioSet Strong, we employ the threshold-independent PSDS1 [2] metric to ensure fine-grained temporal evaluation.


### Student Model Performances on AudioSet Strong (*in paper*)

* For the *Li et al. [1]* row, we reproduced their AudioSet Strong [training pipeline](https://github.com/Audio-WestlakeU/audiossl).
* Alongside the **Proposed Pipeline**, we include ablation studies for three settings: no KD, no RNN in teacher models, and no pre-training on AudioSet Weak.

|                      | **ATST-F** | **BEATs** | **fPaSST** | **M2D**  | **ASiT** |
|----------------------|------------|-----------|------------|----------|----------|
| **Li et al. [1]**    | 40.9       | 36.5      | 38.7       | 36.9     | 37.0     |
| **Proposed Pipeline** | **45.8**   | **46.5**  | **45.4**   | **46.3** | **46.2** |
| **-- without KD**    | 41.8       | 44.1      | 40.7       | 41.1     | 40.9     |
| **-- without RNN**   | 45.7       | 45.8      | 45.3       | 46.0     | 46.1     |
| **-- without Step 2** | 45.7       | 46.3      | 45.2       | 44.9     | **46.2** |

**Conclusions:**
* The significant performance gap to [1] stems mainly from our three design choices (KD, RNNs, Step 2), but also improvements in training on AudioSet Strong, including balanced sampling and aggressive data augmentation.
* Knowledge Distillation (KD) has the most substantial impact, underlining the effectiveness of the ensemble-KD approach.
* RNNs in teacher models and pre-training on AudioSet Weak offer modest improvements but are justified due to their low additional cost. Notably, they do not increase student model complexity, and AudioSet Weak checkpoints are publicly available for most transformers.

###  Teacher Model Performances on AudioSet Strong (*not in paper*)

* The table below shows teacher model results for each transformer.  
* Column **Avg. Ind.** represents the average performance across all single models in the row.
* Column **Ensemble** represents the performance of the ensemble consisting of all models in the respective row.

|                               | **ATST-F** | **BEATs** | **fPaSST** | **M2D**  | **ASiT** | **Avg. Ind.** | **Ensemble** |
|-------------------------------|------------|-----------|------------|----------|----------|---------------|--------------|
| **Proposed Teacher Pipeline** | 43.3       | **45.8**  | **43.3**   | **44.1** | **43.3** | **44.9**      | **47.1**     |
| **-- without RNN**            | 41.8       | 44.1      | 40.7       | 41.1     | 40.9     | 41.7          | 46.2         |
| **-- without Step 2**         | **43.5**   | 34.4      | 40.9       | 43.8     | 43.2     | 41.2          | 46.5         |

**Conclusions:**
* *Ensemble Performance*: The Ensemble column reflects the teacher ensemble performances utilized for Knowledge Distillation (KD).
* *Impact of RNNs and Step 2*: Incorporating RNNs and Step 2 (AudioSet Weak pre-training) notably enhances single-model teacher performance, with the exception of ATST-F without Step 2.
* *Benefits of Ensembling*: While individual model performances show considerable variability (Avg. Ind.), ensembling stabilizes and elevates overall performance, as evidenced by the smaller differences in the Ensemble column.
* *BEATs Specific Insights*: BEATs excels in the *Proposed Teacher Pipeline* and *without RNN* settings but underperforms in the *without Step 2* configuration. This discrepancy may be attributed to its unique SSL pre-training routine and longer sequence lengths (resulting from more tokens being extracted from the input).

### Teacher Model Performances with different Sequence Models (*not in paper*)

* The use of an additional sequence model on top of the AudioSet Weak pre-trained transformers stems from our hypothesis that adding capacity specifically for temporally-strong predictions can enhance performance.
* The table below shows teacher model performances for various sequence models added on top of the transformers before training on AudioSet Strong. The paper uses BiGRUs (RNN) as they deliver the best performance.
* We investigated 4 different sequence models:
  * RNN: BiGRUs
  * Attention: Multi-Head Self-Attention with rotary position embeddings
  * Transformer (TF): Transformer Encoder blocks with rotary position embeddings
  * [MAMBA](https://arxiv.org/abs/2312.00752)): Implementation from [mambapy](https://github.com/alxndrTL/mamba.py)
* We varied the inner dimension (dim) and the number of layers (<Model Type>:<#layers>; e.g., TF:2 means two Transformer layers were added on top of the pre-trained transformer).
* **Setup Notes**:
  * Ablations were performed using **ATST-F** due to its computational efficiency.
  * Performance without a sequence model was 41.8 PSDS1, and removing the top Transformer layers, which may overfit to AudioSet Weak labels, decreased performance.
  * For MAMBA, only a single layer was feasible due to memory constraints.

| PSDS1        | RNN:1 |   RNN:2   | RNN:3 |   TF:1    | TF:2  |   TF:3    | ATT:1 |   ATT:2   | ATT:3 |  MAMBA:1  |
|:-------------|:-----:|:---------:|:-----:|:---------:|:-----:|:---------:|:-----:|:---------:|:-----:|:---------:|
| **dim=256**  | 8.72  |   3.76    | 3.10  |   34.25   | 34.62 |   34.05   | 40.08 |   39.70   | 39.55 |   40.27   | 
| **dim=512**  | 40.62 |   7.26    | 0.12  |   40.41   | 41.11 |   40.30   | 41.78 |   41.91   | 41.95 |   41.25   | 
| **dim=1024** | 42.74 |   42.75   | 43.00 |   42.69   | 42.22 |   42.20   | 42.44 | **42.45** | 42.08 | **41.97** | 
| **dim=2048** | 43.41 | **43.43** | 42.66 | **42.90** | 38.94 | **42.90** | 41.58 |   41.59   | 41.42 |   41.72   |

**Conclusions:** 
* *Best model type*: The highest performance was achieved with 2 BiGRU layers, followed by Transformer, Self-Attention, and MAMBA. All sequence models improved performance compared to using no additional sequence model, though MAMBA's gains were marginal.
* *Inner Dimension*: Larger inner dimensions consistently led to better performance across all sequence models. Significant improvements required dimensions ≥1024, while smaller dimensions (e.g., 256) often degraded performance, with severe failures for BiGRU. We believe that large inner dimensions are essential due to the high number of classes (447) in AudioSet Strong.
* *Number of layers*: Performance was relatively insensitive to the number of layers for most sequence models, with optimal results often achieved with just 1–2 layers. 


### Downstream Task Performances

* Three frame-level downstream tasks:
  * DCASE 2023 Task 4: Domestic Environment Sound Event Detection (*DESED*), metric: PSDS 1
  * DCASE 2016 Task 2 (*DC16-T2*), metric: onset F-measure
  * MAESTRO 5hr (*MAESTRO*), metric: onset F-measure
* For DESED, we followed a simplified setup in line with [1], excluding unsupervised data (no mean teacher approach) and additional CRNN components from the [DCASE 2023 Task 4 baseline system](https://github.com/DCASE-REPO/DESED_task/tree/master/recipes/dcase2023_task4_baseline). While state-of-the-art approaches such as [3] and [4] leverage advanced techniques (e.g., multi-stage/multi-iteration training, sophisticated data augmentation, and interpolation consistency training), we deliberately avoided these complexities, as the focus is on a precise evaluation of pre-training quality.

![Downstream Task Results](/images/downstream_task_results.png)

**Conclusions:**
* *In-Domain Tasks*: The pipeline demonstrates strong, consistent improvements for all transformers on DESED and DC16-T2, showcasing its effectiveness for in-domain tasks.
* *Out-of-Domain Task*: Results on MAESTRO (piano pitch prediction) are inconclusive. This limitation suggests that the proposed pre-training strategy yields substantial gains only when audio and labels are similar to the AudioSet ontology.
* *Simplified DESED Setup*: Despite the simplified setup (no CRNN, no unsupervised data), performance remains comparable to the [DCASE 2023 Task 4 baseline system](https://github.com/DCASE-REPO/DESED_task/tree/master/recipes/dcase2023_task4_baseline).

#### DESED Baseline Setup (*not in paper*)

To complement the simplified DESED setup presented earlier, we provide results for the [DCASE 2023 Task 4 baseline system](https://github.com/DCASE-REPO/DESED_task/tree/master/recipes/dcase2023_task4_baseline) setup for ATST-F and BEATs in the table below. Note that hyperparameters were not extensively tuned, and the data setup may differ slightly from the original baseline.

| **Model** | **Checkpoint**   | **Notes**           | **Performance** |
|-----------|------------------|---------------------|-----------------|
| ATST-F    | Step 1 (SSL)     |                     | 42.7            |
| ATST-F    | Step 2 (AS weak) |                     | 47.1            |
| ATST-F    | Full Pipeline    |                     | 50.4            |
| ATST-F    | Full Pipeline    | dropped 2 TF layers | 51.1            |
| BEATs     | Step 1 (SSL)     |                     | 39.7            |
| BEATs     | Step 2 (AS weak) |                     | 48.1            |
| BEATs     | Full Pipeline    |                     | 48.6            |
| BEATs     | Full Pipeline    | dropped 2 TF layers | 51.1            |

**Conclusions**: 
* The *Full Pipeline* substantially improves performance over *Step 1 (SSL)* and *Step 2 (AS Weak)* for both ATST-F and BEATs.
* Dropping the last two Transformer layers notably enhances results, suggesting that the final layers may focus on AudioSet Strong label-specific features, while earlier layers provide more general, transferable embeddings that benefit the DESED task. We will test whether dropping Transformer layers is generalizable to other tasks or specific to the DESED task.

# References

[1] X. Li, N. Shao, and X. Li, “Self-supervised audio teacher-student transformer for both clip-level and frame-level tasks,” Transactions on Audio, Speech, and Language Processing, vol. 32, pp. 1336–1351, 2024.

[2] J. Ebbers, R. Haeb-Umbach, and R. Serizel, “Threshold independent evaluation of sound event detection scores,” in Proceedings of the International Conference on Acoustics, Speech and Signal Processing (ICASSP), 2022.

[3] N. Shao, X. Li, and X. Li, “Fine-tune the pretrained ATST model for sound event detection,” in Proceedings of the International Conference on Acoustics, Speech and Signal Processing (ICASSP), 2024

[4] F. Schmid, P. Primus, T. Morocutti, J. Greif, and G. Widmer, “Multi-iteration multi-stage fine-tuning of transformers for sound event detection with heterogeneous datasets,” in Workshop on Detection and Classification of Acoustic Scenes and Events (DCASE), 2024.