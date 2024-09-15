# Effective Pre-Training of Audio Transformers for Sound Event Detection

In this repository, we publish pre-trained models and code for the ICASSP'25 submission: **Effective Pre-Training of Audio Transformers for Sound Event Detection**.

In this paper, we propose a pre-training pipeline for audio spectrogram transformers for frame-level sound event detection tasks. On top of common pre-training steps, we add a meticulously designed training routine on AudioSet frame-level annotations. For five transformers, we show that this additional pre-training step leads to substantial performance improvements on frame-level downstream tasks. We release all model checkpoints and hope that they will help researchers improve tasks that require high-quality frame-level representations. 

The codebase is **under construction**; the next steps involve:
* Upload all pre-trained checkpoints and model files [Done for BEATs, ATST-F, and fPaSST]
* Create a script that demonstrates how the pre-trained checkpoints can be loaded and used for inference [Done for BEATs and ATST-F]
* Upload an arxiv version of the submitted paper
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