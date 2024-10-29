# <p align=center> [ECCV 2024] Learning Trimodal Relation for Audio-Visual Question Answering with Missing Modality</p>
Official Repository for "Learning Trimodal Relation for Audio-Visual Question Answering with Missing Modality".

Accepted at [ECCV 2024](https://https://eccv.ecva.net/) <br/>

[[Paper]] [[arXiv](https://arxiv.org/abs/2407.16171)]

## Abstract

Recent Audio-Visual Question Answering (AVQA) methods rely on complete visual and audio input to answer questions accurately. However, in real-world scenarios, issues such as device malfunctions and data transmission errors frequently result in missing audio or visual modality. In such cases, existing AVQA methods suffer significant performance degradation. In this paper, we propose a framework that ensures robust AVQA performance even when a modality is missing. First, we propose a Relation-aware Missing Modal (RMM) generator with Relation-aware Missing Modal Recalling (RMMR) loss to enhance the ability of the generator to recall missing modal information by understanding the relationships and context among the available modalities. Second, we design an Audio-Visual Relation-aware (AVR) diffusion model with Audio-Visual Enhancing (AVE) loss to further enhance audio-visual features by leveraging the relationships and shared cues between the audio-visual modalities. As a result, our method can provide accurate answers by effectively utilizing available information even when input modalities are missing. We believe our method holds potential applications not only in AVQA research but also in various multi-modal scenarios.



## Installation

To set up the project, follow the instructions below:

1. Clone the repository:
   ```bash
   git clone https://github.com/VisualAIKHU/Missing-AVQA.git
2. Install the required dependencies:
    ```bash
    pip install -r requirements.txt

## Usage

1. You can test the model with the following commands:
   ```bash
   bash test.sh
   ```
2. For training your own model:
    ```bash
    bash train.sh
    ```
## Citation
If you use Missing AVQA, please consider citing:

    @article{park2024missingAVQA,
      title={Learning Trimodal Relation for Audio-Visual Question Answering with Missing Modality},
      author={Park, Kyu Ri and Lee, Hong Joo and Kim, Jung Uk},
      journal={arXiv preprint arXiv:2407.16171},
      year={2024}
    }
---

## Acknowlegment

Our codes benefits from the excellent [AVST](https://github.com/GeWu-Lab/MUSIC-AVQA.git), [denoising-diffusion-pytorch
](https://github.com/lucidrains/denoising-diffusion-pytorch).


