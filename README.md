# Missing-AVQA

**Missing-AVQA** is a research project focusing on handling missing modalities in audio-visual question answering (AVQA) tasks. This project aims to improve robustness in scenarios where one or more input modalities are incomplete or missing, leveraging multimodal knowledge to recall missing information.

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
You can test the model with the following commands:
    ```bash
    bash test.sh

For training your own model:
    ```bash
    bash train.sh



