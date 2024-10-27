# Missing-AVQA

**Missing-AVQA** is a research project focusing on handling missing modalities in audio-visual question answering (AVQA) tasks. This project aims to improve robustness in scenarios where one or more input modalities are incomplete or missing, leveraging multimodal knowledge to recall missing information.

## Table of Contents

- [Introduction](#introduction)
- [Motivation](#motivation)
- [Methodology](#methodology)
- [Results](#results)
- [Installation](#installation)
- [Usage](#usage)
- [Contributing](#contributing)
- [License](#license)

## Introduction

Audio-Visual Question Answering (AVQA) is a challenging task where the system must answer questions based on both visual and auditory inputs. However, in real-world applications, one or more input modalities may be missing or incomplete. This project addresses the issue by designing networks that can recall missing modalities, inspired by human cognition.

## Motivation

While existing AVQA methods perform well in controlled settings, they often struggle in dynamic, real-world environments where inputs are noisy or incomplete. Our approach seeks to bridge this gap by focusing on multimodal recall and robust handling of missing data.

## Methodology

We propose a memory network that leverages shared clues between audio and visual data, enabling the system to recall missing modalities from incomplete inputs. The model uses multimodal fusion techniques to enhance performance in missing modality scenarios. The detailed methodology is presented in our [research paper](link to the paper or preprint).

## Results

Our model was tested on the MUSIC-AVQA dataset and demonstrated state-of-the-art performance in handling missing modalities. Key results are summarized below:

- **Accuracy on Complete Data:** 85%
- **Accuracy with Missing Modality:** 78%
- **Improvement over Baseline:** +6% on average

For a full breakdown of the results, refer to Tables 1 and 2 in our paper.

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



