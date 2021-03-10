irectional Spatio-Temporal Reasoning for Video-Grounded Dialogues 
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT) 

**Note**: This branch is for the TGIF-QA experiments. For the video-grounded dialogue experiments on AVSD, please change the repo branch to [main](https://github.com/salesforce/BiST).

This is the PyTorch implementation of the paper:
**[BiST: Bi-directional Spatio-Temporal Reasoning for Video-Grounded Dialogues](<https://www.aclweb.org/anthology/2020.emnlp-main.145/>)**. [**Hung Le**](https://github.com/henryhungle), [Doyen Sahoo](https://scholar.google.com.sg/citations?user=A61jJD4AAAAJ&hl=en), [Nancy F. Chen](https://sites.google.com/site/nancyfchen/home), [Steven C.H. Hoi](https://sites.google.com/view/stevenhoi/). ***[EMNLP 2020](<https://www.aclweb.org/anthology/2020.emnlp-main.145/>)***. ([arXiv](https://arxiv.org/abs/2010.10095)) 


This code has been written using PyTorch 1.0.1. If you find the paper or the source code useful to your projects, please cite the following bibtex: 
<pre>
@inproceedings{le-etal-2020-bist,
    title = "{B}i{ST}: Bi-directional Spatio-Temporal Reasoning for Video-Grounded Dialogues",
    author = "Le, Hung  and
      Sahoo, Doyen  and
      Chen, Nancy  and
      Hoi, Steven C.H.",
    booktitle = "Proceedings of the 2020 Conference on Empirical Methods in Natural Language Processing (EMNLP)",
    month = nov,
    year = "2020",
    address = "Online",
    publisher = "Association for Computational Linguistics",
    url = "https://www.aclweb.org/anthology/2020.emnlp-main.145",
    doi = "10.18653/v1/2020.emnlp-main.145",
    pages = "1846--1859"
}
</pre>



## Abstract
Video-grounded dialogues are very challenging due to (i) the complexity of videos which contain both spatial and temporal variations, and (ii) the complexity of user utterances which query different segments and/or different objects in videos over multiple dialogue turns. However, existing approaches to video-grounded dialogues often focus on superficial temporal-level visual cues, but neglect more fine-grained spatial signals from videos. To address this drawback, we proposed Bi-directional Spatio-Temporal Learning (BiST), a vision-language neural framework for high-resolution queries in videos based on textual cues. Specifically, our approach not only exploits both spatial and temporal-level information, but also learns dynamic information diffusion between the two feature spaces through spatial-to-temporal and temporal-to-spatial reasoning. The bidirectional strategy aims to tackle the evolving semantics of user queries in the dialogue setting. The retrieved visual cues are used as contextual information to construct relevant responses to the users. Our empirical results and comprehensive qualitative analysis show that BiST achieves competitive performance and generates reasonable responses on a large-scale AVSD benchmark. We also adapt our BiST models to the Video QA setting, and substantially outperform prior approaches on the TGIF-QA benchmark.

<p align="center">
<img src="img/sample_dials.png" width="70%" />
 <br>
Examples of video-grounded dialogues from the benchmark datasets of Audio-Visual Scene Aware Dialogues (AVSD) challenge. H: human, A: the dialogue agent.
</p>


## Model Architecture

<p align="center">
<img src="img/model.png" width="100%" />
<br>
Our bidirectional approach models the dependencies between text and vision in two reasoning directions: spatial→temporal and temporal→spatial. ⊗ and ⊕ denote dot-product operation and element-wise summation.
</p>

## Dataset

We use the TGIF-QA benchmark. Refer to the official benchmark repo [here](https://github.com/YunseokJANG/tgif-qa) to download the dataset. Thanks to the authors of TGIF-QA benchmark, we developed the data-related code based on their shared codebase.

To use the spatio-temporal features, we extracted the visual features from a [published](https://github.com/kenshohara/video-classification-3d-cnn-pytorch) pretrained ResNext-101 model. The extraction code is slightly changed to obtain the features right before average pooling along spatial regions. Due to the large size of the video datasets, we are unable to shared to extracted features. 

To extract the features, you can download TGIF videos following the instructions [here](https://github.com/YunseokJANG/tgif-qa/tree/master/dataset) to extract features by yourself. Please refer to  our modified code for feature extraction under the `video-classification-3d-cnn-pytorch` folder. An example running script is in the `run.sh` file in this folder. Videos are extracted by batches, specified by start and end index of video files.  

## Scripts 

We created `scripts/exec.sh` to prepare evaluation code, train models, generate dialogue response, and evaluating the generated responses with automatic metrics. You can directly run this file which includes example parameter setting: 

| Parameter           | Description                                                  | Values                                                       |
| :------------------ | :----------------------------------------------------------- | ------------------------------------------------------------ |
| **device** | device to specific which GPU to be used | e.g. 0, 1, 2, ...
| **stage**  | different value specifying different processes to be run  | 1: training stage, 2. generating stage, 3: evaluating stage |
| **task** | specify the TGIF-QA tasks, choosing from: Count, FrameQA, Action, and Trans |
| **test_mode** | test mode is on for debugging purpose. Set true to run with a small subset of data | true/false |
| **t2s**  | set 1 to use temporal-to-spatial attention operation                                  | 0, 1                                                  |
| **s2t** | set 1 to use spatial-to-temporal attention operation                                | 0, 1                                                      |
| **nb_workers**      | number of worker to preprocess data and create batches                                    | e.g. 4  |

An example to run `scripts/exec.sh` is shown in  `scripts/run.sh`. Please update the `fea_dir` in `scripts/exec.sh` to your local directory of the dialogue data/video features before running. 

Other model parameters can be also be set, either manually or changed as dynamic input, including but not limited to:

| Parameter           | Description                                                  | Values                                                       |
| :------------------ | :----------------------------------------------------------- | ------------------------------------------------------------ |
| **fea_dir** | the directory of extracted feature data  | `video-classification-3d-cnn-pytorch/output/`
| **d_model** | dimension of word embedding as well as transformer layer dimension | e |.g. 128
| **nb_blocks**  | number of response decoding attention layers | e.g. 3                                           |
| **nb_venc_blocks** | number of visual reasoning attention layers                               | e.g. 3                                                     |

Refer to `configs` folder for more definitions of other parameters which can be set through  `scripts/exec.sh`. 

While training, the model with the best validation is saved. The model is evaluated by using the losses from response generation as well as question auto-encoding generation. 
The model output, parameters, vocabulary, and training and validation logs will be save into folder determined in the `expdir` parameter.  

Examples of pretrained BiST models using different parameter settings through `scripts/run.sh` can be downloaded [here](https://drive.google.com/drive/folders/1uhO2YrI5oHNSWp1G3ViMmeAKC2KAsQ7H?usp=sharing). Unzip the download file and update the `expdir` parameter in the test command in the `scripts/test.sh` to the corresponding unzip directory. Using the pretrained model, the test script provides the following results:
|    Task    | Extracted Features |   Link   |  Loss |  Acc  |
|:----------:|:------------------:|:--------:|:-----:|:-----:|
| Count      | ResNet152          | [Download](https://drive.google.com/drive/folders/1A-xIkeFiXanQeXY0sEUfIoSxcUBHjoYO?usp=sharing) | 2.404 | 0.327 |
| Count      | ResNext101         | [Download](https://drive.google.com/drive/folders/1SKwPjWLayiQ2wsNlj34gHUodgJJARW8F?usp=sharing) | 2.194 | 0.304 |
| Action     | ResNet152          | [Download](https://drive.google.com/drive/folders/1wnLbw_ZIxfKq9MZBKBqijPC1q5-5FXLi?usp=sharing) | 0.161 | 0.839 |
| Action     | ResNext101         | [Download](https://drive.google.com/drive/folders/1A4gTLcekSIXh3ut--0Z3tpNkuAaMNOqA?usp=sharing) | 0.136 | 0.847 |
| Transition | ResNet152          | [Download](https://drive.google.com/drive/folders/1HTap_nE4CEMXV9npwl9Iy531Tdgthi4D?usp=sharing) | 0.185 | 0.818 |
| Transition | ResNext101         | [Download](https://drive.google.com/drive/folders/14NMiE6fDJv9rjEhXVSp82U15u7qhTRGv?usp=sharing) | 0.178 | 0.819 |
| FrameQA    | ResNet152          | [Download](https://drive.google.com/drive/folders/1KnwZ5RHH6o83W7XRVxDZpCd5Sh_LXabn?usp=sharing) | 1.659 | 0.630 |
| FrameQA    | ResNext101         | [Download](https://drive.google.com/drive/folders/1nmF3k5lnKL-usU58sQu2GvFdTyIiQnZZ?usp=sharing) | 1.607 | 0.648 |



