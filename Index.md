# Definition for AGI Hallucination
## Image-Text Hallucination

| Year | Source | Name | Author | Content |
| :- | :-: | :- | :- | :- |
| 2023 | ArXiv | [Woodpecker: Hallucination Correction for Multimodal Large Language Models](https://arxiv.org/pdf/2310.16045.pdf) | Shukang Yin ||
| 2023 | ArXiv | [ANALYZING AND MITIGATING OBJECT HALLUCINATION IN LARGE VISION-LANGUAGE MODELS](https://arxiv.org/pdf/2310.00754.pdf) | Yiyang Zhou ||
| 2019 | ArXiv | [Object Hallucination in Image Captioning](https://arxiv.org/pdf/1809.02156.pdf) | Anna Rohrbach ||

## Video-Text Hallucination

| Year | Source | Name | Author | Content |
| :- | :-: | :- | :- | :- |
| 2022 | ACCV | [Thinking Hallucination for Video Captioning](https://arxiv.org/pdf/2209.13853.pdf) | Nasib Ullah | 缓解Object Hallucination和Action Hallucination，提出一个 new metric COAHA 来综合评估这两种幻觉的程度 |
| 2022 | ArXiv | [Audio-visual video face hallucination with frequency supervision and cross modality support by speech based lip reading loss](https://arxiv.org/pdf/2211.10883.pdf) | Shailza Sharma ||
| 2022 | ArXiv | [Efficient Human Vision Inspired Action Recognition Using Adaptive Spatiotemporal Sampling](https://ieeexplore.ieee.org/stamp/stamp.jsp?tp=&arnumber=10236596) | Khoi-Nguyen C. Mac ||




## 3D Hallucination

| Year | Source | Name | Author | Content |
| :- | :-: | :- | :- | :- |
| 2022 | ACCV | [PoseTriplet: Co-evolving 3D Human Pose Estimation, Imitation, and Hallucination under Self-supervision](https://arxiv.org/pdf/2203.15625.pdf) | Kehong Gong ||
| 2014 | IEEE | [3D Face Hallucination from a Single Depth Frame](https://ieeexplore.ieee.org/stamp/stamp.jsp?tp=&arnumber=7035806) | Shu Liang ||
| 2022 | AAAI | [Texture Generation Using Dual-Domain Feature Flow with Multi-View Hallucinations](https://ojs.aaai.org/index.php/AAAI/article/view/19895) | Seunggyu Chang ||



## Audio Hallucination

| Year | Source | Name | Author | Content |
| :- | :-: | :- | :- | :- |
| 2022 | IEEE | [Hallucination of Speech Recognition Errors With Sequence to Sequence Learning](https://ieeexplore.ieee.org/stamp/stamp.jsp?tp=&arnumber=9693404) | Prashant Serai ||
| 2023 | ArXiv | [PARAMETER EFFICIENT AUDIO CAPTIONING WITH FAITHFUL GUIDANCE USING AUDIO-TEXT SHARED LATENT REPRESENTATION](https://arxiv.org/pdf/2309.03340.pdf) | Arvind Krishna Sridhar ||
| 2023 | ArXiv | [Factual Consistency Oriented Speech Recognition](https://arxiv.org/pdf/2302.12369.pdf) | Naoyuki Kanda ||
| 2023 | ArXiv | [LP-MusicCaps: LLM-BASED PSEUDO MUSIC CAPTIONING](https://arxiv.org/pdf/2307.16372.pdf) | SeungHeon Doh ||


## Language Hallucination

| Year | Source | Name | Author | Content |
| :- | :-: | :- | :- | :- |
| 2023 | ArXiv | [Siren's Song in the AI Ocean: A Survey on Hallucination in Large Language Models](https://arxiv.org/pdf/2309.01219.pdf) | Yue Zhang ||

## Robotic & Agent Hallucination
| Year | Source | Name | Author | Content |
| :- | :-: | :- | :- | :- |
| 2023 | ArXiv | [Learning Perceptual Hallucination for Multi-Robot Navigation in Narrow Hallways](https://ieeexplore.ieee.org/stamp/stamp.jsp?tp=&arnumber=10161327) | Jin-Soo Park ||
| 2023 | ArXiv | [Robots That Ask For Help: Uncertainty Alignment for Large Language Model Planners](https://arxiv.org/pdf/2307.01928.pdf) | Allen Z. Ren ||
| 2021 | ArXiv | [Toward Agile Maneuvers in Highly Constrained Spaces: Learning from Hallucination](https://arxiv.org/pdf/2007.14479.pdf) | Xuesu Xiao ||


# Emergence for AGI Hallucination

# Solution for AGI Hallucination
## Video-Text Hallucination

| Year | Source | Name | Author | Content |
| :- | :-: | :- | :- | :- |
| 2023 | ArXiv | [Videochat: Chat-centric video understanding.](https://arxiv.org/pdf/2305.06355.pdf) | KunChang Li | 他们使用Detailed Video Descriptions来减少幻觉，而且引入spatiotemporal reasoning, event localization, 和 causal relationship来丰富video-text语义表达，为未来的研究设定了标准。 |
| 2023 | ArXiv | [Chat-UniVi: Unified Visual Representation Empowers Large Language Models with Image and Video Understanding](https://arxiv.org/pdf/2311.08046.pdf) | Peng Jin |  首次融合了images和 videos，提出动态visual tokens的概念，使用 high-level 语义特征和low level的视觉细节特征，提出一种Unified方法 |
| 2023 | ArXiv | [Deficiency-Aware Masked Transformer for Video Inpainting](https://arxiv.org/pdf/2307.08629.pdf) | Yongsheng Yu | introduce a dual-modality-compatible inpainting framework called Deficiency-aware Masked Transformer (DMT)，they pretrain a image inpainting model DMTimg serve as a prior for distilling the video model DMTvid, thereby benefiting the hallucination of deficiency cases. |
| 2023 | ArXiv | [Video-LLaMA An Instruction-tuned Audio-Visual Language Model for Video Understanding](https://arxiv.org/pdf/2306.02858.pdf) | Hang Zhang | 本文提出一个 Video Q-former来更好的处理video时序中的理解不一致问题，并使用Audio Q-former进一步捕获音频特征，通过adapter使得多模态和自然语言融合，有效的缓解了幻觉的问题。 |
| 2023 | ArXiv | [Unified Model for Image, Video, Audio and Language Tasks](https://arxiv.org/pdf/2307.16184.pdf) | Mustafa Shukor | This model efficiently pretrained on many tasks, based on task balancing and multimodal curriculum learning and they propose a novel study on multimodal model merging via weight interpolation of models trained on different multimodal tasks, showing their benefits in particular for out-ofdistribution generalization |
| 2022 | ArXiv | [Information-Theoretic Text Hallucination Reduction for Video-grounded Dialogue](https://arxiv.org/pdf/2212.05765.pdf) | Sunjae Yoon | 提出了一种 THR 正则损失来减轻幻觉，减轻了 feature-level 幻觉影响，本质上是减少mutual information，从文本特征和图像特征信息层面。|




# Evaluation for AGI Hallucination

## MLLMs
| Year | Source | Name | Author | Content |
| :- | :-: | :- | :- | :- |
| 2023 | ArXiv | [A Multitask, Multilingual, Multimodal Evaluation of ChatGPT on Reasoning, Hallucination, and Interactivity](https://arxiv.org/pdf/2302.04023.pdf) | Yejin Bang ||
| 2023 | ArXiv | [AMBER: An LLM-free Multi-dimensional Benchmark for MLLMs Hallucination Evaluation](https://arxiv.org/pdf/2311.07397.pdf) | Junyang Wang ||

## Video-Text Hallucination
| Year | Source | Name | Author | Content |
| :- | :-: | :- | :- | :- |
| 2023 | ArXiv | [Models See Hallucinations: Evaluating the Factuality in Video Captioning](https://arxiv.org/pdf/2303.02961.pdf) | Hui Liu ||
| 2023 | ArXiv | [VIDEO-CSR: COMPLEX VIDEO DIGEST CREATION FOR VISUAL-LANGUAGE MODELS](https://arxiv.org/pdf/2310.05060.pdf) | Tingkai Liu ||


# Discourse for AGI Hallucination
Hallucinations are not always entirely negative phenomena. To a certain extent, they reflect the creativity inherent in the model. We should embrace hallucinations, striving to minimize those that are unequivocally erroneous.
## Video-Text Hallucination
| Year | Source | Name | Author | Content |
| :- | :-: | :- | :- | :- |
| 2023 | ArXiv | [Putting People in Their Place: Affordance-Aware Human Insertion into Scenes](https://arxiv.org/pdf/2304.14406.pdf) | Sumith Kulal | 本文提出使用人物插入场景的方法，使得模型可以产生人物幻觉和场景幻觉，使得构图协调，又富有创造力。 |
| 2023 | ArXiv | [Multi-Object Tracking with Hallucinated and Unlabeled Videos](https://arxiv.org/pdf/2108.08836.pdf) | Daniel McKee |  |



