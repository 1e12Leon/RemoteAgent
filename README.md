<div align="center">

# [RemoteAgent: Bridging Vague Human Intents and Earth Observation with RL-based Agentic MLLMs]()



[Liang Yao (姚亮)*](https://1e12leon.top/) 
<img src="assets/hhu_logo.png" alt="Logo" width="15">, &nbsp; &nbsp; 
[Shengxiang Xu(徐圣翔)*](https://xushengxianggg.github.io/) 
<img src="assets/HKUST.jpg" alt="Logo" width="15">, &nbsp; &nbsp;
[Fan Liu (刘凡)](https://multimodality.group/author/%E5%88%98%E5%87%A1/) ✉ 
<img src="assets/hhu_logo.png" alt="Logo" width="15">, &nbsp; &nbsp;
[Chuanyi Zhang (张传一)](https://ai.hhu.edu.cn/2023/0809/c17670a264073/page.htm) 
<img src="assets/hhu_logo.png" alt="Logo" width="15">, &nbsp; &nbsp;
[Bishun Yao (姚必顺)](https://multimodality.group/author/%E7%8E%8B%E7%BF%8C%E9%AA%8F/) 
<img src="assets/hhu_logo.png" alt="Logo" width="15">, &nbsp; &nbsp;

[Rui Min (闵锐)]() 
<img src="assets/hhu_logo.png" alt="Logo" width="15">, &nbsp; &nbsp;
[Yongjun Li (李勇俊)]() 
<img src="assets/hhu_logo.png" alt="Logo" width="15">, &nbsp; &nbsp;
[Chaoqian Ouyang(欧阳超前)]() 
<img src="assets/SYU.png" alt="Logo" width="15">, &nbsp; &nbsp; 
[Shimin Di (邸世民)](https://cs.seu.edu.cn/shimindi/main.htm) 
<img src="assets/SEU.png" alt="Logo" width="15">, &nbsp; &nbsp; 
[Min-Ling Zhang (张敏灵)]() 
<img src="assets/SEU.png" alt="Logo" width="15">

\*  *Equal Contribution*    ✉ *Corresponding Author*


</div>


## News
- **2026/4/9**: Welcome to RemoteAgent! The preprint of our paper is available. Dataset and codes will be open-sourced at this repository.



## Introduction


Earth Observation (EO) systems are essentially designed to support domain experts who often express their requirements through vague natural language rather than precise, machine-friendly instructions. Depending on the specific application scenario, these vague queries can demand vastly different levels of visual precision. Consequently, a practical EO AI system must bridge the gap between ambiguous human queries and the appropriate multi-granularity visual analysis tasks, ranging from holistic image interpretation to fine-grained pixel-wise predictions.
While Multi-modal Large Language Models (MLLMs) demonstrate strong semantic understanding, their text-based output format is inherently ill-suited for dense, precision-critical spatial predictions. Existing agentic frameworks address this limitation by delegating tasks to external tools, but indiscriminate tool invocation is computationally inefficient and underutilizes the MLLM's native capabilities.
To this end, we propose RemoteAgent, an agentic framework that strategically respects the intrinsic capability boundaries of MLLMs. To empower this framework to understand real user intents, we construct VagueEO, a human-centric instruction dataset pairing EO tasks with simulated vague natural-language queries. By leveraging VagueEO for reinforcement fine-tuning, we align an MLLM into a robust cognitive core that directly resolves image- and sparse region-level tasks. Consequently, RemoteAgent processes suitable tasks internally while intelligently orchestrating specialized tools via the Model Context Protocol exclusively for dense predictions. Extensive experiments demonstrate that RemoteAgent achieves robust intent recognition capabilities while delivering highly competitive performance across diverse EO tasks. 



## Acknowledge
- Code in this repository is built on [MS-SWIFT](https://github.com/modelscope/ms-swift). We'd like to thank the authors for open sourcing their project.

## Contact
Please Contact yaoliang@hhu.edu.cn


## Cite
If you find this work useful, please cite our papers as:
```bibtex
@misc{yao2025RemoteSAM,
      title={RemoteSAM: Towards Segment Anything for Earth Observation}, 
      author={Liang Yao and Fan Liu and Delong Chen and Chuanyi Zhang and Yijun Wang and Ziyun Chen and Wei Xu and Shimin Di and Yuhui Zheng},
      year={2025},
      eprint={2505.18022},
      archivePrefix={arXiv},
      primaryClass={cs.CV},
      url={https://arxiv.org/abs/2505.18022}, 
}
```