# Low-Rank and Deep Plug-and-play Priors for Missing Traffic Data Imputation

[![MIT License](https://img.shields.io/badge/license-MIT-green.svg)](https://opensource.org/licenses/MIT)

> This is the code repository for our works on low-rank tensor completion for missing traffic data imputation: 
> 
> [1] Peng Chen, Fang Li, Deliang Wei, and Changhong Lu, "Spatiotemporal traffic data completion with truncated minimax-concave penalty," Transportation Research Part C: Emerging Technologies, vol. 164, July 2024, Art. no. 104657.
> 
> [2] Peng Chen, Fang Li, Deliang Wei, and Changhong Lu, "Low-Rank and Deep Plug-and-Play Priors for Missing Traffic Data Imputation," IEEE Transactions on Intelligent Transportation Systems (Early Access), pp. 1-17, Nov. 2024.

## Methodology

<p align="center">
    <img src="./assets/algorithm-general.png" alt="Description" width="800">
</p>

> The general procedure of our proposed methods is as follows:  Initially, we conduct each mode unfolding matrix of $`\mathcal{M}^t`$ to low-rank processing, supplemented by optional deep PnP processing; Subsequently, upon folding and weighted aggregation of the processed matrices, we correlate the resultant $`\widehat{\mathcal{M}}^t`$  with the observation tensor $\mathcal{Y}$ to derive the tensor $`\mathcal{M}^{t+1}`$ for the subsequent iteration.

## Preparation

### Data preparation

We provide four selected public traffic datasets in the shared folder [Google Drive](https://drive.google.com/drive/folders/1uyl5ZZ3EsFYT4h8ItnmtQKycJehoju1I?usp=drive_link). More related datasets can refer to the [`transdim`](https://github.com/xinychen/transdim/tree/master) project. Please download the datasets and put them in the `./datasets` folder. The overview of the provided datasets is as follows:

- **Hangzhou:** [Hangzhou metro passenger flow dataset](https://tianchi.aliyun.com/competition/entrance/231708/information). This dataset contains information on incoming passenger flow for 80 metro stations in Hangzhou, China. The data covers a period of 25 days, from January 1st to January 25th, 2019, with a 10-minute resolution. The time interval from 0:00 a.m. to 6:00 a.m., when there are no services, has been excluded. Only the remaining 108 time intervals of a day are considered. The dataset is presented as a tensor of size $'80 \times 25 \times 108'$ ($'80 \times 2700'$ in the form of a time series matrix).

- **Portland:** [Portland highway traffic volume dataset](https://portal.its.pdx.edu/home). This dataset comprises traffic volume data collected from highways in the Portland-Vancouver Metropolitan region in January 2021. It was obtained from 1156 loop detectors with a 15-minute resolution, resulting in 96 time intervals per day. The dataset is in the form of a tensor of size $'1156 \times 31 \times 96'$ ($'1156 \times 2976'$ in the form of a time series matrix).

- **Seattle:** [Seattle freeway traffic speed dataset](https://github.com/zhiyongc/Seattle-Loop-Data). This dataset contains information on the speed of freeway traffic in Seattle, USA. The data was collected from 323 loop detectors with a 5-minute resolution, resulting in 288 time intervals per day. The data is presented as a tensor of size $'323 \times 28 \times 288'$ ($'323 \times 8064'$ when presented as a time series matrix). 

- **PeMS:** [PeMS freeway traffic volume dataset](https://people.eecs.berkeley.edu/\~varaiya/papers\_ps.dir/PeMSTutorial.pdf). This dataset includes the traffic volume recorded by 228 loop detectors in District 7 of California, with a 5-minute time resolution. The data was collected over the weekdays of May and June in 2012 by Caltrans Performance Measurement System (PeMS). The data is in the form of a tensor of size $'228 \times 44 \times 288'$ ($'228 \times 12672'$ in the form of the time series matrix). 

Besides, we provide the pretrained parameter of the used light DRUNet in [Google Drive](https://drive.google.com/drive/folders/1gUcu8elif7ZiNhcoaDLScAs5ZM1wufa-?usp=drive_link) for the deep plug-and-play prior. Please download the parameter and put it in the `./drunet_light_params` folder.

### Environment preparation

The required packages are listed in the `requirements.txt` file. You can run the following shell command to create a new environment named `lrtc` and install the packages by running the following command:

```bash
conda create --name lrtc --file requirements.txt
```

> The default command installs the `CPU` version of PyTorch. For faster computation of deep PnP processing using NVIDIA's CUDA, install the `CUDA` version based on your hardware and system. Check the [official website](https://pytorch.org/get-started/locally/) for installation instructions.


## Cite Information

> If you find this repo useful for your research, please consider citing the papers:

```
@article{CHEN2024104657,
    title = {Spatiotemporal traffic data completion with truncated minimax-concave penalty},
    journal = {Transportation Research Part C: Emerging Technologies},
    volume = {164},
    pages = {104657},
    year = {2024},
    issn = {0968-090X},
    doi = {https://doi.org/10.1016/j.trc.2024.104657},
    url = {https://www.sciencedirect.com/science/article/pii/S0968090X24001785},
    author = {Peng Chen and Fang Li and Deliang Wei and Changhong Lu},
    publisher={Elsevier}
}

@ARTICLE{10756233,
  author={Chen, Peng and Li, Fang and Wei, Deliang and Lu, Changhong},
  journal={IEEE Transactions on Intelligent Transportation Systems}, 
  title={Low-Rank and Deep Plug-and-Play Priors for Missing Traffic Data Imputation}, 
  year={2025},
  volume={26},
  number={2},
  pages={2690-2706},
  doi={10.1109/TITS.2024.3493864}
}
```
