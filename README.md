![block](./images/title.gif)

# GaussianDreamer: Fast Generation from Text to 3D Gaussian Splatting with Point Cloud Priors
### [Project Page](https://taoranyi.com/gaussiandreamer/) | [Arxiv Paper](https://arxiv.org/abs/2310.08529)

[GaussianDreamer: Fast Generation from Text to 3D Gaussian Splatting with Point Cloud Priors](https://taoranyi.com/gaussiandreamer/)  

[Taoran Yi](https://github.com/taoranyi)<sup>1</sup>,
[Jiemin Fang](https://jaminfong.cn/)<sup>2</sup>,[Guanjun Wu](https://guanjunwu.github.io/)<sup>3</sup>,  [Lingxi Xie](http://lingxixie.com/)<sup>2</sup>, </br>[Xiaopeng Zhang](https://sites.google.com/site/zxphistory/)<sup>2</sup>,[Wenyu Liu](http://eic.hust.edu.cn/professor/liuwenyu/)<sup>1</sup>, [Qi Tian](https://scholar.google.com/citations?hl=en&user=61b6eYkAAAAJ)<sup>2</sup> , [Xinggang Wang](https://xinggangw.info/)<sup>✉1</sup>

<sup>1</sup>School of EIC, HUST &emsp;<sup>2</sup>Huawei Inc. &emsp; <sup>3</sup>School of CS, HUST &emsp; 

![block](./images/architecture.jpg)
In recent times, the generation of 3D assets from text prompts has shown impressive results. Both 2D and 3D diffusion models can generate decent 3D objects based on prompts. 3D diffusion models have good 3D consistency, but their quality and generalization are limited as trainable 3D data is expensive and hard to obtain. 2D diffusion models enjoy strong abilities of generalization and fine generation, but the 3D consistency is hard to guarantee. This paper attempts to bridge the power from the two types of diffusion models via the recent explicit and efficient 3D Gaussian splatting representation. A fast 3D generation framework, named as GaussianDreamer, is proposed, where the 3D diffusion model provides point cloud priors for initialization and the 2D diffusion model enriches the geometry and appearance. Operations of noisy point growing and color perturbation are introduced to enhance the initialized Gaussians. Our GaussianDreamer can generate a high-quality 3D instance within 25 minutes on one GPU, much faster than previous methods, while the generated instances can be directly rendered in real time.
![block](./images/reoutput.gif)

**Our code will be released at the end of October**
## Citation
If you find this repository/work helpful in your research, welcome to cite the paper and give a ⭐.
```
@article{GaussianDreamer,
        title={GaussianDreamer: Fast Generation from Text to 3D Gaussian Splatting with Point Cloud Priors},
        author={Taoran Yi and Jiemin Fang and Guanjun Wu and Lingxi Xie and Xiaopeng Zhang and Wenyu Liu and Qi Tian and Xinggang Wang},
        journal={arxiv:2310.08529},
        year={2023}
        }
```
