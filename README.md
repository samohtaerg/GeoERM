# GeoERM: Geometry-Aware Multi-Task Representation Learning on Riemannian Manifolds

Implementation details for the experiments in the paper **"GeoERM: Geometry-Aware Multi-Task Representation Learning on Riemannian Manifolds"** by Aoran Chen and Yang Feng, Department of Biostatistics, New York University.

> **Abstract.** Multi-Task Learning (MTL) seeks to boost statistical power and learning efficiency by discovering structure shared across related tasks. State-of-the-art MTL representation methods, however, usually treat the latent representation matrix as a point in ordinary Euclidean space, ignoring its often non-Euclidean geometry, thus sacrificing robustness when tasks are heterogeneous or even adversarial. We propose GeoERM, a geometry-aware MTL framework that embeds the shared representation on its natural Riemannian manifold and optimizes it via explicit manifold operations. Each training cycle performs (i) a Riemannian gradient step that respects the intrinsic curvature of the search space, followed by (ii) an efficient polar retraction to remain on the manifold, guaranteeing geometric fidelity at every iteration. The procedure applies to a broad class of matrix-factorized MTL models and substantially reduces runtime compared to existing MTL frameworks, while maintaining competitive per-iteration cost. Across a set of synthetic experiments with task heterogeneity and two real datasets (wearable-sensor activity-recognition and MNIST with corruption), GeoERM consistently improves estimation accuracy, reduces negative transfer, and remains stable under adversarial label noise, outperforming leading MTL and single-task alternatives.

---

## Requirements

The HPC experiments were run on Python 3.9.21 with the following dependencies:

```
numpy==2.0.2
scikit-learn==1.6.1
scipy==1.13.1
torch==2.8.0
```

For the LibMTL experiments (Google Colab), `LibMTL` is installed automatically by the script.

---

## Implementation of Different MTL Methods

- **GeoERM** (Algorithm 1 in our paper): our method, implemented in `geoerm_har.py` and `geoerm_mnist_pairwise.py`. Integrates gradient projection, polar retraction, and loss minimization on the Stiefel manifold.
- **Penalized ERM (pERM)** (Tian et al., 2025): implemented using the public code from [https://github.com/ytstat/RL-MTL-TL](https://github.com/ytstat/RL-MTL-TL), included in `Benchmarks/mtl_func_torch.py`.
- **Method-of-Moments (MoM)** (Tripuraneni et al., 2021): implemented in `Benchmarks/mtl_func_torch.py` with default parameters.
- **ERM** (Du et al., 2020; Tripuraneni et al., 2021): implemented in `Benchmarks/mtl_func_torch.py`.
- **Pooled Regression** (Ben-David et al., 2010; Crammer et al., 2008): implemented in `Benchmarks/mtl_func_torch.py` using `sklearn.linear_model`.
- **Single-Task Regression**: implemented in `Benchmarks/mtl_func_torch.py`. Estimates each task independently without sharing information across tasks. In the logistic setting, uses sklearn's default penalized logistic regression with within-task ridge-type shrinkage.
- **AdaptRep** (Chua et al., 2021): implemented using the code from their paper, included in the `Benchmarks/` folder, with default parameters.
- **ARMUL** (Duan and Wang, 2023): implemented using the public code from [https://github.com/kw2934/ARMUL](https://github.com/kw2934/ARMUL), included in the `Benchmarks/ARMUL/` folder, with default settings.
- **HPS+EW and HPS+PCGrad**: implemented using the [LibMTL](https://github.com/median-research-group/LibMTL) library (Lin and Zhang, 2023), with PCGrad (Yu et al., 2020) for HPS+PCGrad.
- We used `PyTorch` (Paszke et al., 2019) to implement GeoERM, pERM, and ERM. Please make sure `PyTorch` is correctly installed before running.

---

## Reproducing the Experiments

### Experiment 1: HAR (Human Activity Recognition)

- Ensure `Benchmarks/mtl_func_torch.py` and `funcs.py` are available in your working directory (both included in this repository).
- Download the UCI HAR Dataset from the [UCI Machine Learning Repository](https://archive.ics.uci.edu/dataset/240/human+activity+recognition+using+smartphones) and place it under `Real Data/UCI HAR Dataset/`.
- Data preprocessing: run `Real Data/Real Data 1 HAR/HAR data preprocessing/har_preprocessing.py` on your local machine or HPC, which produces `har_standardized.pkl`. Edit the `USER CONFIGURATION` paths at the top of the script before running.
- Experiment: run `Real Data/Real Data 1 HAR/geoerm_har.py --r <rank>` with random seed 0–99 (via `$SLURM_ARRAY_TASK_ID`) and rank values `r = 5, 10, 15`. Edit the `USER CONFIGURATION` paths at the top of the script before running.
- HPS+EW and HPS+PCGrad: run `Real Data/Real Data 1 HAR/geoerm_har_libmtl.py` on Google Colab with a T4 GPU. Upload `har_standardized.pkl` to Google Drive and update `DATA_PATH` and `OUTPUT_DIR` in the `__main__` block before running. The script runs both EW and PCGrad methods across seeds 0–99 automatically.

### Experiment 2: MNIST Pairwise

- Ensure `Benchmarks/mtl_func_torch.py` and `funcs.py` are available in your working directory (both included in this repository).
- Download the MNIST-C dataset (Mu and Gilmer, 2019) from [GitHub](https://github.com/google-research/mnist-c) and place it in a local folder.
- Data preprocessing: run one of the following scripts to generate the corresponding `.pkl` file. Edit the `USER CONFIGURATION` paths at the top of each script before running.
  - `Real Data/Real Data 2 MNIST-C/MNIST-C data preprocessing/mnist_preprocessing_45t.py` → `mnist_pairwise_50_45t.pkl` (45 tasks, identity only)
  - `Real Data/Real Data 2 MNIST-C/MNIST-C data preprocessing/mnist_preprocessing_90t.py` → `mnist_pairwise_50_90t.pkl` (90 tasks, identity + brightness)
  - `Real Data/Real Data 2 MNIST-C/MNIST-C data preprocessing/mnist_preprocessing_92t_adv.py` → `mnist_pairwise_50_92t_adv.pkl` (92 tasks, 90 + 2 adversarial)
- Experiment: run `Real Data/Real Data 2 MNIST-C/geoerm_mnist_pairwise.py --dataset <dataset> --r <rank>` with random seed 0–99 (via `$SLURM_ARRAY_TASK_ID`), dataset values `45t, 90t, 92t_adv`, and rank values `r = 10, 15, 20`. Edit the `USER CONFIGURATION` paths at the top of the script before running.
- HPS+EW and HPS+PCGrad: run the corresponding Colab script on Google Colab with a T4 GPU. Upload the relevant `.pkl` file to Google Drive and update `DATA_PATH` and `OUTPUT_DIR` in the `__main__` block before running. Each script runs both EW and PCGrad methods across seeds 0–99 automatically.
  - `Real Data/Real Data 2 MNIST-C/geoerm_mnist_45t_libmtl.py` → dataset `45t`
  - `Real Data/Real Data 2 MNIST-C/geoerm_mnist_90t_libmtl.py` → dataset `90t`
  - `Real Data/Real Data 2 MNIST-C/geoerm_mnist_92t_libmtl.py` → dataset `92t_adv`

---

## References

- Anguita, D., Ghio, A., Oneto, L., Parra, X., and Reyes-Ortiz, J. L. (2013). A public domain dataset for human activity recognition using smartphones. In *ESANN*.
- Ben-David, S., Blitzer, J., Crammer, K., Kulesza, A., Pereira, F., and Vaughan, J. W. (2010). A theory of learning from different domains. *Machine Learning*, 79, 151–175.
- Chua, K., Lei, Q., and Lee, J. D. (2021). How fine-tuning allows for effective meta-learning. *Advances in Neural Information Processing Systems*, 34, 8871–8884.
- Crammer, K., Kearns, M., and Wortman, J. (2008). Learning from multiple sources. *Journal of Machine Learning Research*, 9(8).
- Du, S. S., Hu, W., Kakade, S. M., Lee, J. D., and Lei, Q. (2020). Few-shot learning via learning the representation, provably. *arXiv preprint arXiv:2002.09434*.
- Duan, Y. and Wang, K. (2023). Adaptive and robust multi-task learning. *The Annals of Statistics*, 51(5), 2015–2039.
- Kingma, D. P. and Ba, J. (2015). Adam: A method for stochastic optimization. *International Conference on Learning Representations (ICLR)*.
- Lin, B. and Zhang, Y. (2023). LibMTL: A Python library for deep multi-task learning. *Journal of Machine Learning Research*, 24(209), 1–7.
- Mu, N. and Gilmer, J. (2019). MNIST-C: A robustness benchmark for computer vision. *arXiv preprint arXiv:1906.02337*.
- Paszke, A., Gross, S., Massa, F., et al. (2019). PyTorch: An imperative style, high-performance deep learning library. *Advances in Neural Information Processing Systems*, 32.
- Tian, Y., Gu, Y., and Feng, Y. (2025). Learning from similar linear representations: Adaptivity, minimaxity, and robustness. *Journal of Machine Learning Research*, 26(187), 1–125.
- Tripuraneni, N., Jin, C., and Jordan, M. (2021). Provable meta-learning of linear representations. In *International Conference on Machine Learning*, pp. 10434–10443. PMLR.
- Yu, T., Kumar, S., Gupta, A., Levine, S., Hausman, K., and Finn, C. (2020). Gradient surgery for multi-task learning. *Advances in Neural Information Processing Systems*, 33, 5824–5836.
