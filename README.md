# CaSLearn
## 1. Introduction
This Library is a supplementary material for paper(**paper**).

CaSLearn simulation platform for PC algorithm in causal structure learning research. In this platform, researcher can easily
take experiments on their conditional independence tests(CITs), according to the following steps.

Currently, we support 14 CITs. There are *fisherz*, *spearman*, *kendall*, *robustQn*, *kci*, *gcm*, *wgcm*, *classifier*,
*lp*, *knn*, *gan*, *dgan*, *diffusion*. Please refer to our paper(**paper**) for more information of these methods. 

## 2. Installation required package

- Python 3.11
- numpy 
- pandas
- networkx
- causal-learn
- scipy
- cdt
- pyyaml
- scikit-learn
- tqdm

Optional requirements:
- xgboost
- [ccit](https://github.com/rajatsen91/CCIT)
- torch (for cit based on deep learning model)
- robustbase
- openpyxl
- matplotlib
- rpy2(for cit that based on R)

There are some CITs' code realization is on R. If you are interested in these method, you can install their package on R,
and perform PC algorithm on our platform.

## 3. Start Experiments



## 4. Benchmark Results
Here we have the simulation results in folder *result*. In *result*, there are 24 seperated folders which contains simulation
results for each CIT(this can be easily inferred by folder name). And there are five xlsx file that is the summary of our 
simulation study.

1. **summary.xlsx** --> Containing merged results of replicates via calculating average and standard error of considered metrics.
2. **summary_10.xlsx** or **summary_50.xlsx**  --> Dividing summary.xlsx in terms of number of variables.
3. **raw_summary_10.xlsx** or **raw_summary_50.xlsx** --> Raw simulation results without merging.  

## 5. Visualization



## Acknowledgement

This is platform is build on the basis of Python Package [causal-learn](https://github.com/py-why/causal-learn), which inspired
me update our code and make some extension functions. 

## Citation
Please cite as:
```
@article{}
```
