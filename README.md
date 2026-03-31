# Kale-Linear

Knowledge-aware learning linear methods.
 <!-- and tensor (multi-linear) regression -->

This package contains implementations of the following methods:

- Transformers (learning feature embeddings):
  - Multilinear Principal component analysis (MPCA), Lu et al., 2008, available at [_IEEE_](https://ieeexplore.ieee.org/abstract/document/4359192) or [_Open Access_](https://ieeexplore.ieee.org/abstract/document/4359192).
  - Domain adaptation via transfer component analysis (TCA) [[Pan et al., 2009]](http://www.aaai.org/ocs/index.php/IJCAI/IJCAI-09/paper/download/294/962).
  - Transfer Feature Learning with Joint Distribution Adaptation (JDA) [[Long et al., 2013]](http://openaccess.thecvf.com/content_iccv_2013/papers/Long_Transfer_Feature_Learning_2013_ICCV_paper.pdf). <!-- [[Matlab Code by Author]](http://ise.thss.tsinghua.edu.cn/~mlong/doc/joint-distribution-adaptation-iccv13.zip). -->
  - Balanced distribution adaptation (BDA) [[Wang et al., 2017]](http://jd92.wang/assets/files/a08_icdm17.pdf).
  - Maximum Independence Domain Adaptation (MIDA) [[Yan et al., 2017]](https://ieeexplore.ieee.org/stamp/stamp.jsp?tp=&arnumber=7815350).
- Estimators (learning classifiers):
  - Manifold Regularisation Learning Framework (LapSVM, LapRLS) [[Belkin et al., 2006]](http://www.jmlr.org/papers/v7/belkin06a.html).
  - Adaptation Regularisation Learning Framework (ARSVM, ARRLS) [[Long et al., 2014]](https://ieeexplore.ieee.org/stamp/stamp.jsp?tp=&arnumber=6550016).
  - Covariate Independence Regularised Learning Framework (CoIRSVM, CoIRLS) [[Zhou et al., 2020](https://aaai.org/ojs/index.php/AAAI/article/view/6179), [Zhou, 2022](https://etheses.whiterose.ac.uk/id/eprint/31044/)].
  - Group-specific Discriminant Analysis (GSDA) [[Zhou et al., 2025](https://academic.oup.com/gigascience/article/doi/10.1093/gigascience/giaf082/8244707), [Zhou, 2022](https://etheses.whiterose.ac.uk/id/eprint/31044/)].

## Dependencies

- [scikit-learn](http://scikit-learn.org/)
- [scipy](https://www.scipy.org/)
- [numpy](http://www.numpy.org/)
- [cvxopt](http://cvxopt.org/)
- [osqp](https://osqp.org/)
- [tensorly](http://tensorly.org/)

## Installation

Install the NumPy-based core package:

```bash
pip install kalelinear
```

Install optional PyTorch tensor interoperability support:

```bash
pip install "kalelinear[torch]"
```

The library computes with NumPy and scikit-learn internally. Installing `torch` lets you pass `torch.Tensor`
inputs and receive `torch.Tensor` outputs, but it does not switch the numerical kernels to native PyTorch.

<!--
### Scikit-learn Style Implementation

##### Learning low-dimensional embedding for input data `X`
```
From kalelinear.transformer.tca import TCA

transformer = TCA(n_components=2)
X_transformed = transformer.fit_transform(X)
```

##### Training classifier for labelled source data (`Xs`, `ys`), and unlabelled target data `Xt`.

Example 1: Using Manifold Regularisation Learning Framework
```
import numpy as np
From kalelinear.estimator.manifold_learn import LapSVM

clf = LapSVM()
clf.fit(np.concatnate((Xs, Xt)), ys)
y_pred = clf.predict(Xt)
```

Example 2: Using Adaptation Regularisation Learning Framework
```
From kalelinear.estimator.artl import ARSVM

clf = ARSVM()
clf.fit(Xs, ys, Xt)
y_pred = clf.predict(Xt)
```

Example 3: Using Side Information Dependence Regularisation Learning Framework
```
From kalelinear.estimator import CoIRSVM

ns = Xs.shape[0]
nt = Xt.shape[0]
D = np.zeros((ns+nt, 2))  # Domain Covariates Matrix
D[:ns, 0] = 1
D[ns:, 1] = 1

clf = CoIRSVM()
clf.fit(np.concatnate((Xs, Xt)), ys, D)
y_pred = clf.predict(Xt)
```
 -->

<!-- ### References
-->
<!-- - Visual domain adaptation via transfer feature learning (VDA). Tahmoresnezhad, J. and Hashemi, S., 2017. Knowledge and Information Systems, 50(2), pp.585-605.
- Cross-domain video concept detection using adaptive svms. Yang, J., Yan, R., & Hauptmann, A. G. (2007, September). In Proceedings of the 15th ACM international conference on Multimedia (pp. 188-197). ACM.
- Cross-domain learning methods for high-level visual concept classification.Jiang, W., Zavesky, E., Chang, S.-F., and Loui, A.  In Image Processing, ICIP, 2008. 15th IEEE International Conference on (2008), IEEE, pp. 161-164.
- Song, X. and Lu, H., 2017, February. Multilinear regression for embedded feature selection with application to fMRI analysis. In     Thirty-First AAAI Conference on Artificial Intelligence (AAAI2017). -->

## Other Transfer Learning/Domain Adaptation Repos on GitHub

- [POT: Python Optimal Transport](https://github.com/rflamary/POT)
- [Everything about Transfer Learning](https://github.com/jindongwang/transferlearning)
- [ADA: (Yet) Another Domain Adaptation library](https://github.com/criteo-research/pytorch-ada)
- [Domain Adaptation & Transfer Learning Repositories](https://github.com/domainadaptation)
- [Library of transfer learners and domain-adaptive classifiers](https://github.com/wmkouw/libTLDA)
- [domain-adaptation-toolbox](https://github.com/viggin/domain-adaptation-toolbox)
- [Domain-Adaptations](https://github.com/wihoho/Domain-Adaptations)
