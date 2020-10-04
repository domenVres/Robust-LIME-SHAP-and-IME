# Robust-LIME-SHAP-and-IME

This is the implementation of methods gLIME, gSHAP and gIME for tabular data (see [[1]](#1) for description of the methods). Methods use three different data generators. Variational autoencoder using Monte Carlo dropout is implemented in Python and is located in folder Generators. The code was downloaded from [[5]](#5). rbfDataGen and treeEnsemble are not yet implemented in Python, so their R library semiArtificial[[6]](#6) implementation is used.

## Experiment

The code for experiment, conducted in [[1]](#1) can be found in folder Fooling-LIME-SHAP. The folder is a modified version of the original code from Slack [[4]](#4).

## gLIME

Method gLIME is implemented in file lime/lime_tabular.py. It is a modified version of method LIME from Python lime library [[2]](#2).

## gSHAP

Method gSHAP is implemented in file shap/explainers/kernel.py. It is a modified version of method Kernel SHAP [[8]](#8) from Python shap library [[3]](#3).

## gIME

Method gIME is implemented in file shap/explainers/sampling.py. It is a modified version of method IME [[7]](#7) included in Python shap library [[3]](#3).

## References
<a id="1">[1]</a>
Anonymous (2020)
Better sampling in explanation methods can prevent dieselgate-like deception
Submitted to International Conference on Learning Representations
https://openreview.net/forum?id=s0Chrsstpv2

<a id="2">[2]</a>
Ribeiro, M.
LIME
https://github.com/marcotcr/lime

<a id="3">[3]</a>
Lundberg, S.
SHAP
https://github.com/slundberg/shap

<a id="4">[4]</a>
Slack, D.
Fooling LIME and SHAP
https://github.com/dylan-slack/Fooling-LIME-SHAP

<a id="5">[5]</a>
Miok, K.
MCD-VAE
https://github.com/KristianMiok/MCD-VAE

<a id="6">[6]</a>
Robnik-Šikonja, M.
semiArtificial
https://CRAN.R-project.org/package=semiArtificial

<a id="7">[7]</a>
Štrumbelj, Erik and Kononenko, Igor (2013)
Explaining prediction models and individual predictions with feature contributions
Knowledge and Information Systems, 41, 647-665

<a id="7">[7]</a>
Štrumbelj, E. and Kononenko, I. (2013)
Explaining prediction models and individual predictions with feature contributions
Knowledge and Information Systems, 41, 647-665

<a id="8">[8]</a>
Lundberg, S. M. and Lee, S. (2017)
A Unified Approach to Interpreting Model Predictions
Advances in Neural Information Processing Systems 30, 4765-4774

<a id="9">[9]</a>
Ribeiro, M. T. and Singh, S. and Guestrin, C. (2016)
"Why Should I Trust You?": Explaining the Predictions of Any Classifier
Proceedings of the 22nd ACM SIGKDD International Conference on Knowledge Discovery and Data Mining, 1135-1144