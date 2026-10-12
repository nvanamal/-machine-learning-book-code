# Machine Learning & Deep Learning — Book Code

**An Illustrated Guide with Python**  
**Dr. Nagendrababu Vanamala**

This companion contains **102 standalone Python examples across 34 topics**. Each online guide includes the code, a line-by-line explanation, the output recorded in the book, and practice notes. QR codes in the book open the matching guide.

## Run an example

Use Python 3.11 or newer. Create a virtual environment, install the dependencies, then run a selected script:

```bash
python -m venv .venv
# macOS/Linux:
source .venv/bin/activate
# Windows PowerShell instead:
# .venv\Scripts\Activate.ps1
python -m pip install -r requirements.txt
python examples/01_linear/example_a.py
```

The pinned versions reproduce the book's dependency versions. Results can still vary slightly across platforms and numerical libraries. Each example runs independently; no external dataset download is required.

## Examples by topic

| Topic | Example A | Example B | Example C |
| --- | --- | --- | --- |
| LINEAR REGRESSION | [Basic fit and held-out prediction](guides/01_linear/example_a.md) | [Polynomial regression for a curved target](guides/01_linear/example_b.md) | [OLS cross-validation on training data](guides/01_linear/example_c.md) |
| LOGISTIC REGRESSION | [Basic fit and held-out prediction](guides/02_logistic/example_a.md) | [Multiclass flower classification](guides/02_logistic/example_b.md) | [Training-only CV for binary classification](guides/02_logistic/example_c.md) |
| DECISION TREE | [Basic fit and held-out prediction](guides/03_tree/example_a.md) | [Numeric regression with a tree](guides/03_tree/example_b.md) | [Training-only classification CV](guides/03_tree/example_c.md) |
| K-NEAREST NEIGHBOURS (KNN) | [Basic fit and held-out prediction](guides/04_knn/example_a.md) | [Distance-weighted numeric regression](guides/04_knn/example_b.md) | [Multiclass nearest-neighbour prediction](guides/04_knn/example_c.md) |
| RANDOM FOREST | [Basic fit and held-out prediction](guides/05_forest/example_a.md) | [Forest regression on a numeric target](guides/05_forest/example_b.md) | [Training-only classification CV](guides/05_forest/example_c.md) |
| NAIVE BAYES | [Basic fit and held-out prediction](guides/06_bayes/example_a.md) | [Multinomial NB for text counts](guides/06_bayes/example_b.md) | [Bernoulli NB for binary features](guides/06_bayes/example_c.md) |
| SUPPORT VECTOR MACHINE (SVM) | [Basic fit and held-out prediction](guides/07_svm/example_a.md) | [Support-vector numeric regression](guides/07_svm/example_b.md) | [Linear SVM on multiclass inputs](guides/07_svm/example_c.md) |
| GRADIENT BOOSTING | [Basic fit and held-out prediction](guides/08_boost/example_a.md) | [Boosted-tree numeric regression](guides/08_boost/example_b.md) | [Histogram boosting with missing inputs](guides/08_boost/example_c.md) |
| RIDGE, LASSO & ELASTIC NET | [Basic fit and held-out prediction](guides/09_ridge/example_a.md) | [Lasso numeric regression](guides/09_ridge/example_b.md) | [Elastic Net combines two penalties](guides/09_ridge/example_c.md) |
| K-MEANS CLUSTERING | [Cluster and measure silhouette](guides/10_kmeans/example_a.md) | [Color grouping as image quantization](guides/10_kmeans/example_b.md) | [Compare candidate cluster counts](guides/10_kmeans/example_c.md) |
| PRINCIPAL COMPONENT ANALYSIS (PCA) | [Project onto two PCA components](guides/11_pca/example_a.md) | [Held-out reconstruction from two components](guides/11_pca/example_b.md) | [PCA inside a predictive validation pipeline](guides/11_pca/example_c.md) |
| SUPERVISED LEARNING | [Basic fit and held-out prediction](guides/12_supervised/example_a.md) | [Regression uses numeric labels](guides/12_supervised/example_b.md) | [Multiclass targets use a classifier](guides/12_supervised/example_c.md) |
| UNSUPERVISED LEARNING | [K-means discovers groups](guides/13_unsupervised/example_a.md) | [Density clustering with DBSCAN](guides/13_unsupervised/example_b.md) | [Unlabeled anomaly screening](guides/13_unsupervised/example_c.md) |
| REINFORCEMENT LEARNING | [Q-learning in a five-state corridor](guides/14_rl/example_a.md) | [SARSA: an on-policy corridor learner](guides/14_rl/example_b.md) | [Evaluate a fixed policy without exploration](guides/14_rl/example_c.md) |
| SEMI-SUPERVISED LEARNING | [Confidence-based self-training](guides/15_semi/example_a.md) | [Graph-based label spreading](guides/15_semi/example_b.md) | [Compare with a labeled-only baseline](guides/15_semi/example_c.md) |
| SELF-SUPERVISED LEARNING | [Masked-feature reconstruction](guides/16_self/example_a.md) | [Denoising reconstruction without manual labels](guides/16_self/example_b.md) | [Next-sample prediction from a raw time signal](guides/16_self/example_c.md) |
| SOFTMAX REGRESSION | [Stable softmax probabilities](guides/17_softmax/example_a.md) | [Cross-entropy and score gradient](guides/17_softmax/example_b.md) | [Train a multiclass linear model](guides/17_softmax/example_c.md) |
| MULTILAYER PERCEPTRON (MLP) | [Forward pass through a two-layer network](guides/18_mlp/example_a.md) | [One explicit backpropagation update](guides/18_mlp/example_b.md) | [Fit a small neural classifier](guides/18_mlp/example_c.md) |
| CONVOLUTIONAL NEURAL NETWORK (CNN) | [A 2-D filter on an image patch](guides/19_cnn/example_a.md) | [Max pooling downsamples a feature map](guides/19_cnn/example_b.md) | [A 1-D convolutional spectral feature](guides/19_cnn/example_c.md) |
| RESIDUAL NETWORKS (RESNET) | [Identity skip connection](guides/20_resnet/example_a.md) | [Projection aligns feature dimensions](guides/20_resnet/example_b.md) | [Identity path appears in the gradient](guides/20_resnet/example_c.md) |
| RECURRENT NEURAL NETWORK (RNN) | [Scalar recurrent state](guides/21_rnn/example_a.md) | [Batch sequence shapes](guides/21_rnn/example_b.md) | [Gradient propagation through time](guides/21_rnn/example_c.md) |
| LONG SHORT-TERM MEMORY (LSTM) | [One gated memory update](guides/22_lstm/example_a.md) | [Forget gates change memory retention](guides/22_lstm/example_b.md) | [A scalar LSTM across a sequence](guides/22_lstm/example_c.md) |
| GATED RECURRENT UNIT (GRU) | [Update-gate interpolation](guides/23_gru/example_a.md) | [Reset gate controls the candidate](guides/23_gru/example_b.md) | [A scalar GRU sequence](guides/23_gru/example_c.md) |
| ENCODER-DECODER / SEQ2SEQ | [Encode a sequence into a context](guides/24_seq2seq/example_a.md) | [Decode conditionally on a context](guides/24_seq2seq/example_b.md) | [Teacher forcing shifts targets by one position](guides/24_seq2seq/example_c.md) |
| ATTENTION MECHANISMS | [Scaled dot-product attention](guides/25_attention/example_a.md) | [Causal self-attention mask](guides/25_attention/example_b.md) | [Attention pooling retrieves similar keys](guides/25_attention/example_c.md) |
| TRANSFORMER ARCHITECTURE | [Sinusoidal positional features](guides/26_transformer/example_a.md) | [Layer normalization and a residual branch](guides/26_transformer/example_b.md) | [A miniature attention-plus-feedforward block](guides/26_transformer/example_c.md) |
| DROPOUT | [Inverted dropout during training](guides/27_dropout/example_a.md) | [Evaluation disables the training mask](guides/27_dropout/example_b.md) | [Expected activation is approximately preserved](guides/27_dropout/example_c.md) |
| BATCH NORMALIZATION | [Normalize each feature in a batch](guides/28_batchnorm/example_a.md) | [Learned affine scale and shift](guides/28_batchnorm/example_b.md) | [Running statistics for inference](guides/28_batchnorm/example_c.md) |
| SGD, MOMENTUM & ADAM | [Gradient descent on a quadratic](guides/29_optimizers/example_a.md) | [Momentum on the same quadratic](guides/29_optimizers/example_b.md) | [Adam with bias correction](guides/29_optimizers/example_c.md) |
| TRANSFER LEARNING / FINE-TUNING | [Train and freeze a source representation](guides/30_transfer/example_a.md) | [Reuse source hidden features for a new label task](guides/30_transfer/example_b.md) | [Fine-tune a transferred hidden weight matrix](guides/30_transfer/example_c.md) |
| GAUSSIAN PROCESS REGRESSION | [GP mean and standard deviation](guides/31_gp/example_a.md) | [Kernel length scale changes similarity](guides/31_gp/example_b.md) | [Held-out GP regression error](guides/31_gp/example_c.md) |
| HYPERPARAMETER OPTIMIZATION | [Grid search using training-only folds](guides/32_hpo/example_a.md) | [Random search over tree settings](guides/32_hpo/example_b.md) | [Validation curve for regularization](guides/32_hpo/example_c.md) |
| GENERATIVE ADVERSARIAL NETWORK (GAN) | [Discriminator probabilities](guides/33_gan/example_a.md) | [Train a one-dimensional location GAN](guides/33_gan/example_b.md) | [Generator loss differs from original minimax loss](guides/33_gan/example_c.md) |
| RECOMMENDER SYSTEMS | [Rank items by content similarity](guides/34_recommender/example_a.md) | [Latent user-item score matrix](guides/34_recommender/example_b.md) | [Train factors only on observed ratings](guides/34_recommender/example_c.md) |

## How to interpret the examples

Some neural-network examples isolate forward computations, gates, gradients or other components. They are teaching demonstrations, not complete pretrained or production models. Synthetic-data scores do not establish performance on medical, financial, materials or space-safety applications. Use suitable held-out data and baselines for your own tasks.

The explanations and example implementations were prepared for this book. Selected additional topics were informed by *Dive into Deep Learning* by Aston Zhang, Zachary C. Lipton, Mu Li and Alexander J. Smola. See the book's source notes and references for further reading.
