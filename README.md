## **A Multi-Affine Fusion Graph Convolutional Network for Aspect-based Sentiment Analysis**

Code and datasets of our paper: **A Multi-Affine Fusion Graph Convolutional Network for Aspect-based Sentiment Analysis**.


## Requirement

- Python 3.6.7
- PyTorch 1.2.0
- NumPy 1.17.2
- GloVe pre-trained word vectors:
  - Download pre-trained word vectors [here](https://github.com/stanfordnlp/GloVe#download-pre-trained-word-vectors).
  - Put [glove.840B.300d.txt](http://nlp.stanford.edu/data/wordvecs/glove.840B.300d.zip) into the `dataset/glove/` folder.
  
## Usage

Training the model:

```bash
python train.py --dataset [dataset]
```

Prepare vocabulary files for the dataset:

```bash
python prepare_vocab.py --dataset [dataset]
```

Evaluate trained model

```bash
python eval.py --model_dir [model_file path]
```

## Credits
The code and datasets in this repository are based on [DM-GCN_ABSA](https://github.com/pangsg/DM-GCN) .
**Dynamic and Multi-Channel Graph Convolutional Network for Aspect-Based Sentiment Analysis**.

I mainly changed the models folder, but also a small part of train.py.