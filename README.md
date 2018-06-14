# ovnlp

A toolkit to download, train, use fastText word vectors on text data.
Also lets you deduplicate data based on TF IDF representation (see txtMatcher)
Developed under MIT license by Openvalue : http://openvalue.co


## Fasttext
 - For more info on fasttext, see :
    - https://fasttext.cc/
    - https://github.com/facebookresearch/fastText/
    - https://arxiv.org/abs/1607.04606
- This lib uses gensim's implementation of fastText.

## Installation

OVNLP runs on Python 3.6 ONLY.

Just run

    > pip install ovnlp

## Usage 

See demo_notebook.ipynb for usage examples

## FT Weights source

Pretrained weights from FB :
 - trained on crawl : https://s3-us-west-1.amazonaws.com/fasttext-vectors/word-vectors-v2/cc.fr.300.bin.gz
 
Feel free to change weightsource.json to add data sources if needed.
