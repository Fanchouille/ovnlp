# ovnlp

A toolkit to download, train, use fastText word vectors on text data.
Developed under MIT license by Openvalue : http://openvalue.co


## Fasttext
 - For more info on fasttext, see :
    - https://fasttext.cc/
    - https://github.com/facebookresearch/fastText/
    - https://arxiv.org/abs/1607.04606
- This lib uses gensim's implementation of fastText.

## Installation

Just run

    > pip install ovnlp

## Usage 

See demo_notebook.ipynb for usage examples

## Weights source

Pretrained weights from FB :
 - trained on crawl : https://s3-us-west-1.amazonaws.com/fasttext-vectors/word-vectors-v2/cc.fr.300.bin.gz
 
Feel free to change weightsource.json to add data sources if needed.
