newMaLSTM model 
===============
This project is based on the Siamese LSTM network + attention mechanism + Manhattan distance to achieve the similarity calculation. 

The training data includes: 

(1) [Quora sentence data on Kaggle, about 400,000 groups, positive and negative sample ratio 1:1.7](https://www.kaggle.com/c/quora-question-pairs/data)

(2) Stack overflow data, about 350,000 groups 

(3) SemEval 2014 train data, about 5,000 groups

The first two files are too large, only the third one is uploaded here as an example

Reference:
----------
•	[Siamese Recurrent Architectures for Learning Sentence Similarity](http://www.mit.edu/~jonasm/info/MuellerThyagarajan_AAAI16.pdf)

•	[How to predict Quora Question Pairs using Siamese Manhattan LSTM](https://medium.com/mlreview/implementing-malstm-on-kaggles-quora-question-pairs-competition-8b31b0b16a07)

•	[It Takes Two To Tango: Modification of Siamese Long Short Term Memory Network with Attention Mechanism in Recognizing Argumentative Relations in Persuasive Essay](https://www.sciencedirect.com/science/article/pii/S1877050917320847)

Word embedding file:
--------------------
•	word2vec:	[GoogleNews-vectors-negative300.bin](https://github.com/mmihaltz/word2vec-GoogleNews-vectors/blob/master/GoogleNews-vectors-negative300.bin.gz)

•	fastText:	[wiki-news-300d-1M.vec.bin](https://fasttext.cc/docs/en/english-vectors.html)

Language model:
---------------
•	ULMFiT: [wikitext103](http://files.fast.ai/models/wt103)


Apply:
------

• Training

``` 
$ python3 newMaLSTM.py
$ type w2v or ft for choosing word embedding file
```


• Spliting data

``` 
$ python3 data_helper.py
```


• Find the most similar existing question for a new question
``` 
$ python3 test.py
```

