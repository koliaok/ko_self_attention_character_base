# A Structured Self-attentive Sentence Embedding(revised version for korea language)

Tensorflow Implementation of "[A Structured Self-attentive Sentence Embedding](https://arxiv.org/abs/1703.03130)" (ICLR 2017).

![image](https://user-images.githubusercontent.com/15166794/41864478-21cbf7c8-78e5-11e8-94d2-5aa035a65c8b.png)


## Usage

### Data
* sentimental dataset for trip review([data source](http://air.changwon.ac.kr/)).
* The only csv files
* Character base learning for using bi-lstm network 
* you can compare the ([basic sentimental classification model](https://github.com/hugman/deep_learning/tree/master/course/nlp/applications/sentiment_analysis)) and my model
* If you can use only character base model, you don't need pre-training word2vec data from korea version 
* I referenced this site
* My OS env is mac high sierra

### Train
* "[facebook research fast text kr_word2vec(text version)download](https://github.com/facebookresearch/fastText/blob/master/pretrained-vectors.md)" is used as pre-trained word2vec model.
* I was training word based model, but it is failed because so low accuracy for sentimental evaluation 
* So I used character based learning, I got a good performance than ([basic sentimental classification model](https://github.com/hugman/deep_learning/tree/master/course/nlp/applications/sentiment_analysis))
	

* **Train Example (with word2vec, If you want get word based model training):**
    ```bash
	$ python train.py --word2vec "wiki.ko.vec"
	```

### Evalutation
* You must give "**checkpoint_dir**" argument, path of checkpoint(trained neural model) file, like below example.
* If you don't want to visualize the attention, give option like `--visualize False`.

* **Evaluation Example:**
	```bash
	$ python eval.py --checkpoint_dir "--checkpoint_dir "runs/2018_07_25_18_24_59/checkpoints/"
	```


## Results
#### 1) Accuracy test data = 0.7867083
#### 2) Visualization of Self Attention for english language with word based model 
![viz](https://user-images.githubusercontent.com/15166794/41875853-1dea6f28-7907-11e8-94e9-398e2699aca5.png)

#### 3) Visualization of Self Attention for korea language with character based model 
![viz](https://github.com/koliaok/ko_self_attention_character_base/tree/master/self_attention_ko_experiment/character_result.png)


## Reference
* A Structured Self-attentive Sentence Embedding (ICLR 2017), Z Lin et al. [[paper]](https://arxiv.org/abs/1703.03130)
* flrngel's [Self-Attentive-tensorflow](https://github.com/roomylee/self-attention-tf) github repository
* ([basic sentimental classification model](https://github.com/hugman/deep_learning/tree/master/course/nlp/applications/sentiment_analysis))

## my thinking 
* The reason for the low accuracy of the word based model is due to the complexity of morpheme, spacing, and atypical sentence in Korean word.
* Therefore, the effect of learned word2vec is insignificant.
* On the other hand, the Attention Mechanism thinks word is better than character. As can be seen from the experimental results
