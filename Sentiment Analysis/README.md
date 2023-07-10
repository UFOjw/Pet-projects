# Sentiment Analysis

For services that closely interact with customers, it is important to know how loyal they are. If the service is small, then everything is clear. But in the case when a person cannot cope, it is necessary to automate the process. That's what sentiment analysis is for.

Classification of the text according to its tonality has the following classes:
* positive
* negative
* neutral

Sentiment analysis can monitor public sentiment towards a brand or product in real-time. This helps businesses gauge their brand reputation, identify and respond to negative sentiment, and track customer perception over time.
By understanding the sentiment behind customer opinions, businesses can identify areas for improvement, make informed decisions, and provide better customer experiences.

This repository contains a baseline solution for this problem:
- First, the review dataset is tokenized, turning it into indices.
- Because the output word representations are of different lengths, multiprocessing is not possible. So the next step is to add `padding` (`normal`, `dynamic`).
- Technically, after applying padding, embeddings have empty extra tokens. In order for the model to not take into account extra tokens during processing, `attention_mask` is calculated and passed, which nulls the weight of empty tokens.
<img src="https://jalammar.github.io/images/distilBERT/bert-distilbert-tutorial-sentence-embedding.png" alt="drawing" width="1200"/>
- At the output of BERT, we have context-based word embeddings, and the first of them is a vector representation of a sentence. Using these ideas, we move on to the final stage - learning logistic regression over embeddings.
![2](https://jalammar.github.io/images/distilBERT/bert-distilbert-train-test-split-sentence-embedding.png)
![3](https://jalammar.github.io/images/distilBERT/bert-training-logistic-regression.png)
- To evaluate the results, the `evaluate` method calculates the cross-entropy for each fold when using cross-validation.

The lightweight and fast version of the `BERT-architecture`, `DistilBERT`, is used as a model. It has 40% less parameters than bert-base-uncased, runs 60% faster while preserving over 95% of BERT’s performances as measured on the GLUE language understanding benchmark.

The full pipeline is shown in the picture below.
![4](https://jalammar.github.io/images/distilBERT/bert-model-calssification-output-vector-cls.png)

All illustrations are taken from the [article](https://jalammar.github.io/a-visual-guide-to-using-bert-for-the-first-time/).
