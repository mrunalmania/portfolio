## Enhancing Text Summarization with NLP: A Dive into Gensim, NLTK, and SpaCy

<figure markdown="span">
  ![Sole Gemma](../images/TextSummarization.jpg){ width="300" }
  <figcaption>Just as a great melody captures the soul of a song, an effective summary distills the essence of a text, making complex information harmonious and accessible.</figcaption>
</figure>

### Introduction

In the age of information overload, the ability to distill large volumes of text into concise summaries is more valuable than ever. Whether it's summarizing articles, reports, or research papers, automated text summarization can save time and help users quickly grasp the essence of content. As part of my recent project, I explored various approaches to text summarization using some of the most popular Natural Language Processing (NLP) libraries: Gensim, NLTK, and SpaCy.

### GitHub Repository

For a detailed look at the code and to explore these summarization techniques further, visit my GitHub repository:

[GitHub Repository: Text Summarization with Gensim, NLTK, and SpaCy](https://github.com/mrunalmania/text-summarization-nlp)

Feel free to clone the repository, contribute, or reach out if you have any questions or feedback!


### Why Text Summarization?

Text summarization is a powerful tool in NLP that helps reduce lengthy documents into shorter, meaningful versions. This is particularly useful in contexts where time is of the essence, such as in news aggregation, content curation, and document review processes. Automated summarization not only enhances productivity but also ensures that critical information is not overlooked.

### Exploring Different Approaches

In this project, I implemented three distinct methods for text summarization:

#### 1. **Gensim Summarization**

Gensim's TextRank algorithm is a popular choice for unsupervised text summarization. Here's a brief look at the implementation:

```python
from gensim.summarization import summarize

def gensim_summarizer(raw_text):
    summary = summarize(raw_text, word_count=100)
    return summary
```

Gensim’s `summarize` function is straightforward to use and works well for extracting the most salient points from a text.

#### 2. **NLTK Summarization**

NLTK's approach involves word frequency analysis to score sentences. Below is the code for this method:

```python
import nltk
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize, sent_tokenize
import heapq  

def nltk_summarizer(raw_text):
    stopWords = set(stopwords.words("english"))
    word_frequencies = {}  
    for word in nltk.word_tokenize(raw_text):  
        if word not in stopWords:
            word_frequencies[word] = word_frequencies.get(word, 0) + 1

    maximum_frequency = max(word_frequencies.values())
    for word in word_frequencies.keys():  
        word_frequencies[word] /= maximum_frequency

    sentence_list = nltk.sent_tokenize(raw_text)
    sentence_scores = {}  
    for sent in sentence_list:  
        for word in nltk.word_tokenize(sent.lower()):
            if word in word_frequencies.keys():
                if len(sent.split(' ')) < 30:
                    sentence_scores[sent] = sentence_scores.get(sent, 0) + word_frequencies[word]

    summary_sentences = heapq.nlargest(7, sentence_scores, key=sentence_scores.get)
    summary = ' '.join(summary_sentences)  
    return summary
```

NLTK’s method focuses on identifying and scoring key sentences based on word frequency.

#### 3. **SpaCy Summarization**

SpaCy provides a powerful NLP pipeline, making it suitable for summarization. Here’s how you can use SpaCy for this task:

```python
from heapq import nlargest
from spacy.lang.en.stop_words import STOP_WORDS
import spacy
nlp = spacy.load("en_core_web_sm")

def text_summarizer(raw_text):
    doc = nlp(raw_text)
    stopwords = list(STOP_WORDS)
    word_frequencies = {}
    for word in doc:
        if word.text not in stopwords:
            word_frequencies[word.text] = word_frequencies.get(word.text, 0) + 1

    maximum_frequency = max(word_frequencies.values())
    for word in word_frequencies.keys():
        word_frequencies[word] /= maximum_frequency

    sentence_list = [sentence for sentence in doc.sents]
    sentence_scores = {}
    for sent in sentence_list:
        for word in sent:
            if word.text.lower() in word_frequencies.keys():
                if len(sent.text.split(' ')) < 30:
                    sentence_scores[sent] = sentence_scores.get(sent, 0) + word_frequencies[word.text.lower()]

    summarized_sentences = nlargest(7, sentence_scores, key=sentence_scores.get)
    final_sentences = [w.text for w in summarized_sentences]
    summary = ' '.join(final_sentences)
    return summary
```

SpaCy’s approach leverages its advanced NLP capabilities for more accurate sentence selection.

### Comparing the Approaches

Each of these methods has its strengths. Gensim’s TextRank is particularly effective for longer documents, providing a summary that captures the broader context. NLTK’s word frequency-based summarization is straightforward and works well for shorter texts. SpaCy, with its advanced NLP pipeline, offers a more nuanced approach to sentence selection, making it ideal for content that requires precise summarization.

### Challenges and Learning

Working on this project presented some interesting challenges. One of the key difficulties was ensuring that the summaries were not only concise but also coherent and meaningful. Balancing the length of the summary with the need to retain essential information required fine-tuning the algorithms and experimenting with different configurations.

Another challenge was handling different text structures. Some documents were dense with information, while others were more narrative. Each summarization method had to be adapted to handle these variations effectively.

### Conclusion

This project was an exciting dive into the world of text summarization, allowing me to experiment with and compare different NLP techniques. Each tool—Gensim, NLTK, and SpaCy—brought its unique strengths to the table, demonstrating the versatility of NLP in solving real-world problems.

By integrating these summarization techniques, I’ve built a versatile solution that can be adapted to various applications, from news summarization to academic research. This experience has deepened my understanding of NLP and has further fueled my passion for developing solutions that make data more accessible and actionable.

