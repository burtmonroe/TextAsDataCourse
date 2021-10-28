# Tutorials / Notebooks / Code

Burt Monroe (Penn State)
Produced for Penn State and Essex Courses in "Text as Data"

## String Processing and Regular Expressions in R & Python

* Introduction to String Manipulation and Regular Expressions in R
    * Notebook html: [here](https://burtmonroe.github.io/TextAsDataCourse/Tutorials/TADA-IntroToTextManipulation.html)
    * Notebook .Rmd [here](https://burtmonroe.github.io/TextAsDataCourse/Tutorials/TADA-IntroToTextManipulation.Rmd)
    * Available on Essex RStudioCloud (Day 1 - Review project)
    
* Introduction to String Manipulation and Regular Expressions in Python
    * Colab notebook [here](https://colab.research.google.com/drive/1wCVf8xaoTAsKya5uuuo5knvizbWgheE_?usp=sharing)


## NLP / Text-as-Data Frameworks in R & Python

### In R

   * [Open Source Tools for Text as Data in R](https://burtmonroe.github.io/TextAsDataCourse/Notes/RText/)
   
   * Introduction to Text-as-Data with quanteda (R)
      * Notebook html: [here](https://burtmonroe.github.io/TextAsDataCourse/Tutorials/TADA-IntroToQuanteda.html)    
      * Notebook .Rmd: [here](https://burtmonroe.github.io/TextAsDataCourse/Tutorials/TADA-IntroToQuanteda.Rmd)
      * Available on Essex RStudioCloud (Day 1 - Review project)
      
   * (3rd party) Text as Data with tidytext: Great tutorials in the book: https://www.tidytextmining.com
   
   * NLP Pipelines in R (Primary focus is on UDPipe and openNLP, with spacyr separated out to other notebook)
      * Colab: Introduction to NLP Annotation Pipelines in R: [here](https://colab.research.google.com/drive/15UcuXNYuhR9wuHbKp4J1HUsOB6GOX6oV?usp=sharing)
      * Colab: Introduction to spacyr and spaCy through R: [here](https://colab.research.google.com/drive/1wrYUNqp--v7tA_umgqajoy9A0ugzGUfz?usp=sharing)
 


### In Python
 
   * [Open Source Tools for Text as Data in Python](https://burtmonroe.github.io/TextAsDataCourse/Notes/PythonText/)
   
   * Introduction to Bag of Words data with CountVectorizer (Python)
      * Colab: [here](https://colab.research.google.com/drive/1YQ-b7VmPBgpe9utqk_aDyY5KYbyRgXQm?usp=sharing)
       
   * Introduction to NLP Annotation Pipelines in Python [spaCy, Stanza/CoreNLP, NLTK, and Flair] (Python)
      * Colab: [here](https://colab.research.google.com/drive/1Us7Hx5xF5pdx-JM3t_6QB8SZZhHfrc0Q?usp=sharing)
      
   * (Old/minimal) Introduction to NLP with textblob, nltk, and pattern (Python)
      * Notebook html: [here](https://burtmonroe.github.io/TextAsDataCourse/Tutorials/Introduction%20to%20NLP%20with%20TextBlob%2C%20NLTK%2C%20and%20pattern.html)

   * (3rd party demos /guide) AllenNLP: Demos (https://demo.allennlp.org/) and Guide (https://guide.allennlp.org) (Python)

## Scraping and Data Wrangling:
        
* Scraping with rvest (R) (Example: United Nations meeting summaries)
   * Notebook nb.html: [here](https://burtmonroe.github.io/TextAsDataCourse/Tutorials/TADA-ScrapingWithRvest_UNMeetingSummaries.nb.html)
   * Notebook Rmd: [here](https://burtmonroe.github.io/TextAsDataCourse/Tutorials/TADA-ScrapingWithRvest_UNMeetingSummaries.Rmd)

* Scraping with RSelenium (R) (Example: UN Document Search)
   * Notebook nb.html: [here](https://burtmonroe.github.io/TextAsDataCourse/Tutorials/TADA-RSelenium.nb.html)
   * Notebook Rmd: [here](https://burtmonroe.github.io/TextAsDataCourse/Tutorials/TADA-RSelenium.Rmd)

* Scraping with Requests and BeautifulSoup (Python)

* Scraping with scraPy (Python)

* Scraping with pattern (Python)

* Scraping with Selenium (Python)

* Dealing with PDFs (pdftools, tabulizer, and textreadr in R; xpdf/pdftotext in Unix; PyPDF2/PyPDF4, PDFQuery/Slate/PDFminer, xpdf, and tabula-py in Python)
   * R Notebook nb.html: [here](https://burtmonroe.github.io/TextAsDataCourse/Tutorials/TADA-PDFs.nb.html)
   * R Notebook Rmd: [here](https://burtmonroe.github.io/TextAsDataCourse/Tutorials/TADA-PDFs.Rmd)


* Dealing with .doc, .docx, .rtf files (textreadr in R; python-docx and python-docx2txt in Python)

* Dealing with XML files

* Dealing with JSON files

* Introduction to encoding, Unicode, UTF-8 and similar concepts

## Measuring, Modeling, and Representation

* Introduction to Cosine Similarity (R)
   * Notebook nb.html: [here](https://burtmonroe.github.io/TextAsDataCourse/Tutorials/TADA-CosineSimTutorial.nb.html)
   * Notebook Rmd: [here](https://burtmonroe.github.io/TextAsDataCourse/Tutorials/TADA-CosineSimTutorial.Rmd)

* Introduction to Dictionary-based Analysis in R
   * Colab: [here](https://colab.research.google.com/drive/1EX4eWKqt7tkBukMxy4jqJ_SS0iAbOvhL?usp=sharing)
   
* Introduction to Text Classification (Naive Bayes, Logistic/ridge/LASSO, Support Vector Machine, Random Forests, and ensembling) (R)
   * Notebook nb.html: [here](https://burtmonroe.github.io/TextAsDataCourse/Tutorials/TADA-ClassificationV2.nb.html)
   * Notebook Rmd: [here](https://burtmonroe.github.io/TextAsDataCourse/Tutorials/TADA-ClassificationV2.Rmd)

* Latent Dirichlet Allocation in R (topicmodels, lda, and MALLET)

* Latent Dirichlet Allocation in Python's "lda" package.
   * Colab notebook: [here](https://colab.research.google.com/drive/1i7DdjegYt4kJqU2fv9-e00qCrbAtEpt2)

* LDA and related analyses in gensim (Python).

* Introduction to the Structural Topic Model (R)
   * Notebook nb.html: [here](https://burtmonroe.github.io/TextAsDataCourse/Tutorials/IntroSTM.nb.html)
   * Notebook Rmd: [here](https://burtmonroe.github.io/TextAsDataCourse/Tutorials/IntroSTM.Rmd)
      * Requires [poliblogs2008.csv](https://burtmonroe.github.io/TextAsDataCourse/Tutorials/poliblogs2008.csv.zip) (25M uncompressed)

* Topic models and unsupervised learning with gensim (Python)

* Code for Fightin Words and Demo (R)
   * Notebook nb.html: [here](https://burtmonroe.github.io/TextAsDataCourse/Tutorials/TADA-FightinWords.nb.html)
   * Notebook Rmd: [here](https://burtmonroe.github.io/TextAsDataCourse/Tutorials/TADA-FightinWords.Rmd)
   
* Introduction to Scaling with Wordfish (R)
   * Notebook nb.html: [here](https://burtmonroe.github.io/TextAsDataCourse/Tutorials/IntroductionToWordfish.nb.html)
   * Notebook Rmd: [here](https://burtmonroe.github.io/TextAsDataCourse/Tutorials/IntroductionToWordfish.Rmd)

* Introduction to Estimating Word Embeddings with gensim (word2vec and fasttext) (Python)
   * https://colab.research.google.com/drive/1eSzd2z5B3CDeTxpdMXCIh3bm1L-gYzCr?usp=sharing#scrollTo=54KJAKL0OD5Q

* (3rd party tutorials) Estimating GloVe embeddings in R with
   * text2vec:  http://text2vec.org/glove.html
   * quanteda:  https://quanteda.io/articles/pkgdown/replication/text2vec.html

## Neural NLP / Deep Learning
   
* (3rd party demo) [Interactive Demo, (Feedforward) Neural Networks (Daniel Smilkov and Shan Carter, TensorFlow](https://playground.tensorflow.org/)

* Text Classification with Keras and Tensorflow in Python:
   * https://colab.research.google.com/drive/1MG2_5Hx5dwN77hmVNY0aUiGo99k2mPGb?usp=sharing

* Text Classidication with Keras and Tensorflow in R - needs to be ported from RStudio Cloud.

* Text Classification with Keras and Tensorflow 2: Dropout and Weight Regularization (Python): 
   * https://colab.research.google.com/drive/1kGhXArEbWDP_A4TtlB1cgSubekIsX4VP?usp=sharing

* Text Classification with Keras and Tensorflow 2: Dropout and Weight Regularization (R):
   * https://colab.research.google.com/drive/1hq9eCrWjDOkpMUY0QJ9fAOHWagcBSXU7?usp=sharing

* Text Classification with Keras and Tensorflow 3: Pretrained Embeddings (Python): 
   * https://colab.research.google.com/drive/1pkJNzWDdqTaVzZFQ1RnkAxx87Wkyr31T?usp=sharing

* Text Classification with Keras and Tensorflow 3: Pretrained Embeddings (R): Not currently functional.

* Text Classification with Keras and Tensorflow 4: Incorporating an Embedding Layer (Python):
   * https://colab.research.google.com/drive/1_6m2DVFQJPZH5UENZDs7jkrOU6kjyuCu?usp=sharing

* Text Classification with Keras and Tensorflow 4: Incorporating an Embedding Layer (R):
   * https://colab.research.google.com/drive/1n1Al0lplHxY78P5vPATBp6kLUUYz6maA?usp=sharing

* (Older) Introduction to Deep Learning with Keras and TensorFlow in R
   * Builds deep and shallow feed-forward ANN models for classification of IMDB data. Discusses interpretation. Compares to classic classifiers. Adds embedding layer with embeddings learned during estimation. Adds pretrained (GloVe) embeddings.
   * Notebook nb.html: [here](https://burtmonroe.github.io/TextAsDataCourse/Tutorials/TADA-IntroToKerasAndTensflowInR.nb.html)
   * Notebook Rmd: [here](https://burtmonroe.github.io/TextAsDataCourse/Tutorials/TADA-IntroToKerasAndTensflowInR.Rmd)

* Text Classification with Keras and Tensorflow 5: LSTMs and Bi-LSTMs (Python)

* Text Classification with Keras and Tensorflow 5: LSTMs and Bi-LSTMs (R)

* Text Classification with Keras and Tensorflow 6: CNNs (Python)

* Text Classification with Keras and Tensorflow 6: CNNs (R)

* Third party notebooks on Transformers that may be of interest:
   * Original Tensor2Tensor notebook (deprecated) (has illustration of self-attention): https://colab.research.google.com/github/tensorflow/tensor2tensor/blob/master/tensor2tensor/notebooks/hello_t2t.ipynb
   * Successor Trax notebook: https://colab.research.google.com/github/google/trax/blob/master/trax/intro.ipynb
   * Text Classification with Transformer (Apoorv Nandan, 2020) - https://colab.research.google.com/github/keras-team/keras-io/blob/master/examples/nlp/ipynb/text_classification_with_transformer.ipynb#scrollTo=anLSsILXyULq (IMDB sentiment)

* Text Classification with Keras and Tensorflow - BERT (Python):
   * https://colab.research.google.com/drive/1OQbZQZtoOB7Kg3RR_nqh52gDivuPkaEU?usp=sharing

* Text Translation Using Pretrained Transformer (Encoder-Decoder) Language Models (Python): 
   * https://colab.research.google.com/drive/1d6SZzl1Rnxr25e8_ecR1vZGG156aOUk-?usp=sharing
