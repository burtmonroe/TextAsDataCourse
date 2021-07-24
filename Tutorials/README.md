Tutorials / Notebooks / Code

(Produced for courses "Text as Data" (Penn State) and "Advanced Text as Data / NLP" (Essex)

NLP / Text as Data Frameworks in R and Python

* R
   * [Open Source Tools for Text as Data in R](https://burtmonroe.github.io/TextAsDataCourse/Notes/RText/)
   
   * Introduction to Text Processing with Quanteda (R) (Warning - this currently uses syntax that has since been deprecated)
      * Notebook nb.html: [here](https://burtmonroe.github.io/TextAsDataCourse/Tutorials/TADA-IntroToQuanteda.nb.html)    
      * Notebook .Rmd: [here](https://burtmonroe.github.io/TextAsDataCourse/Tutorials/TADA-IntroToQuanteda.Rmd)
      
   * (3rd party) Text as Data with tidytext: Great tutorials in the book: https://www.tidytextmining.com
   
   * NLP Pipelines in R (Primary focus is on UDPipe and openNLP, with spacyr separated out to other notebook)
  
   * Introduction to spacyr and spaCy through R (R)


* Python
* 
   * [Open Source Tools for Text as Data in Python](https://burtmonroe.github.io/TextAsDataCourse/Notes/PythonText/)
   
   * Introduction to NLP with spaCy (Python)
      * Notebook html: [here](https://burtmonroe.github.io/TextAsDataCourse/Tutorials/Introduction%20to%20NLP%20with%20spaCy.html)
   
   * Introduction to NLP with textblob, nltk, and pattern (Python)
      * Notebook html: [here](https://burtmonroe.github.io/TextAsDataCourse/Tutorials/Introduction%20to%20NLP%20with%20TextBlob%2C%20NLTK%2C%20and%20pattern.html)
      
   * Introduction to NLP with Stanza / coreNLP (Python)

   * (3rd party demos /guide) AllenNLP: Demos (https://demo.allennlp.org/) and Guide (https://guide.allennlp.org) (Python)

   * (3rd party tutorials) sparkNLP

   * (3rd party tutorials) NLP-Architect

   * Introduction to NLP with Flair

Scraping and Data Wrangling:

* Introduction to String Manipulation and Regular Expressions in R
    * Notebook nb.html: [here](https://burtmonroe.github.io/TextAsDataCourse/Tutorials/TADA-IntroToTextManipulation.nb.html)
    * Notebook .Rmd [here](https://burtmonroe.github.io/TextAsDataCourse/Tutorials/TADA-IntroToTextManipulation.Rmd)
    
* Introduction to String Manipulation and Regular Expressions in Python
    * Notebook html [here](https://burtmonroe.github.io/TextAsDataCourse/Tutorials/Intro%2Bto%2BString%2BManipulation%2Band%2BRegular%2BExpressions%2Bin%2BPython.html)
    * Notebook ipynb [here](https://burtmonroe.github.io/TextAsDataCourse/Tutorials/Intro%2Bto%2BString%2BManipulation%2Band%2BRegular%2BExpressions%2Bin%2BPython.ipynb)
        * Requires [fruit.txt](https://burtmonroe.github.io/TextAsDataCourse/Tutorials/fruit.txt)
        * Requires [sentences.txt](https://burtmonroe.github.io/TextAsDataCourse/Tutorials/sentences.txt)
        
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

Measuring and Modeling

* Introduction to Cosine Similarity (R)
   * Notebook nb.html: [here](https://burtmonroe.github.io/TextAsDataCourse/Tutorials/TADA-CosineSimTutorial.nb.html)
   * Notebook Rmd: [here](https://burtmonroe.github.io/TextAsDataCourse/Tutorials/TADA-CosineSimTutorial.Rmd)

* Introduction to Dictionary-based Sentiment Analysis (R)
   * Notebook nb.html: [here](https://burtmonroe.github.io/TextAsDataCourse/Tutorials/TADA-SentimentAnalysisWithLexicoder.nb.html)
   * Notebook Rmd: [here](https://burtmonroe.github.io/TextAsDataCourse/Tutorials/TADA-SentimentAnalysisWithLexicoder.Rmd)

* Introduction to Text Classification (Naive Bayes, Logistic/ridge/LASSO, Support Vector Machine, Random Forests, and ensembling) (R)
   * Notebook nb.html: [here](https://burtmonroe.github.io/TextAsDataCourse/Tutorials/TADA-Classification.nb.html)
   * Notebook Rmd: [here](https://burtmonroe.github.io/TextAsDataCourse/Tutorials/TADA-Classification.Rmd)

* Latent Dirichlet Allocation in R (topicmodels, lda, and MALLET)

* Latent Dirichlet Allocation in Python's "lda" package. (See also gensim)
   * Colab notebook: [here](https://colab.research.google.com/drive/1i7DdjegYt4kJqU2fv9-e00qCrbAtEpt2)

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
   
* (3rd party demo) [Interactive Demo, (Feedforward) Neural Networks (Daniel Smilkov and Shan Carter, TensorFlow](https://playground.tensorflow.org/)

* Introduction to Deep Learning with Keras and TensorFlow in R
   * Builds deep and shallow feed-forward ANN models for classification of IMDB data. Discusses interpretation. Compares to classic classifiers. Adds embedding layer with embeddings learned during estimation. Adds pretrained (GloVe) embeddings.
   * Notebook nb.html: [here](https://burtmonroe.github.io/TextAsDataCourse/Tutorials/TADA-IntroToKerasAndTensflowInR.nb.html)
   * Notebook Rmd: [here](https://burtmonroe.github.io/TextAsDataCourse/Tutorials/TADA-IntroToKerasAndTensflowInR.Rmd)

* (3rd party notebook) Learned Word Embeddings and Text Classification in Keras and TensorFlow in Python / Google Colab (TensorFlow Tutorial)
   * Explains word embeddings. Builds deep ANN classifier for IMDB data using learned embeddings layer. Provides embeddings visualization.
   * Notebook .ipynb: [here](https://www.tensorflow.org/tutorials/text/word_embeddings)

* (3rd party notebook) Text Classification with Keras and TensorFlow in Python / Google Colab (TensorFlow Hub Authors)
   * Builds basic ANN classifier for IMDB data using pretrained Swivel embeddings from TensorFlow HUB.
   * Notebook .ipynb: [here](https://colab.research.google.com/github/tensorflow/hub/blob/master/examples/colab/tf2_text_classification.ipynb)

* (3rd party notebook) Recurrent Neural Network (LSTM) in Keras and TensorFlow (Python / Google Colab)
   * Builds deep LSTM classifier for IMDB data.
   * Notebook .ipynb: [here](https://colab.research.google.com/github/markwest1972/LSTM-Example-Google-Colaboratory/blob/master/LSTM_IMDB_Sentiment_Example.ipynb)
   * Accompanying blogpost: [Explaining Recurrent Neural Networks, Mark West (2019)](https://www.bouvet.no/bouvet-deler/explaining-recurrent-neural-networks)

* (3rd party notebook) Recurrent Neural Networks (bi-LSTM) in Keras and TensorFlow in Python / Google Colab (TensorFlow tutorial)
   * Builds shallow and deep / stacked bi-LSTM classifier for IMDB data.
   * Notebook .ipynb: [here](https://www.tensorflow.org/tutorials/text/text_classification_rnn)

* (3rd party notebook) Keras Recurrent Neural Networks Guide in Python / Google Colab (TensorFlow guide)
   * In the weeds with RNN options (including LSTMs, bi-LSTMs, and GRUs) in Keras. Example here is not NLP/text, but image (MNIST).
   * Notebook .ipynb: [here](https://www.tensorflow.org/guide/keras/rnn)
 
* Unvetted 3rd party ELMo tutorials
   * https://github.com/UKPLab/elmo-bilstm-cnn-crf/blob/master/Keras_ELMo_Tutorial.ipynb (Keras/IMDB)
   * https://github.com/TheShadow29/ALNLP-Notes/blob/master/notebooks/Using-Elmo.ipynb
   * https://colab.research.google.com/drive/13f6dKakC-0yO6_DxqSqo0Kl41KMHT8A1)
   
* Unvetted 3rd party ULMFiT tutorials
   * https://medium.com/technonerds/using-fastais-ulmfit-to-make-a-state-of-the-art-multi-label-text-classifier-bf54e2943e83
   * https://towardsdatascience.com/transfer-learning-in-nlp-for-tweet-stance-classification-8ab014da8dde

* Demonstration --- Building a Classifier with BERT (Python / Google Colab / Simple Transformers / PyTorch)
   * Colab notebook: [here](https://colab.research.google.com/drive/1JKQj-DWHLv_vBdF3VypAIEC6npULOFGy)
