# Open source Tools for Text as Data / NLP in Python

Notes for students in my Text as Data/NLP courses at Penn State / Essex

### Python generics

  * I strongly recommend the Anaconda distribution and Conda package manager.
  * Google Colab - You can also use Google Colab as a free place to develop and share Python code in notebook form. This is especially useful as a free way to access GPU or TPU computing, which is necessary for neural network modeling of even moderate complexity.
  * For the most part, if you are learning Python new, you should be working in Python 3, at this writing 3.9+. But be aware a great deal has been written in Python 2, typically 2.7, and there are important differences. In addition to some general syntax differences, the main issue in text analysis is the handling of encoding (e.g., UTF-8).
  * You should also learn how to set up "environments" for particular combinations of Python version and packages. This can aid replicability and help with trying different packages without breaking something else that is working.
  * Life will be easier if you work in a good Python "IDE". I like PyCharm (https://www.jetbrains.com/pycharm/).
  * The "NumPy stack" - the basic libraries for numerical computing, scientific computing, and data science in Python. Automatically installed with Anaconda.
      * NumPy - provides array / matrix objects and modules for operations on them. (see also Numba - turns NumPy and related code into machine code for much faster processing.) (see also CuPy, a NumPy alternative with NVIDIA CUDA acceleration.)(see also PyTorch, discussed below)
      * SciPy ("Sigh Pie") - scientific computing ... linear algebra, optimization, integration, signal processing
      * pandas - DataFrame tabular objects and manipulations (file i/o, reshaping data, split-apply-combine); time series and econometrics models. (see also Dask - parallel computing; "Big Data" objects extending NumPy, pandas objects; workflow manager)
      * matplotlib - plotting / graphics. I prefer the extension library "seaborn" (https://seaborn.pydata.org) which is much more R-like. Other visualization libraries, installed with Anaconda, include Bokeh (interactive, for browsers), Datashader (for big data), HoloViews (high level front end to matplotlib, Bokeh, etc can also use Datashader), GeoViews (for geographic data). These and others are incorporated in the "PyViz ecosystem" project (https://pyviz.org/index.html) supported by Anaconda.
      * scikit-learn - go-to (non-neural) machine learning library. Regression, classification, clustering, evaluation, etc.
      * SymPy - symbolic computation (algebra, calculus). (Not generally used in text / NLP work.)
  
  * Cython - technically its own language. A mix of Python and C. Produces Python modules that are implemented in C/C++, and so are much faster. SpaCy, for example, is written in Cython, as are many parts of SciPy, pandas, and scikit-learn. (You will also encounter Jython - an implementation of Python that runs on Java - as there are numerous NLP/data science tools built in Java.)
  * Interacting with R. You can use R code and access R objects within a Python process through the Python library rpy2. You can use Python code and access Python objects within an R process through the R library reticulate. R can be used with Python notebooks in Jupyter or Colab; Python can be used with R Notebooks in RStudio.

### Text processing libraries

See the Python text manipulation notebook for basic operations with str-typed variables, common string operations with module string, pattern matching with regular expressions in the re module, and manipulation and normalization of unicode strings with module unicodedata.

### NLP & text modeling

#### spaCy - https://spacy.io
  * "Industrial Strength Natural Language Processing." spaCy is arguably the default tool for NLP tasks like part of speech tagging and dependency parsing, especially in industry. (Although one 2020 survey indicates, implausibly I think, that SparkNLP is more widely used: https://gradientflow.com/2020nlpsurvey/)
  * Partially ported to R through spacyr.
  * Faster than NLTK for most tasks. Scales. Under active development / large community.
  * NLP: tokenization, named entity recognition, POS tagging, dependency parsing, syntax-driven sentence segmentation, pretrained word embeddings. Models for 17 languages as of this writing. Models are based on convolutional neural nets.
  * Interoperates with numpy and AI/machine learning incl deep learning (TensorFlow, PyTorch, scikit-learn, Gensim)
  * Visualizers (through displaCy) builtin for syntax and NER.
  * Good if you want optimized model for established task. Less good for research on different models. Less flexible than NLTK.
  * Faster than UDPipe, but fewer languages and tends to be a bit less accurate than UDPipe. Stanza authors claim it is less accurate than Stanza.
  * Extensions: Thinc, sense2vec, displaCy
  * (I have used spaCy in several recent projects.)

#### NLTK (Natural Language Toolkit) - http://www.nltk.org
  * Most established NLP library. Lots of tools for lots of NLP tasks in lots of languages. Much easier to tweak / modify / extend than spaCy. Large user community, lots of examples, etc.
  * Can be slow. Not integrated with neural network / word embedding approaches. Definitely not as hip anymore.
  * Classification, tokenization, stemming, tagging, parsing, semantic reasoning.
  * Interfaces to "over 50 corpora and lexical resources such as WordNet."
  * FREE BOOK: Steven Bird, Ewan Klein, and Edward Loper. "Natural Language Processing with Python -- Analyzing Text with the Natural Language Toolkit" updated for Python 3 and NLTK3: http://www.nltk.org/book.
  * (I used NLTK in several of my older projects.)

#### Stanza (formerly StanfordNLP) - https://stanfordnlp.github.io/stanza/
  * Stanford NLP Group's "Python NLP Package for Many Human Languages."
  * Tokenization, multi-word token (MWT) expansion, lemmatization, part-of-speech (POS) and morphological features tagging, dependency parsing, and named entity recognition.
  * Pretrained neural models supporting 66 (human) languages
  * State of the art performance. Claimed to be as or more accurate than UDPipe.
  * Also includes interface for (Java) Stanford CoreNLP.
  * Reference: Peng Qi, Yuhao Zhang, Yuhui Zhang, Jason Bolton and Christopher D. Manning. 2020. Stanza: A Python Natural Language Processing Toolkit for Many Human Languages. In Association for Computational Linguistics (ACL) System Demonstrations.

#### Stanford CoreNLP - https://stanfordnlp.github.io/CoreNLP/ See Stanza.
  * Widely used, but now more or less subsumed by Stanza, especially for Python users.

#### UDPipe (https://ufal.mff.cuni.cz/udpipe)
  * Trainable "language-agnostic" pipeline for tokenizing, tagging, lemmatizing, and dependency parsing.
  * Slower than spaCy but generally better accuracy. Has the most languages available of any general NLP tool here (except perhaps Polyglot, which is narrower in focus).
  * Focused on Universal Dependencies formalism -- https://universaldependencies.org/ -- and ConLL-U formatted files/data.
  * UDPipe 1 is in C++ with Python bindings available, UDPipe 2 is a Python prototype. (Native access through R is available through R library udpipe). (UDPipe can be slotted into spaCy with the package spacy-udpipe.)
  * 94 models of 61 languages, each consisting of a tokenizer, tagger, lemmatizer and dependency parser,
  * References: 
    * (Straka et al. 2017) Milan Straka and Jana Straková. Tokenizing, POS Tagging, Lemmatizing and Parsing UD 2.0 with UDPipe. In Proceedings of the CoNLL 2017 Shared Task: Multilingual Parsing from Raw Text to Universal Dependencies, Vancouver, Canada, August 2017.
    * (Straka et al. 2016) Straka Milan, Hajič Jan, Straková Jana. UDPipe: Trainable Pipeline for Processing CoNLL-U Files Performing Tokenization, Morphological Analysis, POS Tagging and Parsing. In Proceedings of the Tenth International Conference on Language Resources and Evaluation (LREC 2016), Portorož, Slovenia, May 2016.
    * (Clair Kelling and I used UDPipe in Kelling and Monroe 2021.)

#### Apache OpenNLP (https://opennlp.apache.org/)
  * "a machine learning based toolkit for the processing of natural language text."
  * "OpenNLP supports the most common NLP tasks, such as tokenization, sentence segmentation, part-of-speech tagging, named entity extraction, chunking, parsing, language detection and coreference resolution."
  * A well-established part of the Apache open-source (Java) ecosystem, although it seems to have fallen off in usage. 
  * Perhaps because of it being around since 2011 and being based in Java, openNLP is a little bit more opaque to the social science text as data community and not widely used to my knowledge.
  * There is a Python wrapper available: https://github.com/curzona/opennlp-python. (There is also an R wrapper, demonstrated in the R NLP tools tutorial.)

#### Flair - https://github.com/flairNLP/flair
  * "A powerful NLP library. Flair allows you to apply our state-of-the-art natural language processing (NLP) models to your text, such as named entity recognition (NER), part-of-speech tagging (PoS), special support for biomedical data, sense disambiguation and classification, with support for a rapidly growing number of languages."
  * "A text embedding library. Flair has simple interfaces that allow you to use and combine different word and document embeddings, including our proposed Flair embeddings, BERT embeddings and ELMo embeddings."
  * "A PyTorch NLP framework. Our framework builds directly on PyTorch, making it easy to train your own models and experiment with new approaches using Flair embeddings and classes."
  * Most models hosted by HuggingFace.
  * Reference: Alan Akbik, Duncan Blythe, and Roland Vollgraf. 2018. "Contextual String Embeddings for Sequence Labeling." COLING 2018, 27th International Conference on Computational Linguistics. pp: 1638--1649.

#### AllenNLP - https://github.com/allenai/allennlp https://guide.allennlp.org/
  * "AllenNLP is an open source library for building deep learning models for natural language processing, developed by the Allen Institute for Artificial Intelligence. It is built on top of PyTorch and is designed to support researchers, engineers, students, etc., who wish to build high quality deep NLP models with ease. It provides high-level abstractions and APIs for common components and models in modern NLP. It also provides an extensible framework that makes it easy to run and manage NLP experiments."
  * Indicated as the third most used NLP library in industry in one 2020 survey.
  * Demos in which you can try stuff live: https://demo.allennlp.org/named-entity-recognition/named-entity-recognition
  * They have an impressive list of academic projects that have used AllenNLP here: https://gallery.allennlp.org
  * Not just an annotation pipeline, a framework for PyTorch-based NLP model-building. Has models for complex natural language and natural language generation tasks. Has tools for model interpretation, bias mitigation.


#### transformers (and related HuggingFace resources https://huggingface.co/)
  * Hugging Face is an NLP startup which states "We’re on a journey to advance and democratize NLP for everyone. Along the way, we contribute to the development of technology for the better."
  * Hugging Face is perhaps best known as the source of the transformers package, which as of this writing contains 30 pretrained models in over 100 languages and eight major transformer-based neural language understanding architectures: BERT, GPT, GPT-2, Transformer-XL, XLNet, XLM, RoBERTa, and DistilBERT. This is pound-for-pound the coolest NLP stuff you can do with a few lines of code.
  * Hugging Face also has a "datasets" package with over 1000 text/NLP datasets.
  * Models and datasets are available a the Hugging Face "hub": https://huggingface.co/models, https://huggingface.co/datasets.
  * (Sam Bestvater and I used the transformers library to implement a stance classifier leveraging BERT in Bestvater and Monroe 2021.)

#### SparkNLP - https://nlp.johnsnowlabs.com/
  * Spark NLP is built on top of "Spark," an ecosystem for large scale distributed data management and analysis.
  * From wikipedia: "Spark NLP is an open-source text processing library for advanced natural language processing for the Python, Java and Scala programming languages. The library is built on top of Apache Spark and its Spark ML library. Its purpose is to provide an API for natural language processing pipelines that implements recent academic research results as production-grade, scalable, and trainable software. The library offers pre-trained neural network models, pipelines, and embeddings, as well as support for training custom models."
  * Annotators include: "tokenizer, normalizer, stemming, lemmatizer, regular expression, TextMatcher, chunker, DateMatcher, SentenceDetector, DeepSentenceDetector, POS tagger, ViveknSentimentDetector, sentiment analysis, named entity recognition, conditional random field annotator, deep learning annotator, spell checking and correction, dependency parser, typed dependency parser, document classification, and language detection."
  * The SparkNLP "Models Hub" includes "pre-trained pipelines with tokenization, lemmatization, part-of-speech tagging, and named entity recognition exist for more than thirteen languages; word embeddings including GloVe, ELMo, BERT, ALBERT, XLNet, Small BERT, and ELECTRA; sentence embeddings including Universal Sentence Embeddings (USE) and Language Agnostic BERT Sentence Embeddings (LaBSE)."
  * Also Spark OCR. From Wikipedia: "Spark OCR is another extension of Spark NLP for optical character recognition (OCR) from images, scanned PDF documents, and DICOM files.  It provides several image pre-processing features for improving text recognition results such as adaptive thresholding and denoising, skew detection & correction, adaptive scaling, layout analysis and region detection, image cropping, removing background objects. Due to the tight coupling between Spark OCR and Spark NLP, users can combine NLP and OCR pipelines for tasks such as extracting text from images, extracting data from tables, recognizing and highlighting named entities in PDF documents or masking sensitive text in order to de-identify images."
  * There's a Quick Start notebook (https://colab.research.google.com/github/JohnSnowLabs/spark-nlp-workshop/blob/master/jupyter/quick_start_google_colab.ipynb) It's then quite a step up to the "Basics" notebook: https://colab.research.google.com/github/JohnSnowLabs/spark-nlp-workshop/blob/master/tutorials/Certification_Trainings/Public/1.SparkNLP_Basics.ipynb

#### NLP Architect (https://github.com/IntelLabs/nlp-architect)
  * "NLP Architect is an open source Python library for exploring state-of-the-art deep learning topologies and techniques for optimizing Natural Language Processing and Natural Language Understanding Neural Networks." "NLP Architect is a model-oriented library designed to showcase novel and different neural network optimizations. The library contains NLP/NLU related models per task, different neural network topologies (which are used in models), procedures for simplifying workflows in the library, pre-defined data processors and dataset loaders and misc utilities. The library is designed to be a tool for model development: data pre-process, build model, train, validate, infer, save or load a model."
  * NLP models for word chunking, named entity recognition, dependency parsing, intent extraction, sentiment classification, language models
  * Natural Language Understanding models for aspect-based sentiment analysis, intent detection, noun phrase embeddings, word sense detection, relation identification, cross-document coreference, noun phrase semantic segmentation.
  * From Intel.

#### torchtext (https://pytorch.org/text/stable/index.html)
  * torchtext is a library for use with the PyTorch deep learning framework (there are similar libraries torchvision and torchaudio), providing data processing utilities (inlcuding tokenizers, pretrained embeddings, NLP metrics) for text as well as popular datasets.

#### Polyglot - https://polyglot.readthedocs.io/en/latest/
  * NLP for large number of languages ("16-196 for different tasks."). Small community.
  * Language detection (196 languages), tokenization (196), named entity recognition (40), POS tagging (16), sentiment analysis (136), word embeddings (137), morphology (137), transliteration (69)

#### PyNLPl - https://pynlpl.readthedocs.io/
  * "PyNLPl, pronounced as ‘pineapple’, is a Python library for Natural Language Processing. It contains various modules useful for common, and less common, NLP tasks. PyNLPl can be used for basic tasks such as the extraction of n-grams and frequency lists, and to build simple language model. There are also more complex data types and algorithms. Moreover, there are parsers for file formats common in NLP (e.g. FoLiA/Giza/Moses/ARPA/Timbl/CQL). There are also clients to interface with various NLP specific servers. PyNLPl most notably features a very extensive library for working with FoLiA XML (Format for Linguistic Annotatation)."
  * Supposedly advantageous for "more exotic" data formats like FoLiA/Giza/Moses/ARPA/Timbl/CQL.

#### fastText - https://fasttext.cc
  * "Scalable solutions for text representation and classification." Open-source by Facebook AI Research (FAIR) lab.
  * One of the most efficient tools for calculation of word vectors / embeddings. 

#### gensim - https://radimrehurek.com/gensim
  * "Topic modelling for humans"
  * Good for unsupervised NLP tasks (e.g., LDA, LSA/LSI, SVD/NMF, fastText, word2vec, doc2vec). Fast tf-idf and random projections. Fast similarity queries. Parallelized; scales / streams well. Integrates well with neural nets / deep learning. Integrates with NumPy and SciPy.
  * Doesn't really do NLP per se ... pair with, e.g., SpaCy or NLTK.
  * Tutorials and notebooks: https://radimrehurek.com/gensim/tutorial.html

#### TextBlob - https://www.textblob.readthedocs.io/
  * "Simplified text processing." High-level interface to NLTK & pattern
  * Noun phrase extraction, part-of-speech tagging, sentiment analysis, classification, machine translation via Google translate, word tokenization, sentence tokenization, word/phrase frequencies, parsing, inflection & lemmatization, spelling correction, integrated with WordNet.

#### pattern https://clips.uantwerpen.be/pages/pattern
  * "web mining module" - Google, Bing, Twitter, and Wikipedia API, web crawler, HTML DOM parser.
  * NLP - POS tagging, n-gram search, sentiment analysis, WordNet - six European languages
  * Some machine learning - vector space model, clustering, SVM.
  * Has database wrappers, network analysis, javascript visualization

#### MontyLingua


#### Vocabulary
  * A dictionary that provides lookup for definitions, synonyms, antonyms, parts of speech, translation, usage examples,  pronunciation, and hyphenation, returning simple JSON objects.
  * Similar to / an alternative to WordNet (which is available through NLTK).


### Web crawling and scraping

#### Requests
  * "Requests is an elegant and simple HTTP library for Python, built for human beings."
  * The core library for this sort of thing.

#### Scrapy
  * Web crawler / spider. Downloading pages. Has its own extraction utilities, but can be paired with BeautifulSoup.
  * Probably what you should learn.
  * (See also Django ... similar library for web development.)

#### BeautifulSoup
  * Classic, easy to learn library for traversing html (and xml) pages to extract the information you want.
  * Needs at least something like the "Requests" package to actually download the pages you want.

#### Selenium
  * Tool for interacting with and extracting information from dynamically generated (javascript) webpages. Does several things that simply aren't possible with Scrapy or BeautifulSoup.
  * Takes over your actual browser ... opens windows, clicks buttons, scrolls pages. Very memory intensive and basically takes over your computer if you try to get too elaborate with it.

#### See also pattern (above)

#### Different filetypes
  * JSON - package "json"
  * XML - package "xml" (See also BeautifulSoup)
  * PDFs - pdfminer/pdfminer.six, PyPDF2 (consider the standalone tool xpdf -- it is, for example, the most reliable tool for extracting Arabic text from pdfs). Image only pdfs you can try pytesseract & cv2 ("open computer vision") for OCR ("optical character recognition").


### Deep learning frameworks

#### TensorFlow - https://www.tensorflow.org
  * Generally described as the most widely used deep learning framework. Many use through Keras.
  * Lots of tutorials here: https://www.tensorflow.org/tutorials
  * Softer learning curve than PyTorch, especially with Keras front end.

#### PyTorch - https://pytorch.org 
  * Deep learning platform. "PyTorch enables fast, flexible experimentation and efficient production through a hybrid front-end, distributed training and ecosystem of tools and libraries." Very Pythonic. Only Linux / OSx.
  * Tensor computation (a la NumPy) based on GPUs
  * Has "PyTorch" ecosystem of libraries. Includes fast deep learning (fastai, Horovod, Ignite, PySyft, TensorLy), NLP (Flair, AllenNLP), and machine translation (Translate), dialog models (ParlAI)
  * Absorbed Caffe and Caffe2, deep learning libraries that focus on images.
  * We have several projects in C-SoDA using PyTorch.
  * See https://github.com/huggingface/pytorch-transformers for PyTorch implementations of pre-trained Transformer language models, currently BERT, (OpenAI) GPT, (OpenAI) GPT-2, Transformer-XL
  * See http://nlp.seas.harvard.edu/2018/04/03/attention.html for (annotated) PyTorch implementation of original Transformer from "Attention is All You Need."
  * Now a torch library for R gives some of this functionality, but it's early days with that.

#### Keras
  * High-level interface to neural net "backends" TensorFlow, Theano, and CNTK.
  * Slower than working directly with TensorFlow or PyTorch.
  * Has R implementation.

#### fastai
  * Keras-like interface to PyTorch (from machine learning education company Fast.ai)

#### Theano - https://pypi.org/project/Theano
  * "Optimizing compiler for evaluating mathematical expressions on CPUs and GPUs." Especially matrices and tensors.
  * Abstract hybrid of numpy-like array operations and sympy-like algebra operations. Can be used to implement deep learning algorithms. See: http://deeplearning.net.tutorial
  * I've never used Theano. See http://deeplearning.net/software/theano/

#### H2O.ai
  * "Democratizing Artificial Intelligence"
  * Has GUI network builder.

#### Chainer - https://chainer.org
  * "A Powerful, Flexible, and Intuitive Framework for Neural Networks"
  * Uses a "define-by-run" / dynamic graph approach, in which network connections are determined during training.
  * Pythonic, CUDA computation.
  * I've never used Chainer. See https://docs.chainer.org/en/stable/



