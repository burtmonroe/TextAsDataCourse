# Open source Tools for Text as Data / NLP

## Python

### Python generics

  * I strongly recommend the Anaconda distribution and Conda package manager.
  * For the most part, if you are learning Python new, you should be working in Python 3, at this writing 3.6+. But be aware a great deal has been written in Python 2, typically 2.7, and there are important differences. In addition to some general syntax differences, the main issue in text analysis is the handling of encoding (e.g., UTF-8).
  * You should also learn how to set up "environments" for particular combinations of Python version and packages. This can aid replicability and help with trying different packages without breaking something else that is working.
  * The "NumPy stack" - the basic libraries for numerical computing, scientific computing, and data science in Python. Automatically installed with Anaconda.
      * NumPy - provides array / matrix objects and modules for operations on them. (see also Numba - turns NumPy and related code into machine code for much faster processing.) (see also CuPy, a NumPy alternative with NVIDIA CUDA acceleration.)(see also PyTorch, discussed below)
      * SciPy ("Sigh Pie") - scientific computing ... linear algebra, optimization, integration, signal processing
      * pandas - DataFrame tabular objects and manipulations (file i/o, reshaping data, split-apply-combine); time series and econometrics models. (see also Dask - parallel computing; "Big Data" objects extending NumPy, pandas objects; workflow manager)
      * matplotlib - plotting / graphics. Other visualization libraries, installed with Anaconda, include Bokeh (interactive, for browsers), Datashader (for big data), HoloViews (high level front end to matplotlib, Bokeh, etc can also use Datashader), GeoViews (for geographic data). These and others are incorporated in the "PyViz ecosystem" project supported by Anaconda.
      * SymPy - symbolic computation (algebra, calculus). (Not generally used in text / NLP work.)
  * Cython - technically its own language. A mix of Python and C. Produces Python modules that are implemented in C/C++, and so are much faster. SpaCy, for example, is written in Cython, as are many parts of SciPy, pandas, and scikit-learn.


### NLP & text modeling

#### spaCy - https://spacy.io
  * "Industrial Strength Natural Language Processing."
  * Faster than NLTK for most tasks. Scales. Under active development.
  * NLP: tokenization, named entity recognition, POS tagging, dependency parsing, syntax-driven sentence segmentation, pretrained word embeddings. Models for 9 languages as of this writing. Models are based on convolutional neural nets.
  * Interoperates with numpy and AI/machine learning incl deep learning (TensorFlow, PyTorch, scikit-learn, Gensim)
  * Visualizers builtin for syntax and NER.
  * Good if you want optimized model for established task. Not good for research on different models. Less flexible than NLTK.
  * Extensions: Thinc, sense2vec, displaCY

#### NLTK (Natural Language Toolkit) - http://www.nltk.org
  * Most established NLP library. Lots of tools for lots of NLP tasks in lots of languages. Much easier to tweak / modify / extend than spaCy. Large user community, lots of examples, etc.
  * Can be slow. Not integrated with neural network / word embedding approaches.
  * Classification, tokenization, stemming, tagging, parsing, semantic reasoning.
  * Interfaces to "over 50 corpora and lexical resources such as WordNet."
  * FREE BOOK: Steven Bird, Ewan Klein, and Edward Loper. "Natural Language Processing with Python -- Analyzing Text with the Natural Language Toolkit" updated for Python 3 and NLTK3: http://www.nltk.org/book.

#### TextBlob - https://www.textblob.readthedocs.io/
  * "Simplified text processing." High-level interface to NLTK & pattern
  * Noun phrase extraction, part-of-speech tagging, sentiment analysis, classification, machine translation via Google translate, word tokenization, sentence tokenization, word/phrase frequencies, parsing, inflection & lemmatization, spelling correction, integrated with WordNet.

#### fastText - https://fasttext.cc
  * "Scalable solutions for text representation and classification." Open-source by Facebook AI Research (FAIR) lab.
  * We are using components of this in our multilingual work.

#### StanfordNLP 
  * Stanford NLP Group's "Python NLP Library for Many Human Languages."
  * Tokenization, multi-word tokens, lemmatization, POS tagging, dependency parsing
  * Also includes wrapper for (Java) Stanford CoreNLP

#### Polyglot
  * NLP for large number of languages ("16-196 for different tasks."). Small community.
  * Language detection (196 languages), tokenization (196), named entity recognition (40), POS tagging (16), sentiment analysis (136), word embeddings (137), morphology (137), transliteration (69)

#### Gensim (aka "gensim") - https://radimrehurek.com/gensim
  * "Topic modelling for humans"
  * Good for unsupervised NLP tasks (e.g., LDA, LSA/LSI, SVD/NMF, fastText, word2vec, doc2vec). Fast tf-idf and random projections. Fast similarity queries. Parallelized; scales / streams well. Integrates well with neural nets / deep learning. Integrates with NumPy and SciPy.
  * Doesn't really do NLP per se ... pair with SpaCy or NLTK.
  * Tutorials and notebooks: https://radimrehurek.com/gensim/tutorial.html

#### pattern https://clips.uantwerpen.be/pages/pattern
  * "web mining module" - Google, Bing, Twitter, and Wikipedia API, web crawler, HTML DOM parser.
  * NLP - POS tagging, n-gram search, sentiment analysis, WordNet - six European languages
  * Some machine learning - vector space model, clustering, SVM.
  * Has database wrappers, network analysis, javascript visualization

#### Flair
  * New "very simple framework for state-of the-art NLP." In the PyTorch ecosystem.
  * I have not used Flair.

### Web crawling and scraping

#### Scrapy
  * Web crawler / spider. Downloading pages. Has its own extraction utilities, but can be paired with BeautifulSoup.
  * Probably what you should learn.
  * (See also Django ... similar library for web development.)

#### BeautifulSoup
  * Classic, easy to learn library for traversing html (and xml) pages to extract the information you want.
  * Needs at least something like the Requests package to actually download the pages you want.

#### Selenium
  * Tool for interacting with and extracting information from dynamically generated (javascript) webpages. Does several things that simply aren't possible with Scrapy or BeautifulSoup.
  * Takes over your actual browser ... opens windows, clicks buttons, scrolls pages. Very memory intensive and basically takes over your computer if you try to get too elaborate with it.

#### See also pattern (above)

### Machine learning / deep learning

#### scikit-learn

  * Machine learning, some text preprocessing
  * Text preprocessing mostly limited to bag-of-words type approaches.

#### Keras
  * High-level interface to neural net "backends" TensorFlow, Theano, and CNTK.
  * Slower than working directly with TensorFlow or PyTorch.
  * Has R implementation.

#### TensorFlow
  * The most widely used deep learning framework. Many use through Keras.

#### Theano - https://pypi.org/project/Theano
  * "Optimizing compiler for evaluating mathematical expressions on CPUs and GPUs." Especially matrices and tensors.
  * Abstract hybrid of numpy-like array operations and sympy-like algebra operations. Can be used to implement deep learning algorithms. See: http://deeplearning.net.tutorial
  * I've never used Theano. See http://deeplearning.net/software/theano/

#### PyTorch - https://pytorch.org
  * Deep learning platform. "PyTorch enables fast, flexible experimentation and efficient production through a hybrid front-end, distributed training and ecosystem of tools and libraries." Very Pythonic. Only Linux / OSx.
  * Tensor computation (a la NumPy) based on GPUs
  * Has "PyTorch" ecosystem of libraries. Includes fast deep learning (fastai, Horovod, Ignite, PySyft, TensorLy), NLP (Flair, AllenNLP), and machine translation (Translate), dialog models (ParlAI)
  * Absorbed Caffe and Caffe2, deep learning libraries that focus on images.

#### H2O.ai
  * "Democratizing Artificial Intelligence"
  * Has GUI network builder.

#### Chainer - https://chainer.org
  * "A Powerful, Flexible, and Intuitive Framework for Neural Networks"
  * Uses a "define-by-run" / dynamic graph approach, in which network connections are determined during training.
  * Pythonic, CUDA computation.
  * I've never used Chainer. See https://docs.chainer.org/en/stable/

Other language tools that can be wrapped from Python. These require installation of Java.            

#### Stanford CoreNLP
  * Wrap through package StanfordNLP

#### Apache OpenNLP
  * Wrap through package opennlp-python or opennlp_python.

