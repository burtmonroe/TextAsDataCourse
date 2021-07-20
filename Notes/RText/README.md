# Open source Tools for Text as Data / NLP in R

## R generics

* I strongly recommend working within **RStudio**. Some of the notebooks provided are designed for the Essex Summer School Advanced TADA/NLP course, and may not run as is outside of the RStudio Cloud environment provided for that class.
* There are many many useful tools for NLP, and for data science generally, within the **tidyverse** and I strongly recommend you become familiar / take advantage of them. A few specific to text/NLP are discussed below. (A few that are more related to data wrangling / big data computation -- like dplyr, purrr -- are discussed in notebooks for Social Data Analytics 501 here: XX.
* Because text data can become large and sparse, many of the tools interact with frameworks that deal better with such data. Notable are **data.table**, which provides a Pandas/dplyr/SQL like data management framework that is more memory efficient than dplyr, and **Matrix**, which provides utilities for sparse matrices.
* Interacting with Python. You can use R code and access R objects within a Python process through the Python library rpy2. You can use Python code and access Python objects within an R process through the R library reticulate. R can be used with Python notebooks in Jupyter or Colab; Python can be used with R Notebooks in RStudio.

## Text processing and string manipulation

String manipulation operations, with particular focus on pattern matching with **regular expressions** in the **stringr** and **stringi** libraries, are addressed in this tutorial: https://burtmonroe.github.io/TextAsDataCourse/Tutorials/TADA-IntroToTextManipulation.nb.html. (There is a similar notebook for Python.)

You may also be interested in packages **ore** (https://github.com/jonclayden/ore) and **rex** (https://github.com/kevinushey/rex) which provide alternative regular expression engines/syntax, in both cases based on how they work in the programming language Ruby.

Also of note is package **stringdist** (https://github.com/markvanderloo/stringdist) which provides string distance metrics (see Jurafsky and Martin, Chapter 2 for a good discussion of "edit distance") and fuzzy string match searching.


## Text-as-data frameworks/ecosystems (Quanteda, tm, tidytext, and corpus)

Quanteda, tm, and tidytext are general -- partially overlapping, interrelated, and interconnected -- frameworks for text-as-data analysis, and most social scientific text work in R is managed through one of these. Their primary strengths are in the data science aspects of managing/wrangling text data as quantitative data for statistical / machine learning analysis.

**quanteda** (https://quanteda.io)

* Quanteda is a very general ecosystem for text analysis in R. A very large percentage of what is typically done in social science text-as-data research can be done with, or at least through, quanteda.
* Official description: "The package is designed for R users needing to apply natural language processing to texts, from documents to final analysis. Its capabilities match or exceed those provided in many end-user software applications, many of which are expensive and not open source. The package is therefore of great benefit to researchers, students, and other analysts with fewer financial resources. While using quanteda requires R programming knowledge, its API is designed to enable powerful, efficient analysis with a minimum of steps. By emphasizing consistent design, furthermore, quanteda lowers the barriers to learning and using NLP and quantitative text analysis even for proficient R programmers."
* For general usage, see the tutorial here: XX.
* There are additional tutorials on such things as classification, topic modeling, and scaling within quanteda here: XX.

**tm** ("text mining") - https://cran.r-project.org/web/packages/tm/vignettes/tm.pdf

* The *tm* package is the grand-daddy of text analysis packages in R, predating both quanteda and tidytext. The basic text processing features of tm are very similar to those of quanteda (in fact, quanteda calls many tm functions under the hood) but with different syntax. There is slightly more flexibility in some tm functions, but quanteda does a wide range of things that tm does not.
* XX
* The tm package also has some nice utilities for reading different text formats -- e.g., pdf and xml -- as well as "plugins" for other software (e.g., Alceste) and resources (e.g., Lexis-Nexis).

**tidytext** https://cran.r-project.org/web/packages/tidytext/vignettes/tidytext.html

* Tidytext is a package -- a philosophy really -- for approaching text analysis with thelogic and vast software ecosystem of the "tidyverse," which includes libraries like dplyr, tidyr, and ggplot2.
* There is a whole book on it: https://www.tidytextmining.com/ (Text Mining with R, by Julia Silge and David Robinson).
* The data management / pipeline principles and practices are novel and tidytext is definitely worth your time.

**corpustools**

**corpus**

* Official description: "Text corpus data analysis, with full support for international text (Unicode). Functions for reading data from newline-delimited 'JSON' files, for normalizing and tokenizing text, for searching for term occurrences, and for computing term occurrence frequencies, including n-grams." 
* I am not familiar with the corpus package, although it is under active development and its authorship team has social science roots. It appears to occupy a similar space to quanteda or tm, and out of my own ignorance, I can't say what its advantages are relative to those more commonly used packages. (I am open to enlightenment!)

**TextMiningGUI

**RcmdrPlugin.temis** 

* Provides a graphical text mining package, as a plugin for R Commander. 
* It has a variety of importers for different text formats. It calculates doc-term matrices and does related analyses like correspondence analysis and hierarchical clustering. 
* I have never used RcmdrPlugin.temis, or R Commander, for that matter.

**RTextTools** 

* Official description: "A machine learning package for automatic text classification that makes it simple for novice users to get started with machine learning, while allowing experienced users to easily experiment with different settings and algorithm combinations. The package includes eight algorithms for ensemble classification (svm, slda, boosting, bagging, random forests, glmnet, decision trees, neural networks), comprehensive analytics, and thorough documentation." 
* It appears that RTextTools is no longer maintained.


# NLP pipelines

**udpipe**

* Provides an R wrapper for the C++ software UDPipe, described by Straka, et al. (2016): https://aclanthology.org/L16-1680/ 
* Official description of UDPipe: "UDPipe is a trainable pipeline for tokenization, tagging, lemmatization and dependency parsing of CoNLL-U files. UDPipe is language-agnostic and can be trained given annotated data in CoNLL-U format. Trained models are provided for nearly all UD treebanks."
* A major benefit for R users is that it does not have Java, Python or other dependencies that can offer installation challenges with other NLP pipeline packages.
* By default, the `udpipe_annotate` command does tokenization, POS tagging, lemmatization and dependency parsing. 

**spacyR and SpaCy from R**

* The **spacyR** package provides an R wrapper for the Python/Cython NLP package SpaCy. It provides functions for near-state-of-the-art tokenization, part-of-speech tagging, dependency parsing, named entity recognition, and other NLP tasks. The Python packacge spaCy has more functionality, but you have to use reticulate to access this through R.
* See notebook here: XX.

**Stanza (Qi, et al. 2020) / StanfordNLP / Stanford CoreNLP / coreNLP**

* Stanza -- formerly StanfordNLP -- is a Python library not available directly in R. I'm not completely sure it can even hypothetically be accessed via reticulate,  given that it runs on PyTorch and GPUs, but it won't install on RStudio Cloud in any case. I discuss it in a Python notebook here: XX.
* Stanza provides a wrapper to coreNLP, the research group's Java library, and it inherits some coreNLP functionality.  
* Official description: "Stanza is a Python natural language analysis package. It contains tools, which can be used in a pipeline, to convert a string containing human language text into lists of sentences and words, to generate base forms of those words, their parts of speech and morphological features, to give a syntactic structure dependency parse, and to recognize named entities. The toolkit is designed to be parallel among more than 70 languages, using the Universal Dependencies formalism. Stanza is built with highly accurate neural network components that also enable efficient training and evaluation with your own annotated data. The modules are built on top of the PyTorch library. You will get much faster performance if you run the software on a GPU-enabled machine. 
    * In addition, Stanza includes a Python interface to the CoreNLP Java package and inherits additional functionality from there, such as constituency parsing, coreference resolution, and linguistic pattern matching.
    * To summarize, Stanza features:
        * Native Python implementation requiring minimal efforts to set up;
        * Full neural network pipeline for robust text analytics, including tokenization, multi-word token (MWT) expansion, lemmatization, part-of-speech (POS) and morphological features tagging, dependency parsing, and named entity recognition;
        * Pretrained neural models supporting 66 (human) languages;
        * A stable, officially maintained Python interface to CoreNLP.
* Peng Qi, Yuhao Zhang, Yuhui Zhang, Jason Bolton and Christopher D. Manning. 2020. Stanza: A Python Natural Language Processing Toolkit for Many Human Languages. In Association for Computational Linguistics (ACL) System Demonstrations. 2020.
* **coreNLP**  (https://stanfordnlp.github.io/CoreNLP/) -- Stanford CoreNLP has historically been one of the standard, most commonly used, NLP engines.
    * Official description: "CoreNLP is your one stop shop for natural language processing in Java! CoreNLP enables users to derive linguistic annotations for text, including token and sentence boundaries, parts of speech, named entities, numeric and time values, dependency and constituency parses, coreference, sentiment, quote attributions, and relations. CoreNLP currently supports 6 languages: Arabic, Chinese, English, French, German, and Spanish."
    * Unfortunately, the R wrapper provided in the *coreNLP* package no longer seems to function (or, to be precise, I cannot get it to install, on RStudio Cloud or on my own machine, despite having done so before). Your mileage may vary: https://cran.r-project.org/web/packages/coreNLP/index.html.
    * There is a Python port of coreNLP that can be accessed from R via reticulate or the `cleanNLP` package, described below. As noted above, the Stanza package also provides a Python interface to CoreNLP. Given that Stanza largely supercedes coreNLP, I'll leave discussion to the Stanza notebook.


**OpenNLP**

* Official description: "The Apache OpenNLP library is a machine learning based toolkit for the processing of natural language text written in Java. It supports the most common NLP tasks, such as tokenization, sentence segmentation, part-of-speech tagging, named entity extraction, chunking, parsing, and coreference resolution."
* Because Apache OpenNLP is written in Java, this R package requires a successful installation of Java along with the package rJava, so be mindful of this if you attempt to install on your own machines.
* The models/tasks available for each language are listed here: http://opennlp.sourceforge.net/models-1.5/. As of this writing, there are tokenizers, sentence detectors, and two differently trained part-of-speech taggers each for English, Danish, German, Spanish, Dutch, Portuguese, and Swedish, and a varying number of named entity recognizers for English, Spanish, and Dutch.
* Some usage examples are provided in the NLP in R tutorial here: XX.

**sparkNLP**

* Provides an R wrapper for access via **sparklyr** to John Snow Labs' **SparkNLP** libraries. 
* Apache Spark is a "big data" ecosystem, a successor to "Hadoop" in many ways, that allows for machine learning and related data science activities at scale over distributed data. 

**cleanNLP**
* Provides a tidytext interface for NLP "backends" of stringi (just tokenization), udpipe, the Python library spaCy, and the Python port of (Stanford) coreNLP. 
* It interacts with Python and requires the successful installation of the Python package **cleannnlp**.

**korPus**

* Official description from the vignette: "The R package koRpus aims to be a versatile tool for text analysis, with an emphasis on scientific research on that topic. It implements dozens of formulae to measure readability and lexical diversity. On a more basic level koRpus can be used as an R wrapper for third party products, like the tokenizer and POS tagger TreeTagger or language corpora of the Leipzig Corpora Collection. This vignette takes a brief tour around its core components, shows how they can be used and gives some insight on design decisions."
* The korPus package has been around since 2011 and is still under active development as of 2021. I am not aware of any social science applications that use it, so I am less familiar with its particular strengths.

**Other popular libraries accessible through Python

All of the following are discussed in Python notebooks, and are hypothetically accessible in R through reticulate.

* **nltk**
* **AllenNLP**
* **Flair**
* **NLP Architect**

## Text-as-data modeling / topic models / embeddings

**topicmodels**

**stm**

**lda**

**tidylda**

**text2vec**

**wordspace**


## Deep Learning

**keras / tensorflow**

**torch**


## Utility packages

**SnowballC / Rstem**

* Official description of **SnowballC**: "An R interface to the C 'libstemmer' library that implements Porter's word stemming algorithm for collapsing words to a common root to aid comparison of vocabulary. Currently supported languages are Danish, Dutch, English, Finnish, French, German, Hungarian, Italian, Norwegian, Portuguese, Romanian, Russian, Spanish, Swedish and Turkish."
* SnowballC is called by the stemming functions of several of the general packages discussed above.
* The package **Rstem** appears to do the same thing. 

**stopwords**

* Official description: "R package providing “one-stop shopping” (or should that be “one-shop stopping”?) for stopword lists in R, for multiple languages and sources. No longer should text analysis or NLP packages bake in their own stopword lists or functions, since this package can accommodate them all, and is easily extended."
* stopwords is part of the quanteda ecosystem. It wraps stopword lists from multiple sources. A full list is maintained on the project's github: https://github.com/quanteda/stopwords.

**tokenizers**

**tokenizers.bpe**

* Does Byte-Pair Encoding tokenization (syllables/wordpieces).

**NLP**

* Official description: "Basic classes and methods for Natural Language Processing." Provides some of the underlying infrastructure for udpipe, openNLP, and cleanNLP.

**tif - Text Interchange Format

* Official description: "This package describes and validates formats for storing common object arising in text analysis as native R objects. Representations of a text corpus, document term matrix, and tokenized text are included. The tokenized text format is extensible to include other annotations. There are two versions of the corpus and tokens objects; packages should accept both and return or coerce to at least one of these.
* Lincoln Mullen describes tif, which he adopted in the tokenizers package: "The Text Interchange Formats are a set of standards defined at an rOpenSci sponsored meeting in London in 2017. The formats allow R text analysis packages to target defined inputs and outputs for corpora, tokens, and document-term matrices. By adhering to these recommendations, R packages can buy into an interoperable ecosystem."
* See https://github.com/ropensci/tif 

**hunspell**

* Spell-checking

**wordnet**
* Access to WordNet (https://wordnet.princeton.edu/)

**tau**
* Provides encoding utilities

**tokenbrowser**

**textreuse**





