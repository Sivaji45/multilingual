#!/usr/bin/env python
# coding: utf-8

# <a href="https://colab.research.google.com/github/ShrutiBobba/MultiLingual-Document-Summarization/blob/main/Final_MultiLingual_DocumentSummarization.ipynb" target="_parent"><img src="https://colab.research.google.com/assets/colab-badge.svg" alt="Open In Colab"/></a>

# ### **Install Dependencies**

# In[16]:


get_ipython().system('pip install fasttext')


# In[2]:


get_ipython().system('pip install stanza')


# In[12]:


import os
import fasttext
import nltk
import sys
import stanza
import spacy
from nltk import ngrams


# In[ ]:


nltk.download('stopwords')
nltk.download('punkt')


# In[ ]:


nltk.download('state_union')
nltk.download('averaged_perceptron_tagger')


# ## **Read Data Files**

# In[ ]:


import sys

def read_file(filename):
    try:
        with open(filename, 'r') as file:
            data = file.read()
        return data
    except FileNotFoundError:
        print(f"Error: File '{filename}' not found.")
    except IOError as e:
        print(f"Error reading file '{filename}': {e}")
    except Exception as e:
        print(f"An unexpected error occurred: {e}")
    sys.exit(1)


# In[ ]:


import os
import sys

def read_all_files(path):
    file_texts = []
    try:
        for file_name in os.listdir(path):
            file_path = os.path.join(path, file_name)
            if os.path.isfile(file_path):
                with open(file_path, 'r') as file:
                    file_texts.append(file.read())
    except OSError as e:
        print(f"Error reading files in '{path}': {e}")
        sys.exit(1)
    return file_texts


# In[ ]:


file_texts = read_all_files("/content/drive/MyDrive/NLP/project/Data/Arabic")

# Check if there are at least two files before accessing their content
if len(file_texts) >= 2:
    print("Content of the first file:")
    print(file_texts[0])

    print("\nContent of the second file:")
    print(file_texts[1])
else:
    print("There are not enough files to display.")


# In[ ]:


file_texts = read_all_files("/content/drive/MyDrive/NLP/project/Data/English")

# Check if there are at least two files before accessing their content
if len(file_texts) >= 2:
    print("Content of the first file:")
    print(file_texts[0])

    print("\nContent of the second file:")
    print(file_texts[1])
else:
    print("There are not enough files to display.")


# ## **Remove Empty Lines**

# In[ ]:


def remove_empty_lines(text):
    lines = text.split('\n')
    non_empty_lines = [line for line in lines if line.strip()]
    return '\n'.join(non_empty_lines)


# ## **Remove Stop Words**

# In[ ]:


def remove_stop_words(articles):
    return [' '.join([token for token in Lang_Model.word_tokenize(article) if token not in Lang_Model.stop_words() and token not in [',', '،']]) for article in articles]


# In[ ]:


articles_with_sw = file_texts
articles = remove_stop_words(file_texts)


# In[ ]:


#break into sentences with stop words
article_sentences_with_sw = [
    [sentence.strip() for sentence in Lang_Model.sent_tokenize(article)]
    for article in articles_with_sw
]
print(article_sentences_with_sw)


# In[ ]:


#break into sentences without stop words
article_sentences = [
    [sentence.strip() for sentence in Lang_Model.sent_tokenize(article)]
    for article in articles
]

article_sentences = [
    [
        ' '.join([token for token in Lang_Model.word_tokenize(sent) if token != '.'])
        for sent in article
    ]
    for article in article_sentences
]
print(article_sentences)


# ## **Removing Punctuvations**

# In[ ]:


tmp_article_sentences = []
for article in article_sentences:
  tmp = []
  for sent in article:
    sent_tokens = Lang_Model.word_tokenize(sent)
    sent_tokens = [word for word in sent_tokens if word != '.']
    new_sent = ""
    for token in sent_tokens:
      new_sent += token + ' '
    tmp.append(new_sent)
  tmp_article_sentences.append(tmp)

article_sentences = tmp_article_sentences
article_sentences


# ## **Detect Language**

# In[ ]:


model = fasttext.load_model('/content/drive/MyDrive/NLP/project/lid.176.ftz')
lang1 = model.predict(remove_new_lines(file_texts[0]), k=1)
lang2 = model.predict(remove_new_lines(file_texts[1]), k=1)
if lang1[0][0] == lang2[0][0]:
  if lang1[0][0] == '__label__en':
    print("English")
    Lang_Model = English_Language_Model()
  elif lang1[0][0] == '__label__ar':
    print("Arabic")
    Lang_Model = Arabic_Language_Model()
  else:
    print("Unknown language detected")


# ##**English Language Model**

# In[ ]:


import nltk
import spacy
import stanza

class English_Language_Model:
    def word_tokenize(self, article):
        return nltk.word_tokenize(article)

    def stop_words(self):
        return nltk.corpus.stopwords.words()

    def sent_tokenize(self, article):
        return nltk.sent_tokenize(article)

    def lemmatization(self, text):
        nlp = stanza.Pipeline(lang='en', processors='tokenize,lemma')
        return [word.lemma for sent in nlp(text).sentences for word in sent.words]

    def pos_tagging(self, text):
        nlp = stanza.Pipeline(lang='en', processors='tokenize,pos')
        return [word.pos for sent in nlp(text).sentences for word in sent.words]

    def get_pos_importances(self, text):
        tag_importance = {'NOUN': 2, 'VERB': 1.9, 'ADV': 1.3, 'ADJ': 1.25}
        return [tag_importance.get(tag, 1) for tag in self.pos_tagging(text)]

    def get_ner_importance(self, text):
        ner_entities = set([word.text for word in spacy.load("en_core_web_sm")(text).ents])
        return [1.5 if word in ner_entities else 1 for word in self.word_tokenize(text)]

    def get_dependency_importance(self, text):
        return [1.1 + 0.1 * (token.head-1) for token in stanza.Pipeline(lang='en', processors='tokenize,mwt,pos,lemma,depparse')(text).sentences[0].words]

    def get_chunks_importance(self, text):
        chunkParser = nltk.RegexpParser(r"""Chunk: {<RB.?>*<VB.?>*<NNP>+<NN>?}""")
        chunked = chunkParser.parse(nltk.pos_tag(nltk.word_tokenize(text)))
        return [1.5 if subtree.label() == 'Chunk' else 1 for subtree in chunked.subtrees() for x in subtree]


# ## **Arabic Language Model**

# In[ ]:


class ArabicLanguageModel:
    def word_tokenize(self, article):
        return nltk.tokenize.word_tokenize(article)

    def stop_words(self):
        return nltk.corpus.stopwords.words('arabic')

    def sent_tokenize(self, article):
        article = re.sub("؟", "?", article)
        return nltk.tokenize.sent_tokenize(article)

    def lemmatization(self, text):
        url = 'https://farasa.qcri.org/webapi/lemmatization/'
        api_key = "kusYKWEbCaQFtSumEA"
        payload = {'text': text, 'api_key': api_key}
        data = requests.post(url, data=payload)
        lemmas = json.loads(data.text)
        return lemmas['text']

    def pos_tagging(self, text):
        nlp = stanza.Pipeline(lang='ar', processors='tokenize,pos')
        doc = nlp(text)
        return [word.pos for sent in doc.sentences for word in sent.words]

    def get_pos_importances(self, text):
        pos_tags = self.pos_tagging(text)
        pos_importances = [2 if tag == 'NOUN' else 1.9 if tag == 'VERB' else 1.3 if tag == 'ADV' else 1.25 if tag == 'ADJ' else 1 for tag in pos_tags]
        return pos_importances

    def get_ner_importance(self, text):
        nlp = stanza.Pipeline(lang='ar', processors='tokenize,mwt,pos,lemma,depparse')
        doc = nlp(text)
        return [1 for _ in doc.sentences[0].words]

    def get_dependency_importance(self, text):
        nlp = stanza.Pipeline(lang='ar', processors='tokenize,mwt,pos,lemma,depparse')
        doc = nlp(text)
        return [1 for _ in doc.sentences[0].words]

    def get_chunks_importance(self, text):
        nlp = stanza.Pipeline(lang='ar', processors='tokenize,mwt,pos,lemma,depparse')
        doc = nlp(text)
        return [1 for _ in doc.sentences[0].words]


# In[ ]:


def get_lemmas(text):
  return Lang_Model.lemmatization(text)


# In[ ]:


def get_pos_tags(text):
  return Lang_Model.pos_tagging(text)


# In[ ]:


def get_pos_tags_importance(text):
  return Lang_Model.get_pos_importances(text)


# In[ ]:


def get_ner_importnace(text):
  return Lang_Model.get_ner_importnace(text)


# In[ ]:


def get_dependency_importnace(text):
  return Lang_Model.get_dependency_importnace(text)


# In[ ]:


def computeTf(lemmas, lemmas_total):
  fdist = nltk.FreqDist(lemmas)
  lemmas_Tf = {}
  for k, v in fdist.items():
    lemmas_Tf[k[0]] = v / lemmas_total
  return lemmas_Tf


# In[ ]:


def computeIdf(lemma, articles):
  N = len(articles)
  nidf = 0
  for article in articles:
      if (lemma in (get_lemmas(article))):
        nidf += 1
  result = math.log(N / nidf)
  return result


# In[ ]:


def get_chunks_importance(text):
  return Lang_Model.get_chunks_importance(text)


# In[ ]:


def get_lemmas_importance(articles):
  lemmas_importance = []
  for article in articles:
    lemmas_in_doc = get_lemmas(article)
    lemmas_total = len(lemmas_in_doc)
    lemma_doc = ""

    for lemma in lemmas_in_doc:
      lemma_doc += lemma + ' '
    lemmas = ngrams(lemma_doc.split(), 1)

    lemmas_Tf = computeTf(lemmas, lemmas_total)

    for lemma in lemmas:
      lemmas_Tf[lemma] *= computeIdf(lemma, articles)
    lemmas_importance.append(lemmas_Tf)

  return lemmas_importance


# In[ ]:


lemmas_importance = get_lemmas_importance(articles)
print(lemmas_importance[0])


# ## **Calculate Sentence Importance**

# In[ ]:


def get_sent_importance(sent, article_num):
    lemmas = get_lemmas(sent)
    pos_importances = get_pos_tags_importance(sent)
    ner_importances = get_ner_importance(sent)
    dependency_importances = get_dependency_importance(sent)
    chunk_importances = get_chunks_importance(sent)
    print(sent, article_num)
    for lemma in lemmas:
        print(lemma)
    result = sum(
        [
            lemmas_importance[article_num].get(lemma, 0) * importance * ner_importance * chunk_importance
            for lemma in lemmas
            for importance in pos_importances
            for ner_importance in ner_importances
            for dependency_importance in dependency_importances
            for chunk_importance in chunk_importances
        ]
    )
    return result


# In[ ]:


article_lemma_sentences = []
article_sent_importance = []
index = 0
for article in article_sentences:
  article_lemma_sentences.append([])
  article_sent_importance.append([])
  for sent in article:
    lemmas = get_lemmas(sent)
    article_lemma_sentences[index].append(lemmas)
    s = get_sent_importance(sent, index)
    article_sent_importance[index].append(get_sent_importance(sent, index))
  index += 1


# In[ ]:


print(lemmas_importance[0])
print(article_lemma_sentences[0][1])
print(articale_sent_importance[0][1])


# In[ ]:


article_sentences = [
    [(article_sentences_with_sw[i][j], sent, article_sent_importance[i][j]) for j, sent in enumerate(article)]
    for i, article in enumerate(article_sentences)
]


# ## **Summarization**

# In[ ]:


summarization_factor = 0.7


# In[ ]:


def sort_sentences(article):
    return sorted(article, key=lambda x: x[2], reverse=True)


# In[ ]:


def is_similar(lemma_sentence1, lemma_sentence2):
    lemmas1, lemmas2 = map(Lang_Model.word_tokenize, [lemma_sentence1, lemma_sentence2])

    common_lemmas = set(lemmas1) & set(lemmas2)
    similar_ratio = len(common_lemmas) / max(len(lemmas1), len(lemmas2))

    return similar_ratio > 0.8, max(lemma_sentence1, lemma_sentence2, key=len) if similar_ratio > 0.8 else ""


# In[ ]:


def get_k_sentences_in_article(summarization_factor, article):
    k = int(len(article) * summarization_factor)
    sorted_sentences = sort_sentences(article)
    selected_sentences = [sorted_sentences[0]]

    for sent in sorted_sentences[1:k]:
        is_unique = all(not is_similar(sent[1], y[1])[0] for y in selected_sentences)
        if is_unique:
            selected_sentences.append(sent)

    return selected_sentences, k


# In[ ]:


def get_k_sentences(summarization_factor, article_sentences):
    k_sentences = [get_k_sentences_in_article(summarization_factor, article)[0] for article in article_sentences]
    result = [k_sentences[0][0]]

    for i in range(1, max(map(len, k_sentences))):
        new_sent = True
        for sentences in k_sentences:
            if i < len(sentences):
                result.append(sentences[i])
                new_sent = False

        if new_sent:
            break

    return result


# In[ ]:


final_text = get_k_sentences(summarization_factor, article_sentences)


# In[ ]:


final_summary = ""
for test in final_text:
  final_summary += test[0] + '\n'


# In[ ]:


with open('/content/drive/MyDrive/NLP/project/summary.txt', 'w') as f:
    f.write(final_summary)


# ## **LDA Approach**

# In[ ]:


import numpy as np
import math
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.decomposition import LatentDirichletAllocation


# In[ ]:


def get_topics(articles=articles):
    doctopic = {i: {} for i in range(4)}
    topics_scores = np.zeros(4)
    models = []

    for article, article_with_sw in zip(articles, articles_with_sw):
        sentences = Lang_Model.sent_tokenize(article)
        cv_X = CountVectorizer(max_features=10000, ngram_range=(1, 2), max_df=0.98).fit_transform(sentences)
        lda = LatentDirichletAllocation(n_components=4).fit_transform(cv_X)
        models.append(lda)

    for model in models:
        topics_scores += np.sum(model, axis=0)

    for i, article_with_sw in enumerate(articles_with_sw):
        sentences = Lang_Model.sent_tokenize(article_with_sw)
        for j, sentence in enumerate(sentences):
            topic_index = np.argmax(models[i][j])
            doctopic[topic_index][sentence] = np.max(models[i][j])

    return doctopic, topics_scores


# In[ ]:


sent_topics,topics_scores=get_topics(articles)


# In[ ]:


def get_k_sentences_with_lda(summarization_factor, article_sentences, sent_topics, topics_scores):
    k_sentences = []
    final_sum = []
    lenn = 0
    old_k = 0

    for article in article_sentences:
        temp, temp_k = get_k_sentences_in_article(summarization_factor, article)
        k_sentences.append(temp)
        lenn += len(temp)
        old_k = max(temp_k, old_k)

    k = int(lenn * summarization_factor)

    num_sent_topics = [math.ceil(k * topic_score / np.sum(topics_scores)) for topic_score in topics_scores]

    for index in sent_topics:
        final_sum.append(sorted(sent_topics[index], reverse=True)[:num_sent_topics[index]])

    return final_sum


# In[ ]:


final_text_with_lda = get_k_sentences_with_lda(summarization_factor, article_sentences,sent_topics,topics_scores)


# In[ ]:


final_text_with_lda


# In[ ]:


final_summary = ""
for test in final_text_with_lda:
  for t in test:
    final_summary += t + '\n'


# In[ ]:


with open('/content/drive/MyDrive/NLP/project/summary_with_lda.txt', 'w') as f:
    f.write(final_summary)


# In[ ]:




