
# coding: utf-8

# In[51]:


import csv
import string
from html.parser import HTMLParser
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords
from sklearn.feature_extraction.text import TfidfVectorizer
from nltk.stem.porter import PorterStemmer


# In[52]:


# Some variable to use in code
data_file1 = "./data/D1.csv";
data_file2 = "./data/D2.csv";

stemmer = PorterStemmer()
stop_words = set(stopwords.words('english'))


# #### Read the csv file and retrieve all posts:

# In[53]:


def read_data(file):
  '''
  Read the csv file and return all posts
  '''
  all_posts = [];
  with open(file, "rt", encoding="utf8") as datafile:
    next(datafile)
    csvreader = csv.reader(datafile)
    for row in csvreader:
      all_posts.append(row)
  return all_posts


# #### Read the text and code tags seperately with html parser:

# In[69]:


class HTMLDataParser(HTMLParser):
  '''
  Parse the html into two types of text
  1) code
  2) normal text
  '''
  def __init__(self):
    # initialize the base class
    HTMLParser.__init__(self);
    
    self.code_content = "";
    self.text_content = "";
    self.is_code_tag = False;
    
  def handle_starttag(self, tag, attrs):
    if tag == "code":
      self.is_code_tag = True;
    else:
      for attr in attrs:
        if attr[1] is not None:
          self.text_content = self.text_content + attr[1]
        
  def handle_endtag(self, tag):
    if(tag == "code"):
      self.is_code_tag = False;
  
  def handle_data(self, data):
    if self.is_code_tag:
      self.code_content = self.code_content + data;
    else:
      self.text_content = self.text_content + data;


# <h4> Step1 : Data Clean </h4> 
# <p>
#    we clean the data by doing
#     <ol>
#         <li> split text and code into seperate parts </li>
#         <li> trim and remove punctuation in text </li>
#         <li> lowercase all the text </li>
#         </ol>
# </p>

# In[70]:


def data_clean(title, body):
  html_parser = HTMLDataParser()
  if body is not None:
    html_parser.unescape(body)
    html_parser.feed(body)
  
  text = title + html_parser.text_content
  code = html_parser.code_content
  
  text = text.lower()
  #remove the punctuation using the character deletion step of translate
  no_punctuation = text.translate(str.maketrans('','', string.punctuation))
  text_tokens = word_tokenize(no_punctuation)
  
  no_punctuation_code = code.translate(str.maketrans('','','{}[\\]()/?~@'))
  code_tokens = word_tokenize(no_punctuation_code)
  
  return (text_tokens, code_tokens)


# <h4> Step 2: Remove stop words </h4>
#     <p> We can remove stop words only for text but not for code as tokens in code are very syntactic. </p>
#     
# <h4> Step 3. Stemming </h4>
# <p> Apply stemming to tokens in text but not for source code since it has a predefined structure. </p>

# In[71]:


def remove_stopwords_and_do_stemming(text_tokens, code_tokens):
  stemmed = []
  for token in text_tokens:
    if not token in stop_words:
      stemmed.append(stemmer.stem(token))
        
  return (stemmed, code_tokens)


# #### Preprocess the data:

# In[72]:


def preprocess_doc(title="", body=""):
  '''
  preprocess each document by using above steps
  '''
  #1. clean the data
  #2. remove stop words
  #3. do stemming
  (text_tokens, code_tokens) = data_clean(title, body)
  (text_tokens, code_tokens) = remove_stopwords_and_do_stemming(text_tokens, code_tokens);
  # concat into one list and send
  text_tokens = text_tokens + code_tokens
  return text_tokens

def preprocess_corpus(corpus):
  tokenized_corpus= []
  for doc in corpus:
    tokenized_corpus.append(preprocess_doc(doc[0], doc[1]))

  return tokenized_corpus


# In[73]:


corpus = read_data(data_file1)
tokens = preprocess_corpus(corpus)
tf_idf = TfidfVectorizer(lowercase=False)
tf1 = tf_idf.fit_transform(tokens)
tf1

