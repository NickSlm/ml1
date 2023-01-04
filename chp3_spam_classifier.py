import numpy as np
import os
import re
import tarfile
import urllib.request
import email
import email.policy
from html import unescape
from collections import Counter
from bs4 import BeautifulSoup
from sklearn.model_selection import train_test_split
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.pipeline import Pipeline

try:
    import nltk
    stemmer = nltk.PorterStemmer()
except:
    stemmer = None
    
try:
    import urlextract    
    url_extractor = urlextract.URLExtract()
except:
    url_extractor = None

DOWNLOAD_URL = "https://spamassassin.apache.org/old/publiccorpus/"
SPAM_URL = DOWNLOAD_URL + "20030228_spam.tar.bz2"
HAM_URL = DOWNLOAD_URL + "20030228_easy_ham.tar.bz2"
DIR_PATH = os.path.join("datasets","spam")

def fetch_data(file_path=DIR_PATH, spam_url=SPAM_URL, ham_url=HAM_URL):
    if not os.path.isdir(file_path):
        os.makedirs(file_path)
    
    for filename, url in (("ham.tar.bz2", ham_url), ("spam.tar.bz2", spam_url)):
        path = os.path.join(DIR_PATH, filename)
        if not os.path.isfile(path):
            urllib.request.urlretrieve(url, path)
        tar = tarfile.open(path)
        tar.extractall(file_path)
        tar.close()

# fetch_data()

HAM_DIR = os.path.join(DIR_PATH, "easy_ham")
SPAM_DIR = os.path.join(DIR_PATH,"spam")

ham_filenames = [name for name in sorted(os.listdir(HAM_DIR)) if len(name) > 20]
spam_filenames = [name for name in os.listdir(SPAM_DIR) if len(name) > 20]


def load_email(is_spam, filename, dir_path=DIR_PATH):
    directory = "spam" if is_spam else "easy_ham"
    with open(os.path.join(dir_path, directory, filename), "rb") as file:
        return email.parser.BytesParser(policy=email.policy.default).parse(file)
    
ham_emails = [load_email(False, name) for name in ham_filenames]
spam_emails = [load_email(True, name) for name in spam_filenames]


x = np.array(ham_emails + spam_emails, dtype=object)
y = np.array([0] * len(ham_emails) + [1] * len(spam_emails))

x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2, random_state=42)

def get_email_structure(email):
    if isinstance(email, str):
        return email
    payload = email.get_payload()
    if isinstance(payload, list):
        return "multipart({})".format(",".join([
            get_email_structure(sub_email) for sub_email in payload
        ]))
    else:
        return email.get_content_type()
def structures_counter(emails):
    structures = Counter()
    for email in emails:
        structure = get_email_structure(email)
        structures[structure] += 1
    return structures

def html_to_plain_text(html):
    soup = BeautifulSoup(html,features="html.parser")
    return soup.get_text()

html_spam_emails = [email for email in x_train[y_train==1]
                    if get_email_structure(email) == "text/html"]
sample_html_spam = html_spam_emails[7]

def email_to_text(email):
    html = None
    
    for part in email.walk():
        content_type = part.get_content_type()
        if content_type not in ["text/plain","text/html"]:
            continue
        try:
            content =  part.get_content()
        except:
            content = str(part.get_payload())
        if content_type == 'text/plain':
            return content
        else:
            html = content
    if html:
        return html_to_plain_text(html)
            

class EmailToWordCounterTransformer(BaseEstimator, TransformerMixin):
    def __init__(self,strip_headers=True, lower_case=True, remove_punctuation=True,
                 replace_urls=True, replace_numbers=True, stemming=True):
        self.strip_headers = strip_headers
        self.lower_case = lower_case
        self.remove_punctuation = remove_punctuation
        self.replace_urls = replace_urls
        self.replace_numbers = replace_numbers
        self.stemming = stemming
    def fit(self, x, y=None):
        return self
    def transform(self, x, y=None):
        x_transformed = []
        for email in x:
            text = email_to_text(email) or ""
            if self.lower_case:
                text.lower()
            if self.replace_urls and url_extractor is not None:
                urls = list(set(url_extractor.find_urls(text)))
                urls.sort(key=lambda url: len(url), reverse=True)
                for url in urls:
                    text = text.replace(url, " URL ") 
            if self.replace_numbers:
                text = re.sub(r'\d+(?:\.\d*)?(?:[eE][+-]?\d+)?', 'NUMBER', text)
            if self.remove_punctuation:
                text = re.sub(r'\W+', ' ', text, flags=re.M)
            word_counter = Counter(text.split())
            if self.stemming and stemmer:
                stem_counter = Counter()
                for word, count in word_counter.items():
                    stem_word = stemmer.stem(word)
                    stem_counter[stem_word] += count
                word_counter = stem_counter                
            x_transformed.append(word_counter)
        return np.array(x_transformed)
                    
class WordCounterToVectorTransformer(BaseEstimator, TransformerMixin):
    def __init__(self):
        pass

    def fit(self, x, y=None):
        return self
    
    def transform(self, x, y= None):
        pass
                
X_few = x_train[:3]
X_few_wordcounts = EmailToWordCounterTransformer().fit_transform(X_few)
print(X_few_wordcounts)