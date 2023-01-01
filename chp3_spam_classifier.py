import os
import tarfile
import urllib.request

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



fetch_data()
        
    