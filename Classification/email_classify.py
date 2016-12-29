import os
import math
import string
import nltk
from nltk.corpus import stopwords
import numpy as np
import textmining as txtm
from pandas import *

data_path = os.path.abspath(os.path.join('.', 'data'))
spam_path = os.path.join(data_path, 'spam')
spam2_path = os.path.join(data_path, 'spam_2') 
easyham_path = os.path.join(data_path, 'easy_ham')
easyham2_path = os.path.join(data_path, 'easy_ham_2')
hardham_path = os.path.join(data_path, 'hard_ham')
hardham2_path = os.path.join(data_path, 'hard_ham_2')


def get_msg(path):
    with open(path, 'rU') as con:
        msg = con.readlines()
        first_blank_index = msg.index('\n')
        msg = msg[(first_blank_index + 1): ]
        return ''.join(msg)  

def get_msgdir(path):
    filelist = os.listdir(path)
    filelist = filter(lambda x: x != 'cmds', filelist)
    all_msgs =[get_msg(os.path.join(path, f)) for f in filelist]
    return all_msgs

all_spam = get_msgdir(spam_path)
all_easyham = get_msgdir(easyham_path)
all_easyham = all_easyham[:500]
all_hardham = get_msgdir(hardham_path)

sw = stopwords.words('english')
rsw = read_csv('r_stopwords.csv')['x'].values.tolist() 

def tdm_df(doclist, stopwords = [], remove_punctuation = True, 
           remove_digits = True, sparse_df = False):
    
    # Create the TDM from the list of documents.
    tdm = txtm.TermDocumentMatrix()
  
    for doc in doclist:
        if remove_punctuation == True:
            doc = doc.translate(None, string.punctuation.translate(None, '"'))
        if remove_digits == True:
            doc = doc.translate(None, string.digits)
            
        tdm.add_doc(doc)
    
    # Push the TDM data to a list of lists,
    # then make that an ndarray, which then
    # becomes a DataFrame.
    tdm_rows = []
    for row in tdm.rows(cutoff = 1):
        tdm_rows.append(row)
        
    tdm_array = np.array(tdm_rows[1:])
    tdm_terms = tdm_rows[0]
    df = DataFrame(tdm_array, columns = tdm_terms)
    
    # Remove stopwords from the dataset, manually.
    # TermDocumentMatrix does not do this for us.
    if len(stopwords) > 0:
        for col in df:
            if col in stopwords:
                del df[col]
    
    if sparse_df == True:
        df.to_sparse(fill_value = 0)
    
    return df

spam_tdm = tdm_df(all_spam, stopwords = rsw, sparse_df = True)

def make_term_df(tdm):
    '''
    Create a DataFrame that gives statistics for each term in a 
    Term Document Matrix.

    `frequency` is how often the term occurs across all documents.
    `density` is frequency normalized by the sum of all terms' frequencies.
    `occurrence` is the percent of documents that a term appears in.

    Returns a DataFrame, with an index of terms from the input TDM.
    '''
    term_df = DataFrame(tdm.sum(), columns = ['frequency'])
    term_df['density'] = term_df.frequency / float(term_df.frequency.sum())
    term_df['occurrence'] = tdm.apply(lambda x: np.sum((x > 0))) / float(tdm.shape[0])
    
    return term_df.sort_index(by = 'occurrence', ascending = False)

spam_term_df = make_term_df(spam_tdm)
spam_term_df.head()
easyham_tdm = tdm_df(all_easyham, stopwords = rsw, sparse_df = True)
easyham_term_df = make_term_df(easyham_tdm)
easyham_term_df.head(6)

def classify_email(msg, training_df, prior = 0.5, c = 1e-6):
    msg_tdm = tdm_df([msg])
    msg_freq = msg_tdm.sum()
    msg_match = list(set(msg_freq.index).intersection(set(training_df.index)))
    if len(msg_match) < 1:
        return math.log(prior) + math.log(c) * len(msg_freq)
    else:
        match_probs = training_df.occurrence[msg_match]
        return (math.log(prior) + np.log(match_probs).sum() 
                + math.log(c) * (len(msg_freq) - len(msg_match)))

hardham_spamtest = [classify_email(m, spam_term_df) for m in all_hardham]
hardham_hamtest = [classify_email(m, easyham_term_df) for m in all_hardham]
s_spam = np.array(hardham_spamtest) > np.array(hardham_hamtest)

def spam_classifier(msglist):
    spamprob = [classify_email(m, spam_term_df) for m in msglist]
    hamprob = [classify_email(m, easyham_term_df) for m in msglist]
    classify = np.where(np.array(spamprob) > np.array(hamprob), 'Spam', 'Ham')
    out_df = DataFrame({'pr_spam' : spamprob,
                        'pr_ham'  : hamprob, 
                        'classify'   : classify}, 
                       columns = ['pr_spam', 'pr_ham', 'classify'])
    return out_df

def class_stats(df):
    return df.classify.value_counts() / float(len(df.classify))



    
