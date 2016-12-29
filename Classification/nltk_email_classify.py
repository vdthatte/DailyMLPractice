from pandas import *
import numpy as np
import os
import re
from nltk import NaiveBayesClassifier
import nltk.classify
from nltk.tokenize import wordpunct_tokenize
from nltk.corpus import stopwords
from collections import defaultdict

data_path = os.path.abspath(os.path.join('.', 'data'))
spam_path = os.path.join(data_path, 'spam')
spam2_path = os.path.join(data_path, 'spam_2') 
easyham_path = os.path.join(data_path, 'easy_ham')
easyham2_path = os.path.join(data_path, 'easy_ham_2')
hardham_path = os.path.join(data_path, 'hard_ham')
hardham2_path = os.path.join(data_path, 'hard_ham_2')

def get_msgdir(path):
    '''
    Read all messages from files in a directory into
    a list where each item is the text of a message. 
    
    Simply gets a list of e-mail files in a directory,
    and iterates get_msg() over them.

    Returns a list of strings.
    '''
    filelist = os.listdir(path)
    filelist = filter(lambda x: x != 'cmds', filelist)
    all_msgs =[get_msg(os.path.join(path, f)) for f in filelist]
    return all_msgs

def get_msg(path):
    '''
    Read in the 'message' portion of an e-mail, given
    its file path. The 'message' text begins after the first
    blank line; above is header information.

    Returns a string.
    '''
    with open(path, 'rU') as con:
        msg = con.readlines()
        first_blank_index = msg.index('\n')
        msg = msg[(first_blank_index + 1): ]
        return ''.join(msg) 

train_spam_messages    = get_msgdir(spam_path)    # spam messages for training
train_easyham_messages = get_msgdir(easyham_path) # non spam messages for Training ... who the fuck names this ham? like wtf

train_easyham_messages = train_easyham_messages[:500]
train_hardham_messages = get_msgdir(hardham_path)

test_spam_messages    = get_msgdir(spam2_path)
test_easyham_messages = get_msgdir(easyham2_path)
test_hardham_messages = get_msgdir(hardham2_path)

# DATA PROCESSING SHIT
def get_msg_words(msg, stopwords = [], strip_html = False):

    '''
    Returns the set of unique words contained in an e-mail message. Excludes 
    any that are in an optionally-provided list. 

    NLTK's 'wordpunct' tokenizer is used, and this will break contractions.
    For example, don't -> (don, ', t). Therefore, it's advisable to supply
    a stopwords list that includes contraction parts, like 'don' and 't'.
    '''
    
    # Strip out weird '3D' artefacts.
    msg = re.sub('3D', '', msg)
    
    # Strip out html tags and attributes and html character codes,
    # like &nbsp; and &lt;.
    if strip_html:
        msg = re.sub('<(.|\n)*?>', ' ', msg)
        msg = re.sub('&\w+;', ' ', msg)
    
    # wordpunct_tokenize doesn't split on underscores. We don't
    # want to strip them, since the token first_name may be informative
    # moreso than 'first' and 'name' apart. But there are tokens with long
    # underscore strings (e.g. 'name_________'). We'll just replace the
    # multiple underscores with a single one, since 'name_____' is probably
    # not distinct from 'name___' or 'name_' in identifying spam.
    msg = re.sub('_+', '_', msg)

    # Note, remove '=' symbols before tokenizing, since these are
    # sometimes occur within words to indicate, e.g., line-wrapping.
    msg_words = set(wordpunct_tokenize(msg.replace('=\n', '').lower()))
     
    # Get rid of stopwords
    msg_words = msg_words.difference(stopwords)
    
    # Get rid of punctuation tokens, numbers, and single letters.
    msg_words = [w for w in msg_words if re.search('[a-zA-Z]', w) and len(w) > 1]
    
    return msg_words

sw = stopwords.words('english')
sw.extend(['ll', 've'])

def features_from_messages(messages, label, feature_extractor, **kwargs):
    '''
    Make a (features, label) tuple for each message in a list of a certain,
    label of e-mails ('spam', 'ham') and return a list of these tuples.

    Note every e-mail in 'messages' should have the same label.
    '''
    features_labels = []
    for msg in messages:
        features = feature_extractor(msg, **kwargs)
        features_labels.append((features, label))
    return features_labels

def word_indicator(msg, **kwargs):
    '''
    Create a dictionary of entries {word: True} for every unique
    word in a message.

    Note **kwargs are options to the word-set creator,
    get_msg_words().
    '''
    features = defaultdict(list)
    msg_words = get_msg_words(msg, **kwargs)
    for w in msg_words:
            features[w] = True
    return features

def make_train_test_sets(feature_extractor, **kwargs):
    '''
    Make (feature, label) lists for each of the training 
    and testing lists.
    '''
    train_spam = features_from_messages(train_spam_messages, 'spam', 
                                        feature_extractor, **kwargs)
    train_ham = features_from_messages(train_easyham_messages, 'ham', 
                                       feature_extractor, **kwargs)
    train_set = train_spam + train_ham

    test_spam = features_from_messages(test_spam_messages, 'spam',
                                       feature_extractor, **kwargs)

    test_ham = features_from_messages(test_easyham_messages, 'ham',
                                      feature_extractor, **kwargs)

    test_hardham = features_from_messages(test_hardham_messages, 'ham',
                                          feature_extractor, **kwargs)
    
    return train_set, test_spam, test_ham, test_hardham


def check_classifier(feature_extractor, **kwargs):
    '''
    Train the classifier on the training spam and ham, then check its accuracy
    on the test data, and show the classifier's most informative features.
    '''
    
    # Make training and testing sets of (features, label) data
    train_set, test_spam, test_ham, test_hardham = \
        make_train_test_sets(feature_extractor, **kwargs)
    
    # Train the classifier on the training set
    classifier = NaiveBayesClassifier.train(train_set)
    
    # How accurate is the classifier on the test sets?
    print ('Test Spam accuracy: {0:.2f}%'
       .format(100 * nltk.classify.accuracy(classifier, test_spam)))
    print ('Test Ham accuracy: {0:.2f}%'
       .format(100 * nltk.classify.accuracy(classifier, test_ham)))
    print ('Test Hard Ham accuracy: {0:.2f}%'
       .format(100 * nltk.classify.accuracy(classifier, test_hardham)))

    # Show the top 20 informative features
    print classifier.show_most_informative_features(20)



check_classifier(word_indicator, stopwords = sw, strip_html = True)



