import pandas as pd
import numpy as np
import sklearn.metrics

import glob, os
import artm
from artm import score_tracker


SHOULD_RUN_ON_A_SMALL_SUBSET = True


def get_true_labels(vw_filename):
    res = {}
    with open(vw_filename, "r", encoding="utf8") as f:
        for line in f:
            contents = line.split("|")
            doc, _, part = line.partition("|")
            modality_name, _, data = part.partition(" ")
            modality_name = modality_name.strip()
            if modality_name == "@origin":
                res[doc.strip()] = data.strip()
    return res

def get_ys(y_true_dict, y_pred_df):
    y_true = []
    y_pred = []
    for doc, answer in y_true_dict.items():
        y_true.append(answer)
        y_pred.append(y_pred_df[doc].idxmax())
    return y_true, y_pred


def read_collection(target_folder, vw_name):
    if len(glob.glob(os.path.join(target_folder, '*.batch'))) < 1:
        batch_vectorizer = artm.BatchVectorizer(
            data_path=vw_name,
            data_format='vowpal_wabbit',
            target_folder=target_folder)
    else:
        batch_vectorizer = artm.BatchVectorizer(
            data_path=target_folder,
            data_format='batches')

    dictionary = artm.Dictionary()
    dict_path = os.path.join(target_folder, 'dict.dict')

    if not os.path.isfile(dict_path):
        dictionary.gather(data_path=batch_vectorizer.data_path)
        dictionary.save(dictionary_path=dict_path)

    dictionary.load(dictionary_path=dict_path)
    return batch_vectorizer, dictionary


if __name__ == "__main__":
    # read train data
    vw_filename = "vw_train{}.txt".format("_small" if SHOULD_RUN_ON_A_SMALL_SUBSET else "")
    batch_vectorizer, dictionary = read_collection("batches_train", vw_filename)

    # build a simple topic model
    N_TOPICS = 7

    model = artm.ARTM(
        topic_names=['topic {}'.format(i) for i in range(N_TOPICS)],
        theta_columns_naming = 'title',
        regularizers=[], 
        scores=[
            artm.TopTokensScore(name='TopTokensScore', num_tokens=10, class_id="@raw_text"),
        ],
        class_ids={'@raw_text': 1.0, '@origin': 1.0}
    )

    model.initialize(dictionary=dictionary)

    model.num_document_passes = 5
    num_collection_passes = 10

    # fit this model on a train dataset

    model.fit_offline(
        batch_vectorizer=batch_vectorizer,
        num_collection_passes=num_collection_passes
    )

    # let's look at most probable words of each topic
    print("Tokens:")
    top_tokens = model.score_tracker['TopTokensScore'].last_tokens
    for topic_name in model.topic_names:
        print(topic_name + ': ')
        print(top_tokens[topic_name])

    # read the test dataset
    vw_filename = "vw_test{}.txt".format("_small" if SHOULD_RUN_ON_A_SMALL_SUBSET else "")
    batch_vectorizer, dictionary = read_collection("batches_test", vw_filename)

    # get distribution Pr(label | document) for test dataset
    result = model.transform(batch_vectorizer=batch_vectorizer, predict_class_id="@origin")

    # read true labels
    vw_filename = "vw_labels{}.txt".format("_small" if SHOULD_RUN_ON_A_SMALL_SUBSET else "")
    y_true_dict = get_true_labels(vw_filename)

    # convert distribution to the point estimate
    # making sure that numbering of documents is consistent 
    y_true, y_predicted = get_ys(y_true_dict, result)

    # measure classification accuracy
    print('Classification report:')
    print(sklearn.metrics.classification_report(y_true, y_predicted))


