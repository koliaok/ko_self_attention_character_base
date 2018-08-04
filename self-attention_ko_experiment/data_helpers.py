import numpy as np
import re
import csv
import copy


class_dict = {'POS':1, 'NEG':2, 'OBJ':3, 'NEU':3}



def load_data_and_labels(path):
    data = []
    y = []
    with open(path, 'r', encoding='utf-8') as f:
        rd = f.readlines()
        for txt in rd:
            txt = txt.rstrip('\n\r')
            txt = txt.split('\t')

            data.append(data_preprocessing(txt[1]))
            y.append(class_dict[txt[0]])

    data = np.asarray(data)
    y = np.asarray(y)

    # Label Data
    labels_count = np.unique(y).shape[0]
    print(labels_count)

    # convert class labels from scalars to one-hot vectors
    def dense_to_one_hot(labels_dense, num_classes):
        num_labels = labels_dense.shape[0]
        index_offset = np.arange(num_labels) * num_classes
        labels_one_hot = np.zeros((num_labels, num_classes))
        labels_one_hot.flat[index_offset + labels_dense.ravel() - 1] = 1
        return labels_one_hot

    labels = dense_to_one_hot(y, labels_count)
    labels = labels.astype(np.uint8)

    return data, labels


def batch_iter(data, batch_size, num_epochs, shuffle=True):
    """
    Generates a batch iterator for a dataset.
    """
    data = np.array(data)
    data_size = len(data)
    num_batches_per_epoch = int((len(data) - 1) / batch_size) + 1
    for epoch in range(num_epochs):
        # Shuffle the data at each epoch
        if shuffle:
            shuffle_indices = np.random.permutation(np.arange(data_size))
            shuffled_data = data[shuffle_indices]
        else:
            shuffled_data = data
        for batch_num in range(num_batches_per_epoch):
            start_index = batch_num * batch_size
            end_index = min((batch_num + 1) * batch_size, data_size)
            yield shuffled_data[start_index:end_index]

def my_func(iterator):
    res =list()
    res_tuple = list()
    for x in iterator:
        for char in x:
            res.append(char)
        res_tuple.append(copy.deepcopy(res))
        res.clear()
    res = tuple(res_tuple)
    return res

def data_preprocessing(data):
    """
    data를 전처리하는 함수
    :param data:
    :return:
    """
    comment = re.sub('[n\{\}\[\]\/?,.;:|\)*~`!^\-_+<>@\#$%&\\\=\(\'\"]', '', data).strip()
    emoticon_pattern = re.compile("["
                                  u"\U0001F600-\U0001F64F"
                                  u"\U0001F300-\U0001F5FF"
                                  u"\U0001F680-\U0001F6FF"
                                  u"\U0001F1E0-\U0001F1FF"
                                  "]+", flags=re.UNICODE)
    comment = emoticon_pattern.sub(r'', comment)
    return comment.strip()


if __name__ == "__main__":
    load_data_and_labels("data/train.csv")