import numpy as np


def word_to_vec(iWord, iModel, iNormed=True):
    """

    :param iWord:
    :param iModel:
    :param iNormed:
    :return:
    """
    try:
        if iNormed:
            if np.linalg.norm(iModel.wv[iWord]) > 0:
                return iModel.wv[iWord] / np.linalg.norm(iModel.wv[iWord])
            else:
                return np.zeros(iModel.wv.vector_size)
        else:
            return iModel.wv[iWord]
    except:
        return np.zeros(iModel.wv.vector_size)


def wordlist_to_vec(iWordList, iModel, iNormed=True):
    """
    For a sentence, get the sentence vector by averaging all word vectors
    :param iWordList:
    :param iModel:
    :param iNormed:
    :return:
    """
    cleaned_list = []
    for lWord in iWordList:
        cleaned_list.append(word_to_vec(lWord, iModel, iNormed))
    wordVecs = np.array(cleaned_list)
    wordVecsNorms = np.array([np.linalg.norm(x) for x in wordVecs])
    nonNullWV = wordVecs[np.where(wordVecsNorms > 0.0)]
    if iNormed:
        if nonNullWV.shape[0] > 0:
            nonNullWV /= wordVecsNorms[np.where(wordVecsNorms > 0.0)].reshape(-1, 1)
            return nonNullWV.mean(axis=0)
        else:
            return np.zeros(iModel.wv.vector_size)
    else:
        return nonNullWV.mean(axis=0)
