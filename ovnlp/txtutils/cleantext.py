import re
from nltk.corpus import stopwords
from string import punctuation
from nltk.stem.snowball import SnowballStemmer
from nltk import RegexpTokenizer
import unicodedata


class LangTools(object):
    def __init__(self, iLang="en"):
        self.language = iLang

    def get_stop_words(self, iCustomPonct=['«', '’', '»', ':', '–', '-', '`', '\"', "``"],
                       iCustomList=None):
        if self.language == "fr":
            if iCustomList is None:
                iCustomList = ['le', 'la', 'les', 'un', 'une', 'des']
            return set(stopwords.words('french') + list(punctuation) + iCustomPonct + iCustomList)
        elif self.language == "en":
            if iCustomList is not None:
                return set(stopwords.words('french') + list(punctuation) + iCustomPonct + iCustomList)
            else:
                return set(stopwords.words('english') + list(punctuation) + iCustomPonct)
        else:
            return set(list(punctuation) + iCustomPonct)

    def get_stemmer(self):
        if self.language == "fr":
            stemmer = SnowballStemmer("french")
        elif self.language == "en":
            stemmer = SnowballStemmer("english")
        else:
            print("No stemmer found for this language.")
            return
        return stemmer

    def get_tokenizer(self, iRegex=None):
        if iRegex is not None:
            tokenizer = RegexpTokenizer(iRegex)
        else:
            if self.language == "fr":
                tokenizer = RegexpTokenizer(r'''\w'|\w+|[^\w\s]''')
            elif self.language == "en":
                tokenizer = RegexpTokenizer('\w+|\$[\d\.]+|\S+')
            else:
                tokenizer = RegexpTokenizer('\w+|\$[\d\.]+|\S+')
        return tokenizer


def strip_accents(iText):
    """
    Strip accents from input String.

    :param iText: The input string.
    :type iText: String.

    :returns: The processed String.
    :rtype: String.
    """
    try:
        iText = unicode(iText, 'utf-8')
    except (TypeError, NameError):  # unicode is a default on python 3
        pass
    oText = unicodedata.normalize('NFD', iText)
    oText = oText.encode('ascii', 'ignore')
    oText = oText.decode("utf-8")
    return str(oText)


def tokenize(iSentence, iTokenizer, iStopWords, iStemmer=None):
    words = iTokenizer.tokenize(iSentence)
    words = [strip_accents(w.lower().replace("_", "")) for w in words]
    wordslist = [w for w in words if w not in iStopWords and not w.isdigit()]
    if iStemmer is not None:
        wordslist = [iStemmer.stem(w) for w in wordslist]
    else:
        pass
    return wordslist


def text_file_to_sentences(iTextfilePath, iTokenizer, iStopWords, iSplitter="\.|\?|!|\n", iStemmer=None):
    """

    :param iTextfilePath:
    :param iStopWords:
    :param iStemmer:

    # may use https://radimrehurek.com/gensim/models/word2vec.html#gensim.models.word2vec.LineSentence
    :return:
    """
    with open(iTextfilePath, 'r') as myfile:
        texte = myfile.read().replace('\n', '')
    return string_to_sentences(texte, iTokenizer, iStopWords, iSplitter, iStemmer)


def string_to_sentences(iString, iTokenizer, iStopWords, iSplitter="\.|\?|!|\n", iStemmer=None):
    ss = re.split(iSplitter, iString)
    sentences = []
    for s in ss:
        sentences.append(tokenize(s, iTokenizer, iStopWords, iStemmer))
    return [sentence for sentence in sentences if len(sentence) > 0]


def main():
    tx = LangTools("fr")
    print(tx.get_stop_words())
    return


if __name__ == "__main__":
    main()
