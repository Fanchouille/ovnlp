import pandas as pd
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from scipy.sparse import csr_matrix
from . import disjointSet


class TextMatcher(object):
    # See https://labs.yodas.com/large-scale-matrix-multiplication-with-pyspark-or-how-to-match-two-large-datasets-of-company-1be4b1b2871e
    def __init__(self, input_dfs, text_cols, id_cols, group_cols=None, analyzer='word', ngram_range=(1, 1),
                 stop_words=None, max_features=5000):
        """
        Initialize :
        input_dfs : tuple of 2 dataframes (iDf1, iDf2) - DFs with id & text cols => we want to match items based on text
        cols
        text_cols : tuple of 2 strings (iTextCol1, iTextCol2) - names of cols containing the text to match on
        id_cols : tuple of 2 strings (iIdCol1, iIdCol2) - names of cols containing IDs
        group_cols : tuple of 2 strings : None by default : if not None, use these cols to create grouped DataFrame and
        get matches only in the same groups
        stop_words : if not None, will filter out stop words before matching
        analyzer : 'word' by default to tokenize by word : see options on scikit doc
        ngram_range : (1,1) by default to keep only unigram : see options on scikit doc
        max_features : 5000 by default to keep only top 5000 words : see options on scikit doc
        """
        self.input_dfs = input_dfs
        self.text_cols = text_cols
        self.id_cols = id_cols
        self.stop_words = stop_words
        self.analyzer = analyzer
        self.ngram_range = ngram_range
        self.max_features = max_features
        self.group_cols = group_cols
        return

    def get_vocabulary(self, iDfTuple):
        """
        Concatenate all texts and create vocabulary from unique
        :return:
        """
        vect = CountVectorizer(stop_words=self.stop_words, analyzer=self.analyzer, ngram_range=self.ngram_range,
                               max_features=self.max_features)
        all_texts = np.unique(np.stack((iDfTuple[0].loc[:, self.text_cols[0]].values,
                                        iDfTuple[1].loc[:, self.text_cols[1]].values)).flatten())
        vocabulary = vect.fit(all_texts).vocabulary_
        return vocabulary

    def get_tfidf_matrices(self, iDfTuple):
        """
        :param iDfTuple: tuple of Df
        :return: Creates the 2 TFIdf matrices
        """
        vocabulary = self.get_vocabulary(iDfTuple)
        tfidf_vect = TfidfVectorizer(stop_words=self.stop_words, vocabulary=vocabulary, analyzer=self.analyzer,
                                     ngram_range=self.ngram_range)
        XTf1 = tfidf_vect.fit_transform(iDfTuple[0].loc[:, self.text_cols[0]].values)
        XTf2 = tfidf_vect.fit_transform(iDfTuple[1].loc[:, self.text_cols[1]].values)
        return (XTf1, XTf2)

    def parallelize_matrix(self, scipy_mat, rows_per_chunk):
        """
        :param scipy_mat: Scipy Matrix
        :param rows_per_chunk: # of rows per chunk
        :return: Creates chunks of matrix from a big one : with index / sparse sub matrix / shape
        """
        [rows, cols] = scipy_mat.shape
        i = 0
        submatrices = []
        while i < rows:
            current_chunk_size = min(rows_per_chunk, rows - i)
            submat = scipy_mat[i:i + current_chunk_size]
            submatrices.append((i,
                                (submat.data, submat.indices, submat.indptr),
                                (current_chunk_size, cols)
                                )
                               )
            i += current_chunk_size
        return submatrices

    def find_matches_in_submatrix(self, sources, targets, inputs_start_index, threshold):
        """
        :param sources:
        :param targets:
        :param inputs_start_index:
        :param threshold:
        :return:
        """
        cosimilarities = cosine_similarity(sources, targets)
        for i, cosimilarity in enumerate(cosimilarities):
            cosimilarity = cosimilarity.flatten()
            # Find the best matches
            target_index = np.where(cosimilarity >= threshold)[0]
            source_index = inputs_start_index + i
            if target_index.shape[0] > 0:
                for idx in target_index:
                    similarity = cosimilarity[idx]
                    if (source_index != idx):
                        yield (source_index, idx, similarity)

    def get_unit_results(self, iDfTuple, rows_per_chunk, threshold, add_groups):
        """
        :param iDfTuple: input Dfs
        :param rows_per_chunk: # of rows per chunk
        :param threshold: 1 for a perfect match
        :return: a DF with 1st Id / 1st Text / 2nd Id / 2nd Text / Cosine Sim
        """
        tf_matrices = self.get_tfidf_matrices(iDfTuple)
        all_matrices = self.parallelize_matrix(tf_matrices[0], rows_per_chunk)

        results_generator = (self.find_matches_in_submatrix(csr_matrix(matrix[1], shape=matrix[2]),
                                                            tf_matrices[1],
                                                            matrix[0],
                                                            threshold=threshold)
                             for matrix in all_matrices)

        nearest_frn = []
        for res in results_generator:
            for x in res:
                nearest_frn.append((iDfTuple[0].loc[:, self.id_cols[0]].iloc[x[0]],
                                    iDfTuple[1].loc[:, self.id_cols[1]].iloc[x[1]],
                                    iDfTuple[0].loc[:, self.text_cols[0]].iloc[x[0]],
                                    iDfTuple[1].loc[:, self.text_cols[1]].iloc[x[1]],
                                    x[2]
                                    )
                                   )

        if self.id_cols[0] == self.id_cols[1]:
            id_col_1 = self.id_cols[0] + '_1'
            id_col_2 = self.id_cols[0] + '_2'
        else:
            id_col_1 = self.id_cols[0]
            id_col_2 = self.id_cols[1]

        if self.text_cols[0] == self.text_cols[1]:
            text_col_1 = self.text_cols[0] + '_1'
            text_col_2 = self.text_cols[0] + '_2'
        else:
            text_col_1 = self.text_cols[0]
            text_col_2 = self.text_cols[1]

        lCols = [id_col_1, id_col_2, text_col_1, text_col_2, 'COSINE_SIM']
        oDf = pd.DataFrame(nearest_frn, columns=lCols)

        if oDf.shape[0] > 0:
            oDf.loc[:, id_col_1 + '_' + id_col_2 + '_tuple'] = oDf.apply(
                lambda x: frozenset({x[id_col_1], x[id_col_2]}),
                axis=1)

            # Add disjoint groups
            if add_groups:
                x = disjointSet.DisjointSet()
                for ft in oDf.loc[:, id_col_1 + '_' + id_col_2 + '_tuple'].unique():
                    st = list(ft)
                    x.add(st[0], st[1])

                lead = x.leader
                groupDict = x.group
                oDf.loc[:, 'groupId'] = oDf.loc[:, id_col_1].map(lambda x: lead[x])
                oDf.loc[:, 'group'] = oDf.loc[:, 'groupId'].map(lambda x: tuple(groupDict[x]))
                return oDf.loc[:, lCols + [id_col_1 + '_' + id_col_2 + '_tuple', 'groupId', 'group']]

            else:
                return oDf.loc[:, lCols + [id_col_1 + '_' + id_col_2 + '_tuple']]

    def get_results(self, rows_per_chunk=100, threshold=0.8, add_groups=False):
        """

        :param rows_per_chunk: # of rows per chunk
        :param threshold: 1 for a perfect match
        :param add_groups : boolean, use disjointSet class to build a groupId col (from one of the mateches) and group
        col (with composition of group)
        :return: a DF with 1st Id / 1st Text / 2nd Id / 2nd Text / Cosine Sim
        """
        if self.group_cols is not None:
            resDfList = []
            groupNames = self.input_dfs[0].loc[:, self.group_cols[0]].unique()
            for group in groupNames:
                currentDfTuple = (self.input_dfs[0].groupby(self.group_cols[0]).get_group(group),
                                  self.input_dfs[1].groupby(self.group_cols[1]).get_group(group))
                resDfList.append(
                    self.get_unit_results(currentDfTuple, rows_per_chunk=rows_per_chunk, threshold=threshold,
                                          add_groups=add_groups))
            oDf = pd.concat(resDfList)
        else:
            oDf = self.get_unit_results(self.input_dfs, rows_per_chunk=rows_per_chunk, threshold=threshold,
                                        add_groups=add_groups)

        return oDf
