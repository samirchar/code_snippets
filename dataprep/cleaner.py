from collections import defaultdict
from functools import partial
import multiprocessing as mp
from sklearn.preprocessing import LabelEncoder
import pandas as pd
import numpy as np
from fuzzywuzzy.process import extractOne
from unidecode import unidecode


class ReduceLevels:
    """Class that reduces the levels in a categorical variable by applying
    pareto principle (sort of). It preserves most frequent categories and
    groups least frequent categories in a new category called 'other'

    :return: :class:`~ReduceLevels`
    """
    

    def __init__(self):
        """init
        """
        self.mapper = {}

    def fit(self, dataframe, percentage_to_keep):
        """given a column and de percentage to keep, it determines which categories should be
        combined in a category called 'other'. It stores the levels we want to keep, it 
        doesnt transform data.
        Parameters
        
        :param dataframe: the dataframe containing the columns we want to revise
        :type dataframe: :py:class:`pandas.DataFrame`
        :param percentage_to_keep: a dictionary where keys are the name of 
            the columns we want to fit and the values are the percentage we want to keep.
            This percentage is number between 0 and 1 representing the percentage we want to preserve.

             for example:
             {'BRAND_NAME':0.8,'MUNICIPIO':0.9} 

             Of column 'BRAND_NAME' we are keeping 80% and with column 'MUNICIPIO' we are
             keeping 90%
        :type percentage_to_keep: dict
        """
        data = dataframe.copy()

        for col, p in percentage_to_keep.items():
            data[col] = data[col].astype('category')
            # Stores all categoris in an array.
            frequency_table = data[col].value_counts()/data.shape[0]
            percentage_kept_index = np.where(
                frequency_table.cumsum() >= p)[0][0]
            self.mapper[col] = frequency_table.index[0:percentage_kept_index+1].tolist()

    def transform(self, dataframe):
        """it transforms the least frequent categories into a single
        category called 'other' given the levels we want to keep in the dictionary returned by
        'fit' method
        
        :param dataframe: the dataframe containing the columns we want to revise
        :type dataframe: :py:class:`pandas.DataFrame`
        :return: Same dataframe given at input but with transformed columns
        :rtype: :py:class:`pandas.DataFrame`
        """
        data = dataframe.copy()

        for col, levels_to_keep in self.mapper.items():

            if levels_to_keep:
                data.loc[~data[col].isin(levels_to_keep), col] = 'other'

            else:
                print('Need to fit first! Or use fit_transform method')

        return data

    def fit_transform(self, dataframe, percentage_to_keep):
        """given a column and de percentage to keep, it determines which categories should be
        combined in a category called 'other'. It stores the levels we want to keep, it 
        doesnt transform data.
        Parameters
        
        :param dataframe: the dataframe containing the columns we want to revise
        :type dataframe: :py:class:`pandas.DataFrame`
        :param percentage_to_keep: a dictionary where keys are the name of 
            the columns we want to fit and the values are the percentage we want to keep.
            This percentage is number between 0 and 1 representing the percentage we want to preserve.

             for example:
             {'BRAND_NAME':0.8,'MUNICIPIO':0.9} 

             Of column 'BRAND_NAME' we are keeping 80% and with column 'MUNICIPIO' we are
             keeping 90%
        :type percentage_to_keep: dict
        :param dataframe: the dataframe containing the columns we want to revise
        :type dataframe: :py:class:`pandas.DataFrame`
        :return: Same dataframe given at input but with transformed columns
        :rtype: :py:class:`pandas.DataFrame`
        """
        self.fit(dataframe, percentage_to_keep)
        return self.transform(dataframe)

class FuzzyCorrector:
    """Corrects typos in categorical variables
    givel the possible values for each column using
    Fuzzy Matching (with Levenshtein distance)
    
    :return: :class:`~FuzzyCorrector`
    """

    def __init__(self, corrector):
        """Class initializer. 
        
        :param corrector: Dictionary where keys should be the columns we want to correct and values are a tuple.
            The first element of the tuple should be a list of the available categories in the column
            while the second item of the tuple is the score_cutoff. For a string to be corrected its score
            should be equal or greater than the score_cutoff, else it is set as NaN. 
            score_cutoff is a value between 0 and 100.

            For example: 

            corrector = {'BRAND_NAME':(['muy bueno','bueno','regular','nuevo','alto riesgo'],60),
                         'BLUETOOTH':(['y','n'],60)}

        :type corrector: dict
        """
        self.corrector = corrector

    def single_fuzzy_corrector(self, available_categories, score_cutoff, str2match, return_score = False):
        """Function that corrects values with typos in
        categorical variables. Given a list of posible categories
        the function evaluates the similarity of 'str2match' to this 
        possible categories. If the similarity score of the category
        with the highest score passes a threshold it returns this category as output,
        else it returns nan
        
        :param available_categories: A list of possible categorires
        :type available_categories: list
        :param score_cutoff: score threshold. If the best
            match is found, but it is not greater than this number, then
            return None anyway ("not a good enough match").
        :type score_cutoff: int
        :param str2match: The string we want to match against
        :type str2match: str
        :return: returns the matched category or np.nan if didnt pass the threshold.
        :rtype: str
        """
        if str2match == '<NAN>':
            return np.nan
        else:
            cat = extractOne(str2match, available_categories,
                                 score_cutoff=score_cutoff)
            if cat:
                if return_score:
                    return cat[0],cat[1]
                else:
                    return cat[0]
            else:
                return np.nan

    def single_col_fuzzy_corrector(self, dataframe, column, available_categories, score_cutoff):
        """corrects errors from a pandas column using fuzzy matching given
        the set of possible categories.
        
        :param dataframe: the dataframe containing the columns we want to correct
        :type dataframe: :py:class:`pandas.DataFrame`
        :param column: the name of the column in the dataframe that we want to correct
        :type column: str
        :param available_categories: A list of possible categorires
        :type available_categories: list
        :param score_cutoff: score threshold. If the best
            match is found, but it is not greater than this number, then
            return None anyway ("not a good enough match").
        :type score_cutoff: int
        """
        assert isinstance(self.corrector,dict), "To use this function, corrector must be a dict"
        
        dataframe[column] = dataframe[column].fillna('<NAN>')
        strs2match = dataframe[column].tolist()

        max_pool = mp.cpu_count()
        func = partial(self.single_fuzzy_corrector,
                       available_categories, score_cutoff)
        with mp.Pool(processes=max_pool) as pool:
            corrected_values = pool.map(func, strs2match)
        dataframe.drop([column], axis=1, inplace=True)
        dataframe[column] = corrected_values

    def transform(self, dataframe):
        """Corrects entries of the dataframe from the columns specified in the corrector.
        
        :param dataframe: the dataframe containing the columns we want to correct
        :type dataframe: :py:class:`pandas.DataFrame`
        :return: the corrected dataframe
        :rtype: :py:class:`pandas.DataFrame`
        """
        df = dataframe.copy()

        for key, value in self.corrector.items():
            self.single_col_fuzzy_corrector(df, key, value[0], value[1])

        return df

def upper_bound(x):
    """Determines the upper bound to consider a value "normal" (i.e. not outliers)
    considering the interquantile range definition.
    
    :param x: The column of the pandas dataframe we want to check the upper bound.
    :type x: :py:class:`pandas.Series`
    :return: the value of the upper bound
    :rtype: float
    """
    return x.quantile(0.75) + 2 * (x.quantile(0.75) - x.quantile(0.25))


def lower_bound(x):
    """Determines the lower bound to consider a value "normal" (i.e. not outliers)
    considering the interquantile range definition.
    
    :param x: The column of the pandas dataframe we want to check the lower bound.
    :type x: :py:class:`pandas.Series`
    :return: the value of the lower bound
    :rtype: float
    """
    return x.quantile(0.25) - 2 * (x.quantile(0.75) - x.quantile(0.25))


def normalize_str_cols(df,include_cols = None, exclude_cols = None):
    
    df_2 = df.copy()
    
    if include_cols:
        str_cols = include_cols
    else:
        str_cols = df_2.dtypes[df_2.dtypes == 'object'].index
        
        if exclude_cols:
            str_cols =[i for i in str_cols if i not in exclude_cols]
            
    #Remove accents
    df_2[str_cols] = df_2[str_cols].apply(lambda x: x.apply(unidecode))
    
    #Convert columns to lowercase
    df_2[str_cols] = df_2[str_cols].apply(lambda x: x.str.lower())
    
    #Trim trailing and leading spaces
    df_2[str_cols] = df_2[str_cols].apply(lambda x: x.str.strip())
    
    return df_2

