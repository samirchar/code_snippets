from collections import defaultdict
from functools import partial
import multiprocessing as mp
from sklearn.preprocessing import LabelEncoder
import pandas as pd
import numpy as np
from fuzzywuzzy.process import extractOne

class AwesomeLabelEncoder:
    """A wrapper of sklearn for multiple columns and some other perks
    
    :return: :class:`~AwesomeLabelEncoder`
    """

    def __init__(self):
        """init
        """
        self.encoder = None

    def fit(self, dataframe, cols):
        """Fits a label encoding of the columns given in 'cols'. It stores a dictionary
        with the encoders in self.encoder.
        
        :param dataframe: the dataframe containing the columns we want to encode
        :type dataframe: :py:class:`pandas.DataFrame`
        :param cols: a list of the columns we want to encode
        :type cols: list
        """
        self.encoder = defaultdict(LabelEncoder)
        data = dataframe.copy()

        # Fit encoder to the variables
        data[cols].apply(lambda x: self.encoder[x.name].fit(x.dropna()))

    def transform(self, dataframe):
        """using the encoding dictionary stored in self.encoder it encodes
        the corresponding columns of the dataframe
        
        :param dataframe: the dataframe containing the columns we want to encode
        :type dataframe: :py:class:`pandas.DataFrame`
        :return: the same dataframe given as input but with the columns label encoded
        :rtype: :py:class:`pandas.DataFrame`
        """
        if self.encoder:
            cols = list(self.encoder.keys())
            data = dataframe.copy()
            
            # Transform data using de dictionary of fitted columns
            for c in cols:
                data.loc[~data[c].isna(),c]=self.encoder[c].transform(data.loc[~data[c].isna(),c])
                
            return data
        else:
            print('Need to fit first! Or use fit_transform method')

    def fit_transform(self, dataframe, cols):
        """fits and then transforms data
        
        :param dataframe: the dataframe containing the columns we want to encode
        :type dataframe: :py:class:`pandas.DataFrame`
        :param cols: a list of the columns we want to encode
        :type cols: list
        :return: the same dataframe given as input but with the columns label encoded
        :rtype: :py:class:`pandas.DataFrame`
        """
        self.fit(dataframe, cols)
        return self.transform(dataframe)

    def inverse_transform(self, dataframe):
        """decodes a preiously transformed (encoded) dataframe using
        the dictionary stored in self.encoder.
        
        :param dataframe: a dataframe with encoded columns
        :type dataframe: :py:class:`pandas.DataFrame`
        :return: a decoded dataframe
        :rtype: :py:class:`pandas.DataFrame`
        """
        data = dataframe.copy()
        cols = list(self.encoder.keys())
        mask = ~data[cols].isna().any(axis=1)
        # Inverse transform encoded columns
        data.loc[mask,cols] = data.loc[mask,cols].apply(
            lambda x: self.encoder[x.name].inverse_transform( x.astype(int) ))
        return data
        
def cyclical_encoding(df, col, max_val):
    """Encoding of cyclical features using sine and cosine transformation.
    Examples of cyclical features are: hour of day, month, day of week.
    The variable must be a numeric integer starting from 0.

    :param df: A dataframe containing the column we want to encode
    :type df: :py:class:`pandas.DataFrame`
    :param col: The name of the column we want to encode.
    :type col: str
    :param max_val: The maximum value the variable can have. e.g. in hour of day, max value = 23
    :type max_val: int
    :return: dataframe with three new variables: sine and cosine of the features + the multiplication
    of these two columns
    :rtype: :py:class:`pandas.DataFrame`
    """

    data = df.copy()

    data[col] = data[col] - min(data[col])
    data[col] = data[col] - min(data[col])
    data[col + '_sin'] = np.sin(2 * np.pi * data[col] / max_val)
    data[col + '_cos'] = np.cos(2 * np.pi * data[col] / max_val)
    data[col + '_sin_cos_cross'] = data[col + '_sin'] * data[col + '_cos']

    return data

