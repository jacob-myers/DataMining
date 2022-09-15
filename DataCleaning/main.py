"""Program for cleaning up the Iris data set and exporting the clean csv.

    Author: Jake Myers
    Class: DAT-330
    Assignment: Data Cleaning and Packages

Certification of Authenticity:
I certify that this is entirely my own work, except where I have given
fully-documented references to the work of others. I understand the definition
and consequences of plagiarism and acknowledge that the assessor of this
assignment may, for the purpose of assessing this assignment:
- Reproduce this assignment and provide a copy to another member of academic
- staff; and/or Communicate a copy of this assignment to a plagiarism checking
- service (which may then retain a copy of this assignment on its database for
- the purpose of future plagiarism checking)
"""

import numpy as np
import scipy as sp
import pandas as pd
import matplotlib.pyplot as plt


def convert_to_float(str_number: str) -> float:
    """Fix a few issues with numbers identified in the provided dataset.

    Replaces ',' and ';' found in the numbers with '.' and then converts them
    to floats. If they fail to convert, returns None.

    Arguments:
        str_number (str): The number in string format.

    Returns:
        float of the corrected number. or None if it still cannot convert.
    """
    str_number = str_number.replace(',', '.')
    str_number = str_number.replace(';', '.')
    try:
        return float(str_number)
    except ValueError:
        return None


def uniformize_variety(str_var: str) -> str:
    """Uniformize names to be consistent in the variety column in the data.

    Many names in the variety column were incomplete strings. These are to be
    uniformized to be the same.

    Arguments:
        str_var (str): The 'variety' from original data set.
    Returns:
        The corrected/uniformized variety name.
    """
    if type(str_var) is not str:
        return None
    if 'Virgi' in str_var:
        return 'Virginica'
    if 'Seto' in str_var:
        return 'Setosa'
    else:
        return str_var


if __name__ == '__main__':
    data = pd.read_csv("Irish_dataset.csv")
    # Rename columns to a uniform.
    data.rename(columns={'sepal,length': 'sepal_length',
                         'sepal,width': 'sepal_width',
                         'petal.length': 'petal_length',
                         'petal.width': 'petal_width'}, inplace=True)

    # Docs for apply: https://pandas.pydata.org/pandas-docs/stable/reference/api/pandas.Series.apply.html
    data.sepal_length = data.sepal_length.apply(convert_to_float)
    data.sepal_width = data.sepal_width.apply(convert_to_float)
    data.petal_length = data.petal_length.apply(convert_to_float)
    data.petal_width = data.petal_width.apply(convert_to_float)
    data.variety = data.variety.apply(uniformize_variety)

    # Docs for dropna: https://pandas.pydata.org/pandas-docs/stable/reference/api/pandas.DataFrame.dropna.html
    data = data.dropna(how='any', axis=0, subset=['variety'])

    # The invalid data is left as None type, rather than drop the row.
    data.to_csv("Iris_dataset_cleaned.csv", index=False)


"""
Notes

iloc, for getting a value from a cell:
https://www.delftstack.com/howto/python-pandas/how-to-get-a-value-from-a-cell-of-a-dataframe/

Potential ways of dropping data:
https://www.golinuxcloud.com/pandas-drop-rows-examples/

# Prints the one given column
print(data['my_column'].to_string(index=False))
# Prints the one given column WITH header
print(df[['my_column']].to_string(index=False))

# converts everything in column to the given type
data.astype({'sepal_length': 'int32'})
"""

