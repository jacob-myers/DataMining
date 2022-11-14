import pandas as pd
import numpy as np

def import_and_clean_irs(filename: str):
    # CLEAN COLUMNS
    # Reading in data sheet with a multi-level column index.
    # https://stackoverflow.com/questions/69855943/pandas-read-csv-with-columns-name-spread-over-multiple-rows
    df = pd.read_excel(filename, skiprows=3, header=[0, 1, 2])

    # Drop the garbage column index (3rd level of column index).
    # https://www.geeksforgeeks.org/how-to-drop-a-level-from-a-multi-level-column-index-in-pandas-dataframe/
    df.columns = df.columns.droplevel(2)

    # Merge level 1 and 2 of multi-level column index to make single level.
    # https://stackoverflow.com/questions/24290297/pandas-dataframe-with-multiindex-column-merge-levels
    df.columns = df.columns.map(' | '.join).str.strip(' | ')

    # Clean up the names (remove '[number]' and Unnamed columns (from multi-level step)
    # https://stackoverflow.com/questions/61653697/pandas-str-replace-with-regex
    df.columns = df.columns.str.replace(r' \| Unnamed.*', '', regex=True)
    df.columns = df.columns.str.replace(r' \[\d*\]', '', regex=True)
    df.columns = df.columns.str.replace(r'\\n', '', regex=True)

    # CLEAN DATA
    # TO DO

    return df


if __name__ == '__main__':
    df_2019 = import_and_clean_irs("nh_2019.xlsx")
    df_2015 = import_and_clean_irs("nh_2015.xlsx")
    #Income tax before credit (2019) = Income tax (2015)

    cols_2015 = np.array(df_2015.columns)
    #print(cols_2015)

    for col in df_2015.columns:
        if col not in df_2019.columns:
            print(col)