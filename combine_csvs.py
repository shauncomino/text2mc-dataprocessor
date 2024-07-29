import pandas as pd
import sys

def main():

    '''
        Usage:
        Run `python combine_csvs.py path/to/combined/result.csv source_df1.csv source_df2.csv ...`
    '''

    n = len(sys.argv)
    final_save_path = sys.argv[1]

    df_paths = [sys.argv[i] for i in range(2, n)]
    
    dfs = [pd.read_csv(path) for path in df_paths]

    big_df = pd.concat(dfs)
    big_df = big_df.dropna(subset=["PROCESSED_PATHS"])
    big_df.to_csv(final_save_path)


if __name__ == "__main__":
    main()