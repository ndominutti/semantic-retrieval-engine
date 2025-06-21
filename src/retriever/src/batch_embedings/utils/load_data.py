import pandas as pd


def load_data_from_csv(csv_path: str) -> pd.DataFrame:
    """Sample function for the PoC. Reads the data inside the test csv file.

    Args:
        dataframe_path (str): path to the dataframe to be loaded

    Returns:
        pd.DataFrame: loaded dataframe
    """
    return pd.read_csv(csv_path, sep="\t")


# In production for AWS this can be a boto3 request to download data from s3.
def load_data_from_s3():
    """PLACEHOLDER"""
    pass
