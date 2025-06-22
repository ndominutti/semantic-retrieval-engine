class MissingColumnsError(Exception):
    """Exception to be raised when the selected columns to embed are not present in the dataset"""

    pass
