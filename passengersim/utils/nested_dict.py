import pandas as pd


def from_nested_dict(content: dict, dims: list[str]) -> pd.DataFrame:
    """
    Convert a nested dictionary to a pandas DataFrame.

    Parameters
    ----------
    content : dict
        The nested dictionary to convert.
    dims : list[str]
        The names of the dimensions of the DataFrame.  These should match the
        keys of the dictionary, in order.

    Returns
    -------
    pd.DataFrame
        The DataFrame representation of the nested dictionary.
    """
    if len(dims) == 2:
        return pd.DataFrame.from_dict(content, orient="index").rename_axis(
            index=dims[0], columns=dims[1]
        )
    elif len(content) == 0:
        return pd.DataFrame(
            index=pd.MultiIndex(
                levels=[[] for _ in dims], codes=[[] for _ in dims], names=dims
            )
        )
    else:
        raw = {}
        for k, v in content.items():
            raw[k] = from_nested_dict(v, dims[1:])
        return pd.concat(raw, axis=0, names=[dims[0]])
