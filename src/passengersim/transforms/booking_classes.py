from __future__ import annotations

import re
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from passengersim import Config


def pad_digits(strings):
    """
    Transform strings so that numeric suffixes are padded with zeros.

    If every string in the list matches a pattern where it begins
    with one or more letters, then has one or more digits, then
    rewrite all the strings in the list so that there are the same
    number of digits at the end of every string, padding with zeros
    as needed.

    Parameters
    ----------
    strings : list of str
        The list of strings to transform.

    Returns
    -------
    dict
        Mapping the original values to the new values.

    Example
    -------
    >>> pad_digits(['A1', 'B2', 'C33'])
    {'A1': 'A01', 'B2': 'B02', 'C33': 'C33'}
    """
    pattern = re.compile(r"^([A-Za-z]+)(\d+)$")
    matches = [pattern.match(s) for s in strings]
    if not all(matches):
        return {s: s for s in strings}  # No change if any string doesn't match

    max_digits = max((len(m.group(2)) for m in matches if m), default=0)
    result = {}
    for s, m in zip(strings, matches):
        prefix, digits = m.group(1), m.group(2)
        new_digits = digits.zfill(max_digits)
        result[s] = f"{prefix}{new_digits}"
    return result


def class_rename(cfg: Config) -> Config:
    """
    Rename booking classes in fares, classes, and carriers.

    This will pad the numeric suffixes of booking classes
    to ensure they have the same number of digits. This is useful
    for ensuring consistent ordering and reporting of booking classes,
    especially when they are used in reports or visualizations that
    are typically sorted in lexicographic (generalized alphabetic) order.

    Parameters
    ----------
    cfg : Config
        The configuration object containing fares, classes, and carriers.

    Returns
    -------
    Config
        The updated configuration object with zero-padded booking classes.
    """

    # collect all booking classes
    booking_classes = set(f.booking_class for f in cfg.fares) | set(cfg.classes)
    for carrier in cfg.carriers.values():
        booking_classes.update(carrier.classes)

    # pad digits in booking classes
    padded_classes = pad_digits(list(booking_classes))

    # change booking classes in fares, classes, and carriers
    for f in cfg.fares:
        f.booking_class = padded_classes.get(f.booking_class, f.booking_class)
    cfg.classes = [padded_classes.get(c, c) for c in cfg.classes]
    for carrier in cfg.carriers.values():
        carrier.classes = [padded_classes.get(c, c) for c in carrier.classes]
    return cfg
