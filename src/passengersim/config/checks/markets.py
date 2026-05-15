import warnings
from collections import defaultdict
from typing import Any

from passengersim.config import Config


def check_min_fare_price_by_market(cfg: Config) -> dict[str, dict[str, Any]]:
    """
    Identify the minimum price fare for each market.

    Parameters
    ----------
    cfg : Config

    Returns
    -------
    dict[str, dict[str, Any]]
         A dictionary mapping market identifiers (in the format "origin~destination") to a
         dictionary containing the minimum fare price for that market, as well as the
         booking class and carrier for that minimum fare.
    """
    min_prices = {}
    for f in cfg.fares:
        mkt = f"{f.orig}~{f.dest}"
        if mkt not in min_prices or min_prices[mkt]["min_price"] > f.price:
            min_prices[mkt] = {"min_price": f.price, "booking_class": f.booking_class, "carrier": f.carrier}
    return min_prices


def check_markets_without_fares(cfg: Config, *, clean: bool = False, inplace: bool = True) -> Config:
    """Check for markets without fares from the config.

    Parameters
    ----------
    cfg : Config
    clean : bool, default False
        If True, remove markets without fares from the config. If False, raise an error if markets
        without fares are found.
     inplace : bool, default True
        If True and `clean` is also True, modify the input config in place. If False, return a
        modified copy of the config.  This has no effect if `clean` is False, since no modifications
        are made to the config in that case.

    Returns
    -------
    Config
         The input config, potentially modified to remove markets without fares if `clean` is True.

    Raises
    ------
    ValueError
        If markets without fares are found, and `clean` is False.
    """

    if not inplace:
        cfg = cfg.model_copy(deep=True)

    fares_by_market = defaultdict(list)
    for fare in cfg.fares:
        fares_by_market[f"{fare.orig}~{fare.dest}"].append(fare)

    markets_with_fare = []
    markets_without_fare = []
    for mkt in cfg.markets:
        if mkt.identifier in fares_by_market:
            markets_with_fare.append(mkt.identifier)
        else:
            markets_without_fare.append(mkt.identifier)

    if clean:
        if len(markets_without_fare):
            warnings.warn(f"Markets without fares: {len(markets_without_fare)}", stacklevel=2)
        cfg.markets = [m for m in cfg.markets if m.identifier in markets_with_fare]
    else:
        if len(markets_without_fare):
            raise ValueError(
                f"found {len(markets_without_fare)} markets without fares, including {markets_without_fare[:3]}"
            )
    return cfg


def check_market_max_flows(cfg: Config) -> dict[str, int]:
    """Check the maximum capacity when fully utilizing every path the market."""
    try:
        import networkx as nx
    except ImportError:
        raise ImportError("networkx not installed, it is required for this check") from None

    max_flows = {}
    for mkt in cfg.markets:
        orig = mkt.orig
        dest = mkt.dest
        G = nx.DiGraph()
        for pth in cfg.paths.set_filters(orig=orig, dest=dest):
            node_a = "source"
            for leg_id in pth.legs:
                leg = cfg.legs.select(leg_id=leg_id)
                node_b = f"leg-{leg.leg_id}"
                G.add_edge(node_a, node_b, capacity=leg.total_capacity)
                node_a = node_b
            G.add_edge(node_a, "sink")
        flow_value, _ = nx.algorithms.flow.maximum_flow(G, "source", "sink")
        max_flows[mkt.identifier] = flow_value
    return max_flows
