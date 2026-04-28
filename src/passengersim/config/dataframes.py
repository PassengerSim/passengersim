import pandas as pd
import yaml

from .booking_curves import BookingCurve
from .demands import Demand
from .fares import Fare
from .frat5_curves import Frat5Curve
from .legs import Leg
from .places import Place


def legs_to_dataframe(legs: list[Leg]) -> pd.DataFrame:
    """Convert a list of Leg objects into a pandas DataFrame."""
    data = []
    for leg in legs:
        row = {
            "leg_id": leg.leg_id if leg.leg_id is not None else leg.fltno,
            "orig": leg.orig,
            "dest": leg.dest,
            "orig_timezone": leg.orig_timezone,
            "dest_timezone": leg.dest_timezone,
            "carrier": leg.carrier,
            "fltno": leg.fltno,
            "distance": leg.distance,
            "capacity": leg.capacity,
            "date": leg.date,
            "dep_time": leg.dep_time,
            "arr_time": leg.arr_time,
            "dep_time_offset": leg.dep_time_offset,
            "arr_time_offset": leg.arr_time_offset,
            "time_adjusted": leg.time_adjusted,
            "dep_hour_local": leg.dep_localtime.hour + leg.dep_localtime.minute / 60.0,
            "arr_hour_local": leg.arr_localtime.hour + leg.arr_localtime.minute / 60.0,
            "duration_minutes": leg.duration_minutes,
        }
        for k, v in leg.tags.items():
            if k not in row:
                row[k] = v
        data.append(row)
    return pd.DataFrame(data)


def demands_to_dataframe(demands: list[Demand]) -> pd.DataFrame:
    """Convert a list of Demand objects into a pandas DataFrame."""
    data = []
    for demand in demands:
        data.append(
            {
                "orig": demand.orig,
                "dest": demand.dest,
                "segment": demand.segment,
                "base_demand": demand.base_demand,
                "reference_price": demand.reference_price,
                "distance": demand.distance,
                "choice_model": demand.choice_model,
                "dwm_tolerance": demand.dwm_tolerance,
                "todd_curve": demand.todd_curve,
                "curve": demand.curve,
                "group_sizes": demand.group_sizes,
                "prob_saturday_night": demand.prob_saturday_night,
                "prob_num_days": demand.prob_num_days,
                "deterministic": demand.deterministic,
            }
        )
    return pd.DataFrame(data)


def demands_from_dataframe(df: pd.DataFrame, to_yaml: str | None = None) -> list[Demand] | None:
    """Convert a pandas DataFrame back into a list of Demand objects."""
    demands = []
    for _, row in df.iterrows():
        demand = Demand(
            orig=row["orig"],
            dest=row["dest"],
            segment=row["segment"],
            base_demand=row["base_demand"],
            reference_price=row.get("reference_price", row.get("reference_fare")),
            distance=row.get("distance"),
            choice_model=row.get("choice_model"),
            dwm_tolerance=row.get("dwm_tolerance"),
            todd_curve=row.get("todd_curve"),
            curve=row.get("curve"),
            group_sizes=row.get("group_sizes"),
            prob_saturday_night=row.get("prob_saturday_night"),
            prob_num_days=row.get("prob_num_days"),
            deterministic=row.get("deterministic", False),
        )
        demands.append(demand)
    if to_yaml:
        content = {"demands": [d.model_dump(exclude_defaults=True) for d in demands]}
        with open(to_yaml, "w") as f:
            yaml.safe_dump(content, f)
        return None
    else:
        return demands


def fares_to_dataframe(fares: list[Fare]) -> pd.DataFrame:
    """Convert a list of fares into a pandas DataFrame."""
    data = []
    for fare in fares:
        data.append(
            {
                "carrier": fare.carrier,
                "orig": fare.orig,
                "dest": fare.dest,
                "booking_class": fare.booking_class,
                "price": fare.price,
                "advance_purchase": fare.advance_purchase,
                "restrictions": "|".join(fare.restrictions),
                "category": fare.category,
                "cabin": fare.cabin,
                "min_stay": fare.min_stay,
                "saturday_night_required": fare.saturday_night_required,
            }
        )
    return pd.DataFrame(data)


def booking_curves_to_dataframe(booking_curves: dict[str, BookingCurve], add_zero_days: bool = True) -> pd.DataFrame:
    """Convert booking curves from the config into a pandas DataFrame."""
    data = []
    curve_names = set()
    for curve_name, curve_def in booking_curves.items():
        for days_prior, proportion in curve_def.curve.items():
            data.append(
                {
                    "curve_name": curve_name,
                    "days_prior": days_prior,
                    "proportion": proportion,
                }
            )
            curve_names.add(curve_name)
    if add_zero_days:
        for curve_name in curve_names:
            data.append(
                {
                    "curve_name": curve_name,
                    "days_prior": 0,
                    "proportion": 1.0,
                }
            )
    return pd.DataFrame(data).drop_duplicates().sort_values(by=["curve_name", "days_prior"], ascending=[True, False])


def frat5_curves_to_dataframe(frat5_curves: dict[str, Frat5Curve]) -> pd.DataFrame:
    """Convert Frat5 curves from the config into a pandas DataFrame.

    This only uses the `curve` values, the max_cap is ignored.
    """
    data = []
    curve_names = set()
    for curve_name, curve_def in frat5_curves.items():
        for days_prior, f5value in curve_def.curve.items():
            data.append(
                {
                    "curve_name": curve_name,
                    "days_prior": days_prior,
                    "frat5_value": f5value,
                }
            )
            curve_names.add(curve_name)
    return pd.DataFrame(data).drop_duplicates().sort_values(by=["curve_name", "days_prior"], ascending=[True, False])


def places_to_dataframe(places: list[Place]) -> pd.DataFrame:
    """Convert a list of Place objects into a pandas DataFrame."""
    data = []
    for place in places.values():
        row = {
            "name": place.name,
            "label": place.label,
            "country": place.country,
            "state": place.state,
            "lat": place.lat,
            "lon": place.lon,
            "time_zone": place.time_zone,
        }
        data.append(row)
    return pd.DataFrame(data)


class _DataFrameAccessor:
    """Base class for DataFrame accessors for config objects."""

    def __init__(self):
        self._obj = None

    def __get__(self, instance, owner):
        """Called when the attribute is accessed."""
        # instance is the object (e.g., 't' in the example below)
        # owner is the owner class (e.g., 'Temperature')
        if instance is None:
            return self
        self._obj = instance
        return self

    @property
    def legs(self) -> pd.DataFrame:
        """DataFrame representation of the legs in the config."""
        return legs_to_dataframe(self._obj.legs)

    @property
    def demands(self) -> pd.DataFrame:
        """DataFrame representation of the demands in the config."""
        return demands_to_dataframe(self._obj.demands)

    @property
    def fares(self) -> pd.DataFrame:
        """DataFrame representation of the fares in the config."""
        return fares_to_dataframe(self._obj.fares)

    @property
    def booking_curves(self) -> pd.DataFrame:
        """DataFrame representation of the booking curves in the config."""
        return booking_curves_to_dataframe(self._obj.booking_curves)

    @property
    def places(self) -> pd.DataFrame:
        """DataFrame representation of the places in the config."""
        return places_to_dataframe(self._obj.places)
