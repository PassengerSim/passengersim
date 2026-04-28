from __future__ import annotations

from collections.abc import Callable, Sequence

from passengersim import core
from passengersim.config.choice_model import ChoiceModel as ChoiceModelConfig
from passengersim.config.demands import Demand as DemandConfig
from passengersim.config.legs import Leg as LegConfig
from passengersim.config.places import get_mileage
from passengersim.utils.string_counting import StringTracker


def make_core_choice_model(
    cm_cfg: ChoiceModelConfig,
    prng: core.Generator,
    fare_restriction_name_to_num: StringTracker | None = None,
    todd_curves: dict[str, core.DecisionWindow] | None = None,
) -> core.ChoiceModel:
    """Convert a choice model configuration into a core choice model."""
    if fare_restriction_name_to_num is None:
        fare_restriction_name_to_num = StringTracker(start_from=1)
    if todd_curves is None:
        todd_curves = {}

    x = core.ChoiceModel(cm_cfg.name, cm_cfg.kind, random_generator=prng)
    for pname, pvalue in cm_cfg:
        if pname in ("kind", "name") or pvalue is None:
            continue
        if pname == "todd_curve":
            tmp_dwm = todd_curves[pvalue]
            x.add_dwm(tmp_dwm)
        elif pname == "early_dep" and pvalue is not None:
            x.early_dep_offset = pvalue["offset"]
            x.early_dep_slope = pvalue["slope"]
            x.early_dep_beta = pvalue["beta"]
        elif pname == "late_arr" and pvalue is not None:
            x.late_arr_offset = pvalue["offset"]
            x.late_arr_slope = pvalue["slope"]
            x.late_arr_beta = pvalue["beta"]
        elif pname == "replanning" and pvalue is not None:
            x.replanning_alpha = pvalue[0]
            x.replanning_beta = pvalue[1]
        elif pname == "restrictions":
            for rname, rvalue in pvalue.items():
                restriction_num = fare_restriction_name_to_num.get_number(rname)
                if isinstance(rvalue, list | tuple):
                    x.add_restriction(restriction_num, *rvalue)
                else:
                    x.add_restriction(restriction_num, rvalue)
        elif isinstance(pvalue, list | tuple):
            x.add_parm(pname, *pvalue)
        else:
            x.add_parm(pname, pvalue)
    return x


def make_core_leg(
    leg_config: LegConfig,
    *,
    carriers: dict[str, core.Carrier] | None = None,
    next_leg_id: int | None = None,
    leg_id_exists: Callable[[int], bool] | None = None,
    max_leg_id: int = 1_000_000,
    places: dict[str, core.Airport] | None = None,
    booking_classes: Sequence[str | tuple[str, str]] = (),
) -> core.Leg | tuple[core.Leg, int]:
    """
    Convert a Leg configuration into a core Leg object.

    This also creates cabins and buckets as appropriate.

    Parameters
    ----------
    leg_config : passengersim.config.legs.Leg
        The leg configuration to convert into a core Leg object.
    carriers : dict[str, core.Carrier]
        Mapping of carrier name to carrier object. The core Leg constructor requires
        linking to the core Carrier object, not just a name of a carrier.
    next_leg_id : int
        A hint for a possible next leg id, if the config does not provide one.
    leg_id_exists : Callable[[int], bool]
        A function that can check whether a given leg_id already exists in the engine,
        used to ensure that we don't create duplicate leg ids.
    max_leg_id : int, defaults to 1_000_000
        A safety cap that should not be needed, will prevent an infinite loop if
        leg_id_exists is not well-behaved.
    places : dict[str, core.Airport]
        Mapping of place name to core Airport object, used to calculate distance
        if not provided in the config. If you provide a distance explicitly the
        places are not needed. If you provide neither an explicit distance nor
        the places, the distance will be set to zero.
    booking_classes : Sequence[str]
        A list of booking class codes to create buckets for. If the leg has a
        single cabin, the booking classes are all assigned to that cabin. If the
        leg has multiple cabins, the booking classes can be provided as a tuple
        of (bkg_class, cabin_code).

    Returns
    -------
    passengersim.core.Leg
    """
    if carriers is None:
        carriers = {}
    if places is None:
        places = {}

    # if no leg_id is provided, we'll try to use the fltno
    if (
        leg_config.leg_id is None
        and leg_config.fltno is not None
        and leg_id_exists is not None
        and not leg_id_exists(leg_config.fltno)
    ):
        leg_config.leg_id = leg_config.fltno

    if next_leg_id is None:
        if leg_config.leg_id is not None:
            next_leg_id = leg_config.leg_id + 1
        elif leg_config.fltno is not None:
            next_leg_id = leg_config.fltno + 1
        else:
            next_leg_id = 1

    # if the proposed leg_id already exists, we'll ignore it find a new one
    if leg_config.leg_id is not None and leg_id_exists is not None and leg_id_exists(leg_config.leg_id):
        leg_config.leg_id = None

    # if we have no leg_id (or none we can use), we'll use the next available
    if leg_config.leg_id is None and next_leg_id is not None:
        while leg_id_exists(next_leg_id):
            next_leg_id += 1
            if next_leg_id > max_leg_id:
                raise ValueError(f"cannot find a suitable leg_id below the cap {max_leg_id}")
        leg_config.leg_id = next_leg_id

    # create the leg object
    leg = core.Leg(
        leg_config.leg_id or -1,
        carriers.get(leg_config.carrier),
        leg_config.fltno,
        orig=leg_config.orig,
        dest=leg_config.dest,
        dep_time=leg_config.dep_time,
        arr_time=leg_config.arr_time,
        dep_time_offset=leg_config.dep_time_offset,
        arr_time_offset=leg_config.arr_time_offset,
        tags=leg_config.tags,
    )
    if leg_config.distance:
        leg.distance = leg_config.distance
    else:
        leg.distance = get_mileage(places, leg.orig, leg.dest)

    # Now we also create the cabins
    if isinstance(leg_config.capacity, int):
        cap = leg_config.capacity
        leg.capacity = cap
        cabin = core.Cabin("Y", cap)
        leg.add_cabin(cabin)
    else:
        tot_cap = 0
        for cabin_code, tmp_cap in leg_config.capacity.items():
            cap = int(tmp_cap)
            tot_cap += cap
            cabin = core.Cabin(cabin_code, cap)
            leg.add_cabin(cabin)
        leg.capacity = tot_cap

    # Now the buckets
    cabin_code_list = [c.name for c in leg.cabins]
    if len(booking_classes) > 0:
        cap = float(leg.capacity)
        try:
            history_def = leg.carrier.get_history_def()
        except AttributeError:
            history_def = None
        for bkg_class in booking_classes:
            # maybe TODO: any initial RM that might be set on buckets should be applied here
            auth = int(cap)
            if isinstance(bkg_class, tuple):
                # We are likely using multi-cabin, so unpack it
                (bkg_class, cabin_code) = bkg_class
            else:
                cabin_code = bkg_class[0]
            if cabin_code not in cabin_code_list:
                continue
            b = core.Bucket(bkg_class, alloc=auth, history=history_def)
            b.cabin = cabin_code
            leg.add_bucket(b)

    return leg


def make_core_demand(
    dmd_config: DemandConfig,
    *,
    markets: dict[str, core.Market],
    market_multipliers: dict[str, float] | None = None,
    demand_multiplier: float = 1.0,
    airports: dict[str, core.Airport] | None = None,
    choice_models: dict[str, core.ChoiceModel] | None = None,
    booking_curves: dict[str, core.BookingCurve] | None = None,
    todd_curves: dict[str, core.Todd] | None = None,
    carrier_preference_probs: dict[str, float] | None = None,
    dwm_tolerance: list[dict] | None = None,
) -> core.Demand:
    # create the market if needed
    mkt_ident = f"{dmd_config.orig}~{dmd_config.dest}"
    if mkt_ident not in markets:
        mkt = core.Market(dmd_config.orig, dmd_config.dest)
        markets[mkt_ident] = mkt
    else:
        mkt = markets[mkt_ident]

    if market_multipliers is None:
        market_multipliers = dict()

    dmd = core.Demand(
        segment=dmd_config.segment,
        market=mkt,
        deterministic=dmd_config.deterministic,
        base_demand=float(dmd_config.base_demand * demand_multiplier * market_multipliers.get(mkt_ident, 1.0)),
        price=dmd_config.reference_price,
        reference_price=dmd_config.reference_price,
    )
    if dmd_config.distance is not None and dmd_config.distance > 0.01:
        dmd.distance = dmd_config.distance
    elif airports is not None and dmd.orig in airports and dmd.dest in airports:
        dmd.distance = get_mileage(airports, dmd.orig, dmd.dest)

    # Get the choice model to use for this demand.
    # If it is not defined, that is an error
    model_name = dmd_config.choice_model or dmd_config.segment
    cm = choice_models.get(model_name, None)
    if cm is not None:
        dmd.choice_model = cm
    else:
        raise ValueError(f"Choice model {model_name} not found for demand {dmd}")

    # Attach a booking curve to this demand
    if dmd_config.curve:
        booking_curve_name = str(dmd_config.curve).strip()
        if booking_curve_name in booking_curves:
            curve = booking_curves[booking_curve_name]
            dmd.add_curve(curve)

    if todd_curves is not None:
        if dmd_config.todd_curve in todd_curves:
            dmd.add_dwm(todd_curves[dmd_config.todd_curve])

    if dmd_config.group_sizes is not None:
        dmd.add_group_sizes(dmd_config.group_sizes)

    dmd.prob_saturday_night = dmd_config.prob_saturday_night
    dmd.prob_num_days = dmd_config.prob_num_days
    if carrier_preference_probs is not None:
        dmd.prob_favored_carrier = carrier_preference_probs

    for o in dmd_config.overrides:
        dmd.add_override(o.carrier, o.discount_pct, o.pref_adj)

    if dmd_config.dwm_tolerance > 0.0:
        dmd.dwm_tolerance = dmd_config.dwm_tolerance
    elif dwm_tolerance is not None and len(dwm_tolerance) > 0:
        for tolerance in dwm_tolerance:
            if tolerance["min_dist"] <= dmd.distance <= tolerance["max_dist"]:
                if dmd.segment in tolerance:
                    dmd.dwm_tolerance = tolerance[dmd.segment]
                else:
                    raise Exception(f"DWM tolerance data is missing segment '{dmd.segment}'")

    return dmd
