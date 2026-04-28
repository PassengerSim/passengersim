# Forecast

A PassengerSim forecast is not just an attempt to predict the future -- it is a very
specific format for a set of predictions of some very specific things. The default
implementation in PassengerSim is a `ForecastGroup`, but it can be replaced by any
other code that exposes the same API. The forecast
is generated for one or more booking classes on a common simulation element.  This
simulation element can be a `Leg` representing a particular flight, or a `Path`
representing a sequence of one or more flights sold to customers as a non-stop
itinerary (for a single leg path) or a connecting itinerary (for multi-leg paths).

For each `Leg`, there is a collection of `Bucket` objects corresponding to various
booking classes.  Similarly, for each `Path` there is a collection of `PathClass`
objects corresponding to various booking classes.

Each `Leg` or `Path` can have one or more `ForecastGroup` objects associated with it, although
for most situations generally there will only be at most one `ForecastGroup` for each,
unless the user is intentionally studying, comparing, or mixing different Forecasts. Various
RM optimization algorithms will require either a leg-based or a path-based forecast
to operate, although in theory it is possible to work with both, or other variations.


## Inputs

A `ForecastGroup` is populated with some input data.  The key foundation of this is a
set of `History` objects. Each `Bucket` or `PathClass` has a `History` associated with
it, which is updated dynamically as the simulation runs.  Each `History` makes
available three arrays:

- *sold*, which contains the total number of bookings by sample day by timeframe,
- *sold_priceable*, which identifies how many of those bookings were made at moments
    when the booking class was the lowest price option available at that moment, and
- *closed_flags*, which indicates whether the booking class was not available for
    sale at the beginning of the timeframe, at the end of the timeframe, or both.

All of these arrays are sized similarly with rows for historical sample days and
columns for timeframes.

### Inputs By Booking Class

The input data for the forecast includes a reference to the `History` for each
booking class, as well as a selection of other relevant attributes of each
booking class:

- `identifier`, a unique string identifier, typically the name of the associated
    booking class.
- `advpurch_max_tf_index`, the maximum timeframe position where this booking class
    remains available for customers to purchase. After this timeframe, the booking
    class is closed to all further purchases by an advance purchase restriction.
    For booking classes with no advance purchase restriction, this should be set to
    a sufficiently large value (by default, 9999).
- `customer_price`, the price that a customer pays to book this booking class. This
    input may not be needed for all forecasting algorithms.

### Inputs in Aggregate

In addition to the mapping of data by booking class described above, there is also
some input data that is generic across all booking classes:

- `dcp_days_prior`, a strictly monotonically decreasing vector of `int`, giving the
    number of days prior to departure at each DCP. This vector should not include `0`
    as the final value, as there is no forecasting at the moment of departure.  Since
    the `0` is not included, the size of this vector will be exactly equal to the
    total number of timeframes. This vector can be used as the `index` for the timeframe
    dimension of the history input arrays, as well as the output vectors of the forecast.


## Processing

The processing of the Forecast can be more or less any algorithm that can transform the
inputs into the outputs.  This can include demand detruncation, aggregation of priceable
history, and whatever else is desired.

Individual `Forecast` methods may define additional parameters that serve
as inputs to the forecasting process. For example, the standard fully restricted forecast
includes an option to select between additive pickup and exponential smoothing.


## Outputs

The outputs of the Forecast

### Expected Demand

A primary output for a forecast is the expected demand.  A forecast should ultimately
provide a set of vectors for each booking class:

- `mean_to_departure`
- `mean_in_timeframe`
- `stdev_to_departure`
- `stdev_in_timeframe`

Each vector should be of length equal to the number of timeframes.  If a forecast
is generated part way through the booking process, this does not change the shape
of the vector; both past and future timeframes remain a part of the vector. It is
generally not necessary or useful to update the portion of the vector that pertains
the to the past, but the allocated size of the vector remains the same.

### Marginal Revenue

In addition to a forecast of demand, the forecast *may* also provide a measure of
marginal revenue per unit of forecasted demand in each booking class.  In simple standard
forecasts in fully restricted markets, this marginal revenue is equal simply to the
customer price. However, for more sophisticated forecasts that incorporate elasticity,
this marginal revenue may be different, as the forecast may include some probability that
some customers will buy-up to higher classes if their preferred class is not available.
The sellup rates will vary over timeframes, so the the `adjusted_marginal_revenue` output
is a vector for each booking class (with length equal to the number of timeframes), not
just a single value.
