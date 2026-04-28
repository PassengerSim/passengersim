import os
import pickle
import shelve
from functools import wraps

import platformdirs
from geopy.geocoders import Nominatim


def disk_cache(cache_dir=None, retry_on_none=True):
    """
    A decorator to cache function results to disk using shelve.
    """
    if cache_dir is None:
        cache_dir = platformdirs.user_cache_dir("passengersim")
    if not os.path.exists(cache_dir):
        os.makedirs(cache_dir)

    def decorator(func):
        cache_file = os.path.join(cache_dir, f"{func.__name__}_cache.shelve")

        @wraps(func)
        def wrapper(*args, **kwargs):
            # Create a unique key for the cache entry based on function name and arguments
            # Ensure args and kwargs are hashable; convert to tuple for dictionary key
            cache_key = pickle.dumps((args, sorted(kwargs.items()))).decode("latin1")

            with shelve.open(cache_file) as cache:
                if cache_key in cache:
                    out = cache[cache_key]
                    if out is None and retry_on_none:
                        print(f"Cache hit for {func.__name__}, result is None, retrying...")
                        result = func(*args, **kwargs)
                        cache[cache_key] = result
                        return result
                    return out
                else:
                    print(f"Executing {func.__name__} and caching result...")
                    result = func(*args, **kwargs)
                    cache[cache_key] = result
                    return result

        return wrapper

    return decorator


@disk_cache()
def reverse_geocode(latitude, longitude):
    geolocator = Nominatim(user_agent="my_geocoder_app")  # Replace with a unique user agent
    try:
        return geolocator.reverse(f"{latitude}, {longitude}", exactly_one=True)
    except Exception as e:
        print(f"Error during reverse geocoding: {e}")
        return None


def get_country_code(latitude, longitude):
    """
    Converts latitude and longitude to a country code using Nominatim (OpenStreetMap).
    """
    try:
        location = reverse_geocode(latitude, longitude)
        if location and "address" in location.raw and "country_code" in location.raw["address"]:
            return location.raw["address"]["country_code"].upper()  # Return as uppercase ISO 3166-1 alpha-2
        else:
            return None
    except Exception as e:
        print(f"Error during reverse geocoding: {e}")
        return None


def get_state(latitude, longitude):
    """
    Converts latitude and longitude to a country code using Nominatim (OpenStreetMap).
    """
    try:
        location = reverse_geocode(latitude, longitude)
        if location and "address" in location.raw and "state" in location.raw["address"]:
            return location.raw["address"]["state"].upper()  # Return as uppercase
        else:
            return None
    except Exception as e:
        print(f"Error during reverse geocoding: {e}")
        return None


def is_conus(latitude, longitude):
    if latitude < 24.39 or latitude > 49.38:
        return False
    if longitude < -125 or longitude > -66.7:
        return False
    country_code = get_country_code(latitude, longitude)
    if country_code != "US":
        return False
    state = get_state(latitude, longitude)
    if state in ["ALASKA", "HAWAII"]:
        return False
    else:
        return True


# # Example usage:
# lat = 40.7128  # New York City latitude
# lon = -74.0060 # New York City longitude
# country_code = get_country_code(lat, lon)
#
# if country_code:
#     print(f"The country code for ({lat}, {lon}) is: {country_code}")
# else:
#     print(f"Could not determine country code for ({lat}, {lon}).")
#
# lat_paris = 48.8566 # Paris latitude
# lon_paris = 2.3522  # Paris longitude
# country_code_paris = get_country_code(lat_paris, lon_paris)
#
# if country_code_paris:
#     print(f"The country code for ({lat_paris}, {lon_paris}) is: {country_code_paris}")
# else:
#     print(f"Could not determine country code for ({lat_paris}, {lon_paris}).")
#
