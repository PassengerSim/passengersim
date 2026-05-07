from difflib import get_close_matches


def did_you_mean(original_error, bad_key, possible_keys):
    close_matches = get_close_matches(bad_key, possible_keys, n=1)
    if close_matches:
        return f"{original_error}, did you mean '{close_matches[0]}'?"
    else:
        return original_error
