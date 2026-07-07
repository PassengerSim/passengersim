from pydantic import ValidationError


def format_validation_errors(e: ValidationError):
    errors = e.errors()
    header = f"{len(errors)} validation error{'s' if len(errors) > 1 else ''} for Model"
    formatted_errs = []
    for e in errors:
        loc_str = ".".join(map(str, e["loc"]))
        formatted_errs.append(f"  ({loc_str})({e['loc']}) {e['msg']} [type={e['type']}]")
    print("\n".join([header] + formatted_errs))
    raise RuntimeError("Validation failed, see errors above.") from None
