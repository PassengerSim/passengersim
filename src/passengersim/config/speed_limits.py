from pydantic import BaseModel


class SpeedLimits(BaseModel):
    SHORT_HAUL_MAX_DISTANCE: float = 500
    """Short haul legs are all legs less than this distance (in miles)."""

    LONG_HAUL_MIN_DISTANCE: float = 1500
    """Long haul legs are all legs greater than this distance (in miles)."""

    # speed limits in miles per hour
    SHORT_HAUL_MINIMUM_SPEED: float = 50.0
    SHORT_HAUL_MAXIMUM_SPEED: float = 400.0

    LONG_HAUL_MINIMUM_SPEED: float = 300.0
    LONG_HAUL_MAXIMUM_SPEED: float = 700.0


def get_speed_limits(distance: float, limits: SpeedLimits | None = None) -> tuple[str, float, float]:
    """Get lower and upper bounds on reasonable leg speeds."""

    if limits is None:
        limits = SpeedLimits()

    if distance < limits.SHORT_HAUL_MAX_DISTANCE:
        return "SHORT", limits.SHORT_HAUL_MINIMUM_SPEED, limits.SHORT_HAUL_MAXIMUM_SPEED

    if distance > limits.LONG_HAUL_MIN_DISTANCE:
        return "LONG", limits.LONG_HAUL_MINIMUM_SPEED, limits.LONG_HAUL_MAXIMUM_SPEED

    # between the thresholds, interpolate
    position = (distance - limits.SHORT_HAUL_MAX_DISTANCE) / (
        limits.LONG_HAUL_MIN_DISTANCE - limits.SHORT_HAUL_MAX_DISTANCE
    )
    inv_position = 1.0 - position
    mid_haul_minimum_speed = inv_position * limits.SHORT_HAUL_MINIMUM_SPEED + position * limits.LONG_HAUL_MINIMUM_SPEED
    mid_haul_maximum_speed = inv_position * limits.SHORT_HAUL_MAXIMUM_SPEED + position * limits.LONG_HAUL_MAXIMUM_SPEED

    return "MID", mid_haul_minimum_speed, mid_haul_maximum_speed


# def clear_speed_limits():
#     global SHORT_HAUL_MINIMUM_SPEED, SHORT_HAUL_MAXIMUM_SPEED, LONG_HAUL_MINIMUM_SPEED, LONG_HAUL_MAXIMUM_SPEED
#     SHORT_HAUL_MINIMUM_SPEED = 0
#     SHORT_HAUL_MAXIMUM_SPEED = 9e9
#     LONG_HAUL_MINIMUM_SPEED = 0
#     LONG_HAUL_MAXIMUM_SPEED = 9e9
