from dataclasses import dataclass


@dataclass(frozen=True)
class SpecialEvent:
    name: str
    month: int
    day_start: int
    day_end: int
    mult: float


SPECIAL_EVENTS: list[SpecialEvent] = [
    SpecialEvent("Hot Sale", 5, 27, 31, 2.8),
    SpecialEvent("Buen Fin", 11, 15, 18, 3.5),
    SpecialEvent("Black Fri", 11, 28, 30, 3.0),
    SpecialEvent("Navidad", 12, 20, 31, 2.2),
    SpecialEvent("Cyber", 12, 1, 3, 2.4),
    SpecialEvent("Reyes", 1, 4, 6, 1.6),
    SpecialEvent("Día Madres", 5, 9, 10, 2.0),
    SpecialEvent("San Valentín", 2, 13, 14, 1.5),
]


def get_event(month: int, day: int) -> SpecialEvent | None:
    for ev in SPECIAL_EVENTS:
        if ev.month == month and ev.day_start <= day <= ev.day_end:
            return ev
    return None
