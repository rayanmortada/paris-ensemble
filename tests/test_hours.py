from datetime import date

from src.data.hours import day_intervals, earliest_start, fits, is_valid

FRI, MON, TUE = date(2026, 12, 4), date(2026, 12, 7), date(2026, 12, 8)
LOUVRE = "Mo,Th,Sa-Su 09:00-18:00; Tu off; We,Fr 09:00-21:00; Jan 01,May 01,Dec 25 off"


def test_weekdays_and_late_opening():
    assert day_intervals(LOUVRE, FRI) == ((540, 1260),)         # late on Fridays
    assert day_intervals(LOUVRE, MON) == ((540, 1080),)
    assert day_intervals(LOUVRE, TUE) == ()                     # closed on Tuesdays
    assert day_intervals(LOUVRE, date(2026, 12, 25)) == ()      # closed on Christmas day


def test_seasons():
    spec = "Apr-Sep 07:30-21:00; Oct-Mar 08:00-17:30"
    assert day_intervals(spec, date(2026, 7, 1)) == ((450, 1260),)
    assert day_intervals(spec, FRI) == ((480, 1050),)


def test_split_service_and_earliest_start():
    spec = "Mo-Sa 12:00-14:30,19:00-22:30"
    assert day_intervals(spec, FRI) == ((720, 870), (1140, 1350))
    assert earliest_start(spec, FRI, 700, 75) == 720            # waits for opening
    assert earliest_start(spec, FRI, 840, 75) == 1140           # too late for lunch, so the evening service
    assert earliest_start(spec, FRI, 1300, 90) is None          # no room left today
    assert fits(spec, FRI, 720, 795) and not fits(spec, FRI, 840, 900)


def test_after_midnight_is_clipped_to_the_day():
    assert day_intervals("Mo-Su 18:00-02:00", FRI) == ((0, 120), (1080, 1440))


def test_open_all_the_time():
    assert day_intervals("24/7", FRI) == ((0, 1440),)


def test_by_appointment_does_not_count_as_open():
    assert day_intervals('Mo-Fr 10:00-18:00 "sur rendez-vous"', FRI) == ()


def test_validity():
    assert is_valid("Mo-Su 10:00-18:00")
    assert not is_valid("by appointment")
    assert not is_valid("off")
    assert not is_valid(None)
