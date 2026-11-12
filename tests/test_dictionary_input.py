from copy import deepcopy
from datetime import date, datetime, timedelta, timezone

import numpy as np
import pytest

from candle_maker import Candle, render, render_into
from candle_maker._native import render as native_render
from candle_maker._native import render_into as native_render_into


PRICES = np.array([[2., 5., 1., 4.], [4., 6., 2., 3.]])
KEYS = ("open", "high", "low", "close")


def records(times=None):
    rows = [dict(zip(KEYS, values)) for values in PRICES]
    if times is not None:
        for row, value in zip(rows, times):
            row["time"] = value
    return rows


@pytest.mark.parametrize("use_axis", [False, True])
@pytest.mark.parametrize("nested", [False, True])
def test_dictionary_prices_match_array_pixels_and_reuse(use_axis, nested):
    data = records()
    data[0]["volume"] = 123  # Extra fields are ignored.
    data[0] = dict(reversed(list(data[0].items())))  # Key order is irrelevant.
    if nested:
        data = [data, records()]
    before = deepcopy(data)
    expected = render(np.tile(PRICES, (2, 1, 1)) if nested else PRICES, 128, 512, use_axis=use_axis)
    np.testing.assert_array_equal(render(data, 128, 512, use_axis=use_axis), expected)
    out = np.full_like(expected, 123)
    assert render_into(data, out, use_axis=use_axis) is out
    np.testing.assert_array_equal(out, expected)
    assert data == before


def test_public_candle_typing_is_constructible():
    data = [Candle(open=1, high=3, low=0, close=2)]
    assert render(data, 8, 8).shape == (1, 8, 8, 3)


@pytest.mark.parametrize("dtype", [np.float16, np.float32, np.float64, np.int32, np.uint64])
def test_numpy_scalar_price_fields(dtype):
    data = [{key: dtype(value) for key, value in row.items()} for row in records()]
    np.testing.assert_array_equal(render(data, 64, 112, use_axis=True), render(PRICES, 64, 112, use_axis=True))


def test_datetime_labels_use_the_candle_dates_and_times():
    data = records([datetime(2024, 2, 29, 9, 30), datetime(2024, 3, 1, 13, 45)])
    result = render(data, 128, 512, use_axis=True)
    expected = native_render(PRICES[None], 128, 512, True, [[
        (0, "2024-02-29\n09:30"), (1, "2024-03-01\n13:45"),
    ]])
    np.testing.assert_array_equal(result, expected)
    # Colon in the first HH:MM label, rendered as a bitmap rather than omitted.
    colon = np.array([[bool(row & (1 << (4 - col))) for col in range(5)]
                      for row in [0, 12, 12, 0, 12, 12, 0]])
    np.testing.assert_array_equal(np.all(result[0, 113:120, 172:177] == 200, axis=-1), colon)
    np.testing.assert_array_equal(result[:, 8:96, 64:504], render(PRICES, 88, 440))


@pytest.mark.parametrize("size", [(1, 1), (21, 10), (128, 128), (256, 512)])
def test_times_do_not_change_axis_free_pixels(size):
    data = records([datetime(1, 1, 1), datetime(9999, 12, 31, 23, 59, 59, 999999)])
    np.testing.assert_array_equal(render(data, *size), render(PRICES, *size))


def test_changing_times_only_changes_the_bottom_labels():
    first = records([datetime(2024, 2, 29, 9, 30), datetime(2024, 3, 1, 13, 45)])
    second = records([datetime(2025, 2, 28, 9, 30), datetime(2025, 3, 1, 13, 45)])
    a, b = render(first, 128, 512, use_axis=True), render(second, 128, 512, use_axis=True)
    np.testing.assert_array_equal(a[:, :103], b[:, :103])
    assert not np.array_equal(a[:, 103:], b[:, 103:])


def test_timezone_aware_times_are_normalized_to_utc_without_mutating_input():
    times = [datetime(2024, 3, 1, 1, 30, tzinfo=timezone(timedelta(hours=3))),
             datetime(2024, 3, 1, 1, 30, tzinfo=timezone(timedelta(hours=-4)))]
    data = records(times)
    expected = records([value.astimezone(timezone.utc) for value in times])
    np.testing.assert_array_equal(render(data, 128, 512, use_axis=True),
                                  render(expected, 128, 512, use_axis=True))
    assert all(row["time"] is value for row, value in zip(data, times))


@pytest.mark.parametrize("times,labels", [
    ([datetime(2024, 1, 1, 0, 0, 1), datetime(2024, 1, 1, 0, 0, 2)],
     ["2024-01-01\n00:00:01", "2024-01-01\n00:00:02"]),
    ([datetime(2024, 1, 1, microsecond=1), datetime(2024, 1, 1, microsecond=123456)],
     ["2024-01-01\n00:00:00.000001", "2024-01-01\n00:00:00.123456"]),
    ([datetime(1, 1, 1), datetime(9999, 12, 31)],
     ["0001-01-01\n00:00", "9999-12-31\n00:00"]),
])
def test_seconds_microseconds_and_full_year_range(times, labels):
    expected = native_render(PRICES[None], 128, 512, True, [[(0, labels[0]), (1, labels[1])]])
    np.testing.assert_array_equal(render(records(times), 128, 512, use_axis=True), expected)


@pytest.mark.parametrize("size", [(64, 112), (128, 128), (256, 512)])
def test_batched_mixed_time_presence_matches_individual_renders_and_reuse(size):
    times = [datetime(2024, 2, 29, 9, 30), datetime(2024, 3, 1, 13, 45)]
    data = [records(times if i % 2 else None) for i in range(17)]
    result = render(data, *size, use_axis=True)
    for i, sample in enumerate(data):
        np.testing.assert_array_equal(result[i], render(sample, *size, use_axis=True)[0])
    out = np.full_like(result, 123)
    pointer = out.ctypes.data
    for batch, flag in [(data, True), ([records() for _ in data], True), (data, False), (data, True)]:
        assert render_into(batch, out, use_axis=flag) is out
        np.testing.assert_array_equal(out, render(batch, *size, use_axis=flag))
        assert out.ctypes.data == pointer


def test_input_order_is_preserved_even_when_times_are_irregular_or_reversed():
    times = [datetime(2024, 3, 1), datetime(2024, 1, 1)]
    image = render(records(times), 128, 512, use_axis=True)
    np.testing.assert_array_equal(image[:, 8:96, 64:504], render(PRICES, 88, 440))


@pytest.mark.parametrize("key", KEYS)
def test_required_keys_are_checked_before_mutating_output(key):
    data = records()
    del data[1][key]
    out = np.full((1, 128, 512, 3), 123, dtype=np.uint8)
    with pytest.raises(ValueError, match=f"missing required key '{key}'"):
        render_into(data, out)
    assert np.all(out == 123)


@pytest.mark.parametrize("value", [True, np.bool_(False), "1", None, 1j])
def test_non_numeric_record_prices_are_rejected(value):
    data = records()
    data[0]["open"] = value
    with pytest.raises(TypeError, match="real numbers"):
        render(data, 64, 112)


@pytest.mark.parametrize("value", [np.nan, np.inf, -np.inf])
def test_nonfinite_record_prices_leave_output_unchanged(value):
    data = records()
    data[0]["high"] = value
    out = np.full((1, 64, 112, 3), 123, dtype=np.uint8)
    with pytest.raises(ValueError, match="invalid OHLC"):
        render_into(data, out, use_axis=True)
    assert np.all(out == 123)


@pytest.mark.parametrize("value", [None, "2024-01-01", date(2024, 1, 1), np.datetime64("2024-01-01"), 123, True])
@pytest.mark.parametrize("use_axis", [False, True])
def test_time_must_be_a_datetime_even_when_axes_are_disabled(value, use_axis):
    data = records([value, datetime(2024, 1, 1)])
    out = np.full((1, 128, 128, 3), 123, dtype=np.uint8)
    with pytest.raises(TypeError, match="time must be a datetime"):
        render_into(data, out, use_axis=use_axis)
    assert np.all(out == 123)


@pytest.mark.parametrize("use_axis", [False, True])
def test_partially_missing_times_and_mixed_awareness_are_rejected(use_axis):
    partial = records()
    partial[0]["time"] = datetime(2024, 1, 1)
    mixed = records([datetime(2024, 1, 1), datetime(2024, 1, 2, tzinfo=timezone.utc)])
    out = np.full((1, 128, 128, 3), 123, dtype=np.uint8)
    for data, message in [(partial, "every candle"), (mixed, "mix naive")]:
        with pytest.raises(ValueError, match=message):
            render_into(data, out, use_axis=use_axis)
        assert np.all(out == 123)


def test_utc_conversion_overflow_is_reported_before_writing_output():
    value = datetime(1, 1, 1, tzinfo=timezone(timedelta(hours=1)))
    with pytest.raises(ValueError, match="represented in UTC"):
        render(records([value, value]), 128, 128, use_axis=True)


@pytest.mark.parametrize("data,error", [
    ([records(), records()[:1]], ValueError),
    ([records(), [1, 2]], TypeError),
    ([records()[0], 2], TypeError),
    ([{}], ValueError),
    ([], ValueError),
    ([[]], ValueError),
])
def test_invalid_record_shapes(data, error):
    with pytest.raises(error):
        render(data, 128, 128)


@pytest.mark.parametrize("labels", [[], [[]], [[(2, "2024-01-01\n00:00")]],
    [[(1, "2024-01-01\n00:00"), (0, "2024-01-01\n00:00")]],
    [[(0, "x\n00:00")]], [[(0, "2024-01-01")]],
    [[(0, "2024-01-01\n00:00\n00")]], [[(0, "1" * 16 + "\n00:00")]],
    [[(0, "\n00:00")]], [[(0, "2024-01-01\n🙂")]],
])
def test_native_time_label_validation_preserves_output(labels):
    out = np.full((1, 128, 128, 3), 123, dtype=np.uint8)
    with pytest.raises(ValueError):
        native_render_into(PRICES[None], out, True, labels)
    assert np.all(out == 123)


def test_native_time_labels_require_axes():
    with pytest.raises(ValueError, match="require use_axis"):
        native_render(PRICES[None], 128, 128, False, [[(0, "2024-01-01\n00:00")]])


def test_datetime_labels_are_safe_across_python_threads():
    from concurrent.futures import ThreadPoolExecutor

    times = [datetime(2024, 2, 29, microsecond=123456), datetime(2024, 3, 1, 13, 45)]
    data = [records(times if i % 2 else None) for i in range(17)]
    settings = [(128, width, use_axis) for width in [128, 256, 512] for use_axis in [True, False]] * 2
    expected = [render(data, h, w, use_axis=flag) for h, w, flag in settings]
    with ThreadPoolExecutor(max_workers=4) as pool:
        results = list(pool.map(lambda setting: render(data, setting[0], setting[1], use_axis=setting[2]), settings))
    for actual, reference in zip(results, expected):
        np.testing.assert_array_equal(actual, reference)
