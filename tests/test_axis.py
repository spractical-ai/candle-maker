import numpy as np
import pytest

from candle_maker import render, render_into
from candle_maker._native import render as native_render
from candle_maker._native import render_into as native_render_into


DATA = np.array([[2, 5, 1, 4], [4, 6, 2, 3]], dtype=np.float64)
GRAY = np.array([200, 200, 200], dtype=np.uint8)


@pytest.mark.parametrize("batch", [1, 17])
@pytest.mark.parametrize("size", [(64, 112), (128, 128), (256, 512)])
def test_axes_preserve_the_entire_trajectory_in_the_plot(batch, size):
    height, width = size
    data = np.tile(DATA, (batch, 3, 1))
    before = data.copy()
    images = render(data, height, width, use_axis=True)
    assert images.shape == (batch, height, width, 3)
    assert images.dtype == np.uint8 and images.flags.c_contiguous
    np.testing.assert_array_equal(images[:, 8:height - 22, 64:width - 8],
                                  render(data, height - 30, width - 72))
    np.testing.assert_array_equal(data, before)
    gray = np.all(images == GRAY, axis=-1)
    assert gray[:, 8:height - 22, :57].any()  # Price labels, outside plot.
    assert gray[:, height - 15:, 64:].any()  # Candle index labels.
    assert gray[:, 8:height - 21, 63].all()  # Left spine.
    assert gray[:, height - 22, 63:width - 8].all()  # Bottom spine.
    assert {tuple(pixel) for pixel in images.reshape(-1, 3)} == {
        (0, 0, 0), (0, 255, 0), (255, 0, 0), (200, 200, 200),
    }


def test_x_labels_are_zero_based_candle_indices():
    image = render(DATA, 128, 128, use_axis=True)[0]
    # The first label is the fixed 5x7 glyph for zero at candle 0's center.
    expected = np.array([[bool(row & (1 << (4 - col))) for col in range(5)]
                         for row in [14, 17, 19, 21, 25, 17, 14]])
    np.testing.assert_array_equal(np.all(image[113:120, 76:81] == GRAY, axis=-1), expected)


def test_y_labels_show_the_actual_high_and_low_prices():
    image = render(DATA, 128, 128, use_axis=True)[0]
    for y, rows in [(9, [14, 16, 16, 30, 17, 17, 14]),  # High = 6.
                    (98, [4, 12, 4, 4, 4, 4, 14])]:  # Low = 1.
        expected = np.array([[bool(row & (1 << (4 - col))) for col in range(5)] for row in rows])
        np.testing.assert_array_equal(np.all(image[y:y + 7, 52:57] == GRAY, axis=-1), expected)


def test_reuse_can_toggle_axes_without_leaving_old_labels_or_wrong_geometry():
    data = np.tile(DATA, (17, 5, 1))
    out = np.full((17, 128, 128, 3), 123, dtype=np.uint8)
    pointer = out.ctypes.data
    for use_axis in [True, False, True, False]:
        assert render_into(data, out, use_axis=use_axis) is out
        np.testing.assert_array_equal(out, render(data, 128, 128, use_axis=use_axis))
        assert out.ctypes.data == pointer
    assert not np.any(np.all(out == GRAY, axis=-1))


@pytest.mark.parametrize("use_axis", [True, False, np.bool_(True), np.bool_(False)])
def test_boolean_flags_and_native_defaults(use_axis):
    data = DATA[None]
    expected = render(data, 128, 128, use_axis=use_axis)
    np.testing.assert_array_equal(expected, native_render(data, 128, 128, bool(use_axis)))
    if not use_axis:
        np.testing.assert_array_equal(expected, render(data, 128, 128))
        np.testing.assert_array_equal(expected, native_render(data, 128, 128))


@pytest.mark.parametrize("flag", [1, 0, "yes", None, [], object()])
def test_invalid_flags_leave_reused_output_unchanged(flag):
    out = np.full((1, 128, 128, 3), 123, dtype=np.uint8)
    with pytest.raises(TypeError, match="use_axis"):
        render(DATA, 128, 128, use_axis=flag)
    with pytest.raises(TypeError, match="use_axis"):
        render_into(DATA, out, use_axis=flag)
    assert np.all(out == 123)


@pytest.mark.parametrize("size", [(63, 128), (128, 111), (1, 1)])
def test_axis_minimum_size_is_checked_before_mutating_output(size):
    out = np.full((1, *size, 3), 123, dtype=np.uint8)
    for function in [lambda: render(DATA, *size, use_axis=True),
                     lambda: render_into(DATA, out, use_axis=True),
                     lambda: native_render(DATA[None], *size, True),
                     lambda: native_render_into(DATA[None], out, True)]:
        with pytest.raises(ValueError, match="height >= 64 and width >= 112"):
            function()
    assert np.all(out == 123)


@pytest.mark.parametrize("price", [0.0, -5.0, np.finfo(float).max,
                                   -np.finfo(float).max, np.nextafter(0., 1.),
                                   np.nextafter(0., -1.)])
def test_flat_and_extreme_prices_have_safe_visible_labels(price):
    data = np.full((1, 1, 4), price)
    images = render(data, 64, 112, use_axis=True)
    out = np.empty_like(images)
    np.testing.assert_array_equal(render_into(data, out, use_axis=True), images)
    assert np.all(images[0, :, :57] == GRAY, axis=-1).any()
    assert images[0, 8:42, 64:104, 1].any()


def test_extreme_price_range_and_long_sequence():
    limit = np.finfo(float).max
    for data in [np.array([[0, limit, -limit, 1.]]), np.tile(DATA, (2050, 1))]:
        result = render(data, 128, 128, use_axis=True)
        out = np.empty_like(result)
        np.testing.assert_array_equal(render_into(data, out, use_axis=True), result)


def test_empty_batch_with_axes():
    data = np.empty((0, 2, 4))
    result = render(data, 64, 112, use_axis=True)
    assert result.shape == (0, 64, 112, 3)
    assert render_into(data, result, use_axis=True) is result


def test_invalid_prices_with_axes_do_not_modify_output():
    out = np.full((1, 128, 128, 3), 123, dtype=np.uint8)
    with pytest.raises(ValueError):
        render_into([[2, 1, 0, 2]], out, use_axis=True)
    assert np.all(out == 123)


def test_axes_are_deterministic_across_threads_and_cache_changes():
    from concurrent.futures import ThreadPoolExecutor

    data = np.tile(DATA, (17, 2, 1))
    settings = [(128, width, axis) for width in [128, 256, 512] for axis in [True, False]] * 2
    expected = [render(data, h, w, use_axis=axis) for h, w, axis in settings]
    with ThreadPoolExecutor(max_workers=4) as pool:
        actual = list(pool.map(lambda setting: render(data, setting[0], setting[1], use_axis=setting[2]), settings))
    for image, reference in zip(actual, expected):
        np.testing.assert_array_equal(image, reference)
