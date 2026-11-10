import numpy as np
import pytest

from candle_maker import render, render_into
from candle_maker._native import render as native_render


@pytest.fixture
def trajectory():
    return np.array([[2, 5, 1, 4], [4, 6, 2, 3]], dtype=np.float64)


def test_batch_shape_colors_and_trajectory(trajectory):
    images = render(np.stack([trajectory, trajectory]), 40, 60)
    assert images.shape == (2, 40, 60, 3)
    assert images.dtype == np.uint8
    assert images.flags.c_contiguous
    colors = {tuple(pixel) for pixel in images.reshape(-1, 3)}
    assert colors == {(0, 0, 0), (0, 255, 0), (255, 0, 0)}
    assert np.any(images[0, :, :30, 1])  # First candle is green.
    assert not np.any(images[0, :, :30, 0])
    assert np.any(images[0, :, 30:, 0])  # Second candle is red.
    assert not np.any(images[0, :, 30:, 1])
    assert np.all(images[:, 0] == 0)
    assert np.all(images[:, -1] == 0)


def test_wicks_bodies_and_price_orientation():
    image = render([[2, 5, 1, 4]], 21, 10)[0]
    green = (0, 255, 0)
    np.testing.assert_array_equal(image[1, 5], green)  # High at top.
    np.testing.assert_array_equal(image[19, 5], green)  # Low at bottom.
    np.testing.assert_array_equal(image[10, 2], green)  # Filled body.
    np.testing.assert_array_equal(image[1, 2], (0, 0, 0))  # Thin wick.


def test_each_chart_scales_independently(trajectory):
    image = render(np.stack([trajectory, trajectory * 1000 + 100]), 51, 60)
    np.testing.assert_array_equal(image[0], image[1])


@pytest.mark.parametrize("dtype", [np.float32, np.float64, np.int32, np.uint64])
def test_numeric_inputs(trajectory, dtype):
    np.testing.assert_array_equal(render(trajectory.astype(dtype), 32, 32), render(trajectory, 32, 32))


def test_strided_readonly_inputs_are_not_modified(trajectory):
    data = np.tile(trajectory, (1, 3, 1))[:, ::2, :]
    before = data.copy()
    data.setflags(write=False)
    np.testing.assert_array_equal(render(data, 32, 32), render(before, 32, 32))
    np.testing.assert_array_equal(data, before)


@pytest.mark.parametrize("price", [0.0, 12.0, -5.0])
def test_flat_doji_is_visible_and_green(price):
    images = render(np.full((2, 4, 4), price), 21, 40)
    assert np.any(images[:, 10, :, 1] == 255)
    assert not images[:, :, :, 0].any()
    assert not images[:, :10].any()
    assert not images[:, 11:].any()


@pytest.mark.parametrize("size", [(1, 1), (1, 8), (8, 1), (2, 2)])
def test_tiny_images_and_dense_candles(trajectory, size):
    images = render(np.tile(trajectory, (8, 1)), *size)
    assert images.shape == (1, *size, 3)
    assert images.any()


def test_empty_batch():
    assert render(np.empty((0, 4, 4)), 32, 64).shape == (0, 32, 64, 3)


@pytest.mark.parametrize("data", [np.zeros(4), np.zeros((3, 5)), np.zeros((2, 3, 5)), np.zeros((1, 2, 3, 4)), np.empty((2, 0, 4))])
def test_invalid_shapes(data):
    with pytest.raises(ValueError):
        render(data, 32, 32)


@pytest.mark.parametrize("candle", [[2, 1, 0, 1], [1, 3, 2, 3], [1, 0, 3, 2], [1, np.nan, 0, 2], [1, np.inf, 0, 2], [1, 2, -np.inf, 2]])
def test_invalid_prices(candle):
    with pytest.raises(ValueError, match="sample 0, candle 0"):
        render([candle], 32, 32)


@pytest.mark.parametrize("data", [[["1", "2", "0", "1"]], [[1j] * 4], [[True] * 4], [[None] * 4]])
def test_non_numeric_data(data):
    with pytest.raises(TypeError):
        render(data, 32, 32)


@pytest.mark.parametrize("size,error", [((0, 32), ValueError), ((32, -1), ValueError), ((1.5, 32), TypeError), ((True, 32), TypeError), ((np.bool_(True), 32), TypeError)])
def test_invalid_dimensions(trajectory, size, error):
    with pytest.raises(error):
        render(trajectory, *size)


def test_dimension_overflow(trajectory):
    with pytest.raises(ValueError, match="too large"):
        render(trajectory, np.iinfo(np.intp).max, 2)


def test_native_interface_validates_shape():
    with pytest.raises(ValueError, match="shape"):
        native_render(np.zeros((1, 2, 5)), 32, 32)


def test_extreme_finite_prices():
    limit = np.finfo(np.float64).max
    images = render([[0, limit, -limit, 1]], 32, 32)
    assert images.any()


def test_subnormal_prices():
    small = np.nextafter(0.0, 1.0)
    images = render([[small, 4 * small, 0, 3 * small]], 32, 32)
    assert images.any()


def test_deterministic_batch_equals_individual_calls(trajectory):
    batch = np.stack([trajectory, trajectory[::-1], trajectory * 2])
    result = render(batch, 64, 96)
    for i, sample in enumerate(batch):
        np.testing.assert_array_equal(result[i], render(sample, 64, 96)[0])


@pytest.mark.parametrize("batch", [1, 8, 17, 32])
@pytest.mark.parametrize("size", [(1, 1), (40, 60), (128, 128), (256, 512)])
def test_reuse_clears_previous_pixels_and_matches_render(trajectory, batch, size):
    data = np.tile(trajectory, (batch, 1, 1))
    out = np.full((batch, *size, 3), 123, dtype=np.uint8)
    pointer = out.ctypes.data
    assert render_into(data, out) is out
    np.testing.assert_array_equal(out, render(data, *size))
    changed = np.full_like(data, 10.0)
    render_into(changed, out)
    np.testing.assert_array_equal(out, render(changed, *size))
    assert out.ctypes.data == pointer


def test_reuse_accepts_single_sequence(trajectory):
    out = np.empty((1, 32, 64, 3), dtype=np.uint8)
    np.testing.assert_array_equal(render_into(trajectory, out), render(trajectory, 32, 64))


@pytest.mark.parametrize("data", [[[1, np.nan, 0, 2]], [[2, 1, 0, 2]]])
def test_reuse_invalid_prices_leave_output_unchanged(data):
    out = np.full((1, 32, 64, 3), 123, dtype=np.uint8)
    with pytest.raises(ValueError):
        render_into(data, out)
    assert np.all(out == 123)


@pytest.mark.parametrize("out,error", [
    (np.zeros((1, 32, 64, 3), dtype=np.float32), TypeError),
    ([], TypeError),
    (np.zeros((32, 64, 3), dtype=np.uint8), ValueError),
    (np.zeros((1, 32, 64, 4), dtype=np.uint8), ValueError),
    (np.zeros((2, 32, 64, 3), dtype=np.uint8), ValueError),
    (np.zeros((1, 0, 64, 3), dtype=np.uint8), ValueError),
    (np.zeros((1, 32, 64, 3), dtype=np.uint8)[:, :, ::2], ValueError),
    (np.zeros((1, 32, 64, 3), dtype=np.uint8, order="F"), ValueError),
])
def test_reuse_rejects_invalid_output(trajectory, out, error):
    with pytest.raises(error):
        render_into(trajectory, out)


def test_reuse_rejects_readonly_output(trajectory):
    out = np.zeros((1, 32, 64, 3), dtype=np.uint8)
    out.setflags(write=False)
    with pytest.raises(ValueError, match="writable"):
        render_into(trajectory, out)


def test_reuse_rejects_overlapping_input_and_output():
    out = np.zeros((1, 32, 64, 3), dtype=np.uint8)
    data = out.reshape(-1)[:64].view(np.float64).reshape(1, 2, 4)
    with pytest.raises(ValueError, match="overlap"):
        render_into(data, out)


def test_reuse_empty_batch():
    out = np.empty((0, 32, 64, 3), dtype=np.uint8)
    assert render_into(np.empty((0, 2, 4)), out) is out


def test_retained_render_results_are_independent(trajectory):
    first = render(trajectory, 64, 64)
    saved = first.copy()
    second = render(np.full_like(trajectory, 10), 64, 64)
    assert not np.shares_memory(first, second)
    np.testing.assert_array_equal(first, saved)


def test_unaligned_inputs(trajectory):
    storage = np.zeros(trajectory.nbytes + 1, dtype=np.uint8)
    unaligned = np.ndarray(trajectory.shape, dtype=np.float64, buffer=storage, offset=1)
    unaligned[:] = trajectory
    assert not unaligned.flags.aligned
    expected = render(trajectory, 32, 64)
    np.testing.assert_array_equal(render(unaligned, 32, 64), expected)
    out = np.empty_like(expected)
    np.testing.assert_array_equal(render_into(unaligned, out), expected)


def test_long_sequences_beyond_geometry_cache():
    data = np.tile(np.array([[1, 4, 0, 3], [3, 4, 0, 1]], dtype=np.float64), (3, 2050, 1))
    out = np.empty((3, 32, 16, 3), dtype=np.uint8)
    np.testing.assert_array_equal(render_into(data, out), render(data, 32, 16))
    assert np.any(out[:, :, :, 0])


def test_geometry_cache_is_safe_across_python_threads(trajectory):
    from concurrent.futures import ThreadPoolExecutor

    sizes = [(21, width) for width in [1, 2, 17, 64, 128]] * 3
    data = np.tile(trajectory, (17, 3, 1))
    expected = [render(data, *size) for size in sizes]
    with ThreadPoolExecutor(max_workers=4) as pool:
        actual = list(pool.map(lambda size: render(data, *size), sizes))
    for result, reference in zip(actual, expected):
        np.testing.assert_array_equal(result, reference)
