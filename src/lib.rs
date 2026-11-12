mod axis;
mod renderer;

use numpy::{ndarray::Array4, IntoPyArray, PyArray4, PyReadonlyArray3, PyReadwriteArray4};
use pyo3::exceptions::{PyMemoryError, PyValueError};
use pyo3::prelude::*;

/// Render complete OHLC trajectories. Python's public wrapper handles coercion.
#[pyfunction]
#[pyo3(signature = (ohlc, height, width, use_axis=false))]
fn render<'py>(
    py: Python<'py>,
    ohlc: PyReadonlyArray3<'py, f64>,
    height: usize,
    width: usize,
    use_axis: bool,
) -> PyResult<Bound<'py, PyArray4<u8>>> {
    let view = ohlc.as_array();
    let (batch, candles, columns) = view.dim();
    if columns != 4 {
        return Err(PyValueError::new_err("ohlc must have shape (B, N, 4)"));
    }
    // Own the snapshot before releasing the GIL: callers may mutate their array
    // from other Python threads while Rayon renders independent charts.
    let data: Vec<f64> = match ohlc.as_slice() {
        Ok(slice) if view.is_standard_layout() => slice.to_vec(),
        _ => view.iter().copied().collect(),
    };
    let pixels = py
        .detach(|| renderer::render(&data, batch, candles, height, width, use_axis))
        .map_err(to_python_error)?;
    let images = Array4::from_shape_vec((batch, height, width, 3), pixels)
        .map_err(|error| PyValueError::new_err(error.to_string()))?;
    // Transfer the output allocation to NumPy without copying the pixel buffer.
    Ok(images.into_pyarray(py))
}

fn to_python_error(error: renderer::RenderError) -> PyErr {
    match error {
        renderer::RenderError::Invalid(message) => PyValueError::new_err(message),
        renderer::RenderError::Allocation(message) => PyMemoryError::new_err(message),
    }
}

/// Fill a caller-owned RGB buffer. Keep the GIL while borrowing Python-visible
/// mutable memory; Rayon workers operate exclusively on disjoint Rust slices.
#[pyfunction]
#[pyo3(signature = (ohlc, out, use_axis=false))]
fn render_into(
    ohlc: PyReadonlyArray3<'_, f64>,
    mut out: PyReadwriteArray4<'_, u8>,
    use_axis: bool,
) -> PyResult<()> {
    let view = ohlc.as_array();
    let (batch, candles, columns) = view.dim();
    let (out_batch, height, width, channels) = out.as_array().dim();
    if columns != 4 || out_batch != batch || channels != 3 {
        return Err(PyValueError::new_err(
            "expected OHLC (B, N, 4) and output (B, H, W, 3)",
        ));
    }
    if !view.is_standard_layout() || !out.as_array().is_standard_layout() {
        return Err(PyValueError::new_err("ohlc and out must be C-contiguous"));
    }
    let data = ohlc
        .as_slice()
        .map_err(|error| PyValueError::new_err(error.to_string()))?;
    let pixels = out
        .as_slice_mut()
        .map_err(|error| PyValueError::new_err(error.to_string()))?;
    renderer::render_into(data, batch, candles, height, width, pixels, use_axis)
        .map_err(to_python_error)
}

#[pymodule]
fn _native(module: &Bound<'_, PyModule>) -> PyResult<()> {
    module.add_function(wrap_pyfunction!(render, module)?)?;
    module.add_function(wrap_pyfunction!(render_into, module)?)?;
    Ok(())
}
