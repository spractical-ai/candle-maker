use rayon::prelude::*;
use std::alloc::{alloc_zeroed, Layout};
use std::cell::RefCell;

const GREEN: [u8; 3] = [0, 255, 0];
const RED: [u8; 3] = [255, 0, 0];

#[derive(Clone, Copy)]
struct CandleX {
    center: usize,
    left: usize,
    right: usize,
}

impl CandleX {
    fn new(i: usize, candles: usize, width: usize) -> Self {
        let spacing = width as f64 / candles as f64;
        let center = (((i as f64 + 0.5) * spacing).floor() as usize).min(width - 1);
        let left = (((i as f64 + 0.15) * spacing).floor() as usize).min(width - 1);
        let right = ((((i as f64 + 0.85) * spacing).ceil() as usize).saturating_sub(1))
            .max(left)
            .min(width - 1);
        Self {
            center,
            left,
            right,
        }
    }
}

struct CandlePositions {
    candles: usize,
    width: usize,
    coordinates: Vec<CandleX>,
}

thread_local! {
    // Bounded cache per calling thread. Only layout is cached, never pixels or
    // caller data. Rayon borrows this immutable slice until its workers join.
    static POSITIONS: RefCell<Option<CandlePositions>> = const { RefCell::new(None) };
}

#[derive(Debug)]
pub enum RenderError {
    Invalid(String),
    Allocation(String),
}

pub fn render(
    data: &[f64],
    batch: usize,
    candles: usize,
    height: usize,
    width: usize,
) -> Result<Vec<u8>, RenderError> {
    let (sample_len, image_len, total_len) = validate(data, batch, candles, height, width)?;
    let mut output = zeroed_buffer(total_len)?;
    draw_batch(
        &mut output,
        data,
        sample_len,
        image_len,
        height,
        width,
        false,
    );
    Ok(output)
}

pub fn render_into(
    data: &[f64],
    batch: usize,
    candles: usize,
    height: usize,
    width: usize,
    output: &mut [u8],
) -> Result<(), RenderError> {
    // Validate everything before modifying caller-owned output.
    let (sample_len, image_len, total_len) = validate(data, batch, candles, height, width)?;
    if output.len() != total_len {
        return Err(RenderError::Invalid(
            "output size does not match image batch".into(),
        ));
    }
    draw_batch(output, data, sample_len, image_len, height, width, true);
    Ok(())
}

fn validate(
    data: &[f64],
    batch: usize,
    candles: usize,
    height: usize,
    width: usize,
) -> Result<(usize, usize, usize), RenderError> {
    if candles == 0 || height == 0 || width == 0 {
        return Err(RenderError::Invalid(
            "candles, height and width must be positive".into(),
        ));
    }
    let sample_len = candles
        .checked_mul(4)
        .ok_or_else(|| RenderError::Invalid("input shape is too large".into()))?;
    if batch.checked_mul(sample_len) != Some(data.len()) {
        return Err(RenderError::Invalid(
            "input shape does not match data".into(),
        ));
    }
    for (i, candle) in data.chunks_exact(4).enumerate() {
        let [open, high, low, close] = [candle[0], candle[1], candle[2], candle[3]];
        if !candle.iter().all(|value| value.is_finite())
            || low > high
            || low > open.min(close)
            || high < open.max(close)
        {
            return Err(RenderError::Invalid(format!(
                "invalid OHLC at sample {}, candle {}: values must be finite and low <= open/close <= high",
                i / candles, i % candles
            )));
        }
    }
    // Check even empty-batch dimensions, since NumPy strides must fit isize.
    let image_len = height
        .checked_mul(width)
        .and_then(|size| size.checked_mul(3))
        .filter(|&size| size <= isize::MAX as usize)
        .ok_or_else(|| RenderError::Invalid("image dimensions are too large".into()))?;
    let total_len = batch
        .checked_mul(image_len)
        .filter(|&size| size <= isize::MAX as usize)
        .ok_or_else(|| RenderError::Invalid("image batch is too large".into()))?;
    Ok((sample_len, image_len, total_len))
}

fn zeroed_buffer(len: usize) -> Result<Vec<u8>, RenderError> {
    if len == 0 {
        return Ok(Vec::new());
    }
    if len <= 512 * 1024 {
        // For small buffers, reserve + memset benefits from allocator reuse and
        // avoids the overhead of requesting zeroed memory from the system heap.
        let mut output = Vec::new();
        output
            .try_reserve_exact(len)
            .map_err(|error| RenderError::Allocation(error.to_string()))?;
        output.resize(len, 0);
        return Ok(output);
    }
    let layout =
        Layout::array::<u8>(len).map_err(|error| RenderError::Invalid(error.to_string()))?;
    // SAFETY: len is nonzero and Layout checked the allocation size. alloc_zeroed
    // uses the global allocator, initializes all len bytes, and aligns for u8.
    // Vec takes ownership with exactly the same element layout and capacity, so
    // its eventual deallocation matches this allocation. A null pointer is never
    // passed to Vec. No references to uninitialized pixels are created.
    unsafe {
        let pointer = alloc_zeroed(layout);
        if pointer.is_null() {
            return Err(RenderError::Allocation(format!(
                "could not allocate {len} output bytes"
            )));
        }
        Ok(Vec::from_raw_parts(pointer, len, len))
    }
}

fn draw_batch(
    output: &mut [u8],
    data: &[f64],
    sample_len: usize,
    image_len: usize,
    height: usize,
    width: usize,
    clear: bool,
) {
    let candles = sample_len / 4;
    let mut process = |positions: Option<&[CandleX]>| {
        let draw = |(image, sample): (&mut [u8], &[f64])| {
            if clear {
                image.fill(0);
            }
            draw_chart(image, sample, height, width, positions);
        };
        let batch = output.len() / image_len;
        if batch == 1 || (batch <= 8 && image_len <= 64 * 1024) {
            output
                .chunks_mut(image_len)
                .zip(data.chunks(sample_len))
                .for_each(draw);
        } else {
            output
                .par_chunks_mut(image_len)
                .zip(data.par_chunks(sample_len))
                .with_min_len(4)
                .for_each(draw);
        }
    };
    if candles <= 4096 {
        POSITIONS.with(|cache| {
            let mut cache = cache.borrow_mut();
            if cache
                .as_ref()
                .is_none_or(|entry| entry.candles != candles || entry.width != width)
            {
                *cache = Some(CandlePositions {
                    candles,
                    width,
                    coordinates: (0..candles)
                        .map(|i| CandleX::new(i, candles, width))
                        .collect(),
                });
            }
            process(Some(&cache.as_ref().unwrap().coordinates));
        });
    } else {
        process(None);
    }
}

fn draw_chart(
    image: &mut [u8],
    sample: &[f64],
    height: usize,
    width: usize,
    positions: Option<&[CandleX]>,
) {
    let low = sample
        .chunks_exact(4)
        .map(|c| c[2])
        .fold(f64::INFINITY, f64::min);
    let high = sample
        .chunks_exact(4)
        .map(|c| c[1])
        .fold(f64::NEG_INFINITY, f64::max);
    // Normalize before subtracting to avoid overflowing on extreme finite data.
    let scale = low.abs().max(high.abs()).max(f64::MIN_POSITIVE);
    let bottom = low / scale;
    let top = high / scale;
    let padding = ((height as f64 * 0.05).floor() as usize).min((height - 1) / 2);
    let span = (height - 1 - padding * 2) as f64;
    let to_y = |value: f64| -> usize {
        if low == high {
            (height - 1) / 2
        } else {
            let fraction = ((top - value / scale) / (top - bottom)).clamp(0.0, 1.0);
            (padding as f64 + fraction * span).round() as usize
        }
    };
    for (i, candle) in sample.chunks_exact(4).enumerate() {
        let [open, high, low, close] = [candle[0], candle[1], candle[2], candle[3]];
        let color = if close >= open { GREEN } else { RED };
        let CandleX {
            center,
            left,
            right,
        } = positions.map_or_else(|| CandleX::new(i, sample.len() / 4, width), |xs| xs[i]);
        for y in to_y(high)..=to_y(low) {
            let offset = (y * width + center) * 3;
            image[offset..offset + 3].copy_from_slice(&color);
        }
        let open_y = to_y(open);
        let close_y = to_y(close);
        for y in open_y.min(close_y)..=open_y.max(close_y) {
            let start = (y * width + left) * 3;
            let end = (y * width + right + 1) * 3;
            for pixel in image[start..end].chunks_exact_mut(3) {
                pixel.copy_from_slice(&color);
            }
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn overflow_is_rejected_before_allocating() {
        assert!(matches!(
            render(&[1.0; 4], 1, 1, usize::MAX, 2),
            Err(RenderError::Invalid(_))
        ));
    }

    #[test]
    fn shape_mismatch_is_rejected() {
        assert!(matches!(
            render(&[1.0; 4], 2, 1, 8, 8),
            Err(RenderError::Invalid(_))
        ));
    }

    #[test]
    fn invalid_input_does_not_modify_reused_output() {
        let mut output = vec![123; 8 * 8 * 3];
        assert!(render_into(&[2.0, 1.0, 0.0, 2.0], 1, 1, 8, 8, &mut output).is_err());
        assert!(output.iter().all(|&pixel| pixel == 123));
    }

    #[test]
    fn output_size_mismatch_is_rejected() {
        assert!(render_into(&[1.0; 4], 1, 1, 8, 8, &mut [0; 8]).is_err());
    }
}
