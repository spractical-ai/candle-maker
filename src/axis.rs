//! Small fixed bitmap labels: axes require no font files or plotting library.

pub const AXIS_COLOR: [u8; 3] = [200, 200, 200];
pub type TimeTicks = Vec<(usize, String)>;
pub type TimeLabels = Vec<Option<TimeTicks>>;

pub fn validate_time_labels(
    labels: Option<&[Option<TimeTicks>]>,
    batch: usize,
    candles: usize,
    use_axis: bool,
) -> Result<(), &'static str> {
    let Some(samples) = labels else {
        return Ok(());
    };
    if !use_axis || samples.len() != batch {
        return Err("time labels require use_axis=True and one entry per sample");
    }
    for ticks in samples.iter().flatten() {
        if ticks.is_empty() || ticks.len() > 5 {
            return Err("time labels must contain between one and five ticks per sample");
        }
        let mut previous = None;
        for (index, label) in ticks {
            if *index >= candles || previous.is_some_and(|value| *index <= value) {
                return Err("time tick indices must increase and be within the candle sequence");
            }
            previous = Some(*index);
            let lines: Vec<_> = label.split('\n').collect();
            if lines.len() != 2
                || lines.iter().any(|line| {
                    line.is_empty()
                        || line.len() > 15
                        || !line
                            .bytes()
                            .all(|byte| byte.is_ascii_digit() || matches!(byte, b'-' | b':' | b'.'))
                })
            {
                return Err("time labels must contain two nonempty ASCII numeric date/time lines of at most 15 characters");
            }
        }
    }
    Ok(())
}

#[derive(Clone, Copy)]
pub struct Plot {
    pub x: usize,
    pub y: usize,
    pub width: usize,
    pub height: usize,
}

impl Plot {
    pub fn new(height: usize, width: usize, use_axis: bool) -> Result<Self, &'static str> {
        if use_axis && (height < 64 || width < 112) {
            return Err("use_axis=True requires height >= 64 and width >= 112");
        }
        Ok(if use_axis {
            Self {
                x: 64,
                y: 8,
                width: width - 72,
                height: height - 30,
            }
        } else {
            Self {
                x: 0,
                y: 0,
                width,
                height,
            }
        })
    }

    pub fn padding(self) -> usize {
        ((self.height as f64 * 0.05).floor() as usize).min((self.height - 1) / 2)
    }

    pub fn with_time_labels(mut self) -> Self {
        // Two bitmap lines need ten more pixels than a one-line index label.
        self.height -= 10;
        self
    }
}

pub fn format_price(value: f64) -> String {
    if value == 0.0 {
        return "0".into();
    }
    if !(0.01..1e6).contains(&value.abs()) {
        return scientific(value);
    }
    let text = format!("{value:.2}");
    let text = text.trim_end_matches('0').trim_end_matches('.');
    if text.len() > 9 {
        scientific(value)
    } else {
        text.into()
    }
}

fn scientific(value: f64) -> String {
    let label = format!("{value:.2e}");
    if label.len() > 9 {
        format!("{value:.1e}")
    } else {
        label
    }
}

pub fn draw(
    image: &mut [u8],
    stride: usize,
    plot: Plot,
    low: f64,
    high: f64,
    candles: usize,
    time_ticks: Option<&TimeTicks>,
) {
    let axis_x = plot.x - 1;
    let axis_y = plot.y + plot.height;
    for y in plot.y..=axis_y {
        pixel(image, stride, axis_x, y);
    }
    for x in axis_x..plot.x + plot.width {
        pixel(image, stride, x, axis_y);
    }

    let count = if low == high {
        1
    } else {
        (plot.height / 24 + 1).clamp(2, 5)
    };
    let span = plot.height - 1 - 2 * plot.padding();
    for i in 0..count {
        let fraction = if count == 1 {
            0.5
        } else {
            i as f64 / (count - 1) as f64
        };
        let y = plot.y
            + if count == 1 {
                (plot.height - 1) / 2
            } else {
                plot.padding() + (fraction * span as f64).round() as usize
            };
        // A convex combination avoids overflowing high - low for finite prices.
        let scale = low.abs().max(high.abs()).max(f64::MIN_POSITIVE);
        let value = if low == high {
            low
        } else {
            scale * ((high / scale) * (1.0 - fraction) + (low / scale) * fraction).clamp(-1.0, 1.0)
        };
        let label = format_price(value);
        for x in axis_x - 3..axis_x {
            pixel(image, stride, x, y);
        }
        text(
            image,
            stride,
            axis_x - 6 - text_width(&label),
            y - 3,
            &label,
        );
    }

    if let Some(ticks) = time_ticks {
        for (index, label) in ticks {
            draw_x_label(image, stride, plot, candles, *index, label);
        }
        return;
    }
    let count = (plot.width / 48 + 1).clamp(2, 5).min(candles);
    for i in 0..count {
        let index = if count == 1 {
            0
        } else {
            (candles - 1) * i / (count - 1)
        };
        let label = index.to_string();
        draw_x_label(image, stride, plot, candles, index, &label);
    }
}

fn draw_x_label(
    image: &mut [u8],
    stride: usize,
    plot: Plot,
    candles: usize,
    index: usize,
    label: &str,
) {
    let axis_y = plot.y + plot.height;
    let center = (((index as f64 + 0.5) * plot.width as f64 / candles as f64).floor() as usize)
        .min(plot.width - 1)
        + plot.x;
    for y in axis_y + 1..=axis_y + 3 {
        pixel(image, stride, center, y);
    }
    for (row, line) in label.split('\n').enumerate() {
        let x = center
            .saturating_sub(text_width(line) / 2)
            .min(stride - text_width(line));
        text(image, stride, x, axis_y + 7 + row * 10, line);
    }
}

fn text_width(value: &str) -> usize {
    value.len() * 6 - 1
}

fn pixel(image: &mut [u8], stride: usize, x: usize, y: usize) {
    let offset = (y * stride + x) * 3;
    image[offset..offset + 3].copy_from_slice(&AXIS_COLOR);
}

fn text(image: &mut [u8], stride: usize, x: usize, y: usize, value: &str) {
    for (i, byte) in value.bytes().enumerate() {
        for (row, bits) in glyph(byte).iter().enumerate() {
            for column in 0..5 {
                if bits & (1 << (4 - column)) != 0 {
                    pixel(image, stride, x + i * 6 + column, y + row);
                }
            }
        }
    }
}

fn glyph(byte: u8) -> [u8; 7] {
    match byte {
        b'0' => [14, 17, 19, 21, 25, 17, 14],
        b'1' => [4, 12, 4, 4, 4, 4, 14],
        b'2' => [14, 17, 1, 2, 4, 8, 31],
        b'3' => [30, 1, 1, 14, 1, 1, 30],
        b'4' => [2, 6, 10, 18, 31, 2, 2],
        b'5' => [31, 16, 16, 30, 1, 1, 30],
        b'6' => [14, 16, 16, 30, 17, 17, 14],
        b'7' => [31, 1, 2, 4, 8, 8, 8],
        b'8' => [14, 17, 17, 14, 17, 17, 14],
        b'9' => [14, 17, 17, 15, 1, 1, 14],
        b'.' => [0, 0, 0, 0, 0, 12, 12],
        b'-' => [0, 0, 0, 31, 0, 0, 0],
        b':' => [0, 12, 12, 0, 12, 12, 0],
        b'e' => [0, 0, 14, 17, 31, 16, 14],
        _ => [0; 7],
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn finite_price_labels_fit_the_margin() {
        for value in [
            0.0,
            -0.01,
            -999999.99,
            f64::MAX,
            -f64::MAX,
            f64::from_bits(1),
            -f64::from_bits(1),
        ] {
            let label = format_price(value);
            assert!(text_width(&label) <= 57, "{label}");
            assert!(label.bytes().all(|byte| glyph(byte) != [0; 7]));
        }
    }
}
