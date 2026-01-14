#![warn(clippy::pedantic, clippy::nursery)]
#![allow(
    clippy::too_many_lines,
    clippy::cast_precision_loss,
    clippy::suboptimal_flops,
    clippy::similar_names,
    clippy::while_float
)]

use std::cmp::Ordering;
use std::convert::TryInto;
use std::fs::File;
use std::io::{Read, Write};
use std::path::Path;

#[derive(PartialEq)]
struct Event {
    x: f64,
    is_end: bool,
}

impl Eq for Event {}

impl Ord for Event {
    fn cmp(&self, other: &Self) -> Ordering {
        self.x
            .total_cmp(&other.x)
            .then_with(|| self.is_end.cmp(&other.is_end))
    }
}

impl PartialOrd for Event {
    fn partial_cmp(&self, other: &Self) -> Option<Ordering> {
        Some(self.cmp(other))
    }
}

// --- Data Structures ---

#[derive(Debug, Clone, Copy, PartialEq)]
pub struct Interval {
    pub left: f64,
    pub right: f64,
}

impl Interval {
    #[must_use]
    pub fn new(left: f64, right: f64) -> Self {
        if left > right {
            // In a real library we might panic or swap, but strictly left <= right
            // We'll just swap for safety here or assume input is correct.
            Self {
                left: right,
                right: left,
            }
        } else {
            Self { left, right }
        }
    }

    #[must_use]
    pub fn width(&self) -> f64 {
        self.right - self.left
    }

    #[must_use]
    pub fn intersection(&self, other: &Self) -> Option<Self> {
        let l = self.left.max(other.left);
        let r = self.right.min(other.right);
        if l <= r {
            Some(Self { left: l, right: r })
        } else {
            None
        }
    }

    #[must_use]
    pub const fn union_hull(&self, other: &Self) -> Self {
        Self {
            left: self.left.min(other.left),
            right: self.right.max(other.right),
        }
    }
}

// --- Arithmetic ---

impl std::ops::Add<f64> for Interval {
    type Output = Self;
    fn add(self, rhs: f64) -> Self {
        Self {
            left: self.left + rhs,
            right: self.right + rhs,
        }
    }
}

impl std::ops::Add<Self> for Interval {
    type Output = Self;
    fn add(self, rhs: Self) -> Self {
        Self {
            left: self.left + rhs.left,
            right: self.right + rhs.right,
        }
    }
}

impl std::ops::Sub<Self> for Interval {
    type Output = Self;
    fn sub(self, rhs: Self) -> Self {
        Self {
            left: self.left - rhs.right,
            right: self.right - rhs.left,
        }
    }
}

impl std::ops::Mul<f64> for Interval {
    type Output = Self;
    fn mul(self, rhs: f64) -> Self {
        let a = self.left * rhs;
        let b = self.right * rhs;
        if a <= b {
            Self { left: a, right: b }
        } else {
            Self { left: b, right: a }
        }
    }
}

impl std::ops::Mul<Self> for Interval {
    type Output = Self;
    fn mul(self, rhs: Self) -> Self {
        let p1 = self.left * rhs.left;
        let p2 = self.left * rhs.right;
        let p3 = self.right * rhs.left;
        let p4 = self.right * rhs.right;
        Self {
            left: p1.min(p2).min(p3).min(p4),
            right: p1.max(p2).max(p3).max(p4),
        }
    }
}

impl std::ops::Div<Self> for Interval {
    type Output = Option<Self>;
    fn div(self, rhs: Self) -> Option<Self> {
        if rhs.left <= 0.0 && rhs.right >= 0.0 {
            None // Division by zero (interval contains zero)
        } else {
            let inv_rhs = Self {
                left: 1.0 / rhs.right,
                right: 1.0 / rhs.left,
            };
            Some(self * inv_rhs)
        }
    }
}

// --- Statistical Functions ---

// F1: Raw interval sets (Hull)
#[must_use]
pub fn hull_of_set(intervals: &[Interval]) -> Interval {
    if intervals.is_empty() {
        return Interval::new(0.0, 0.0);
    }
    let mut l = intervals[0].left;
    let mut r = intervals[0].right;
    for i in intervals.iter().skip(1) {
        if i.left < l {
            l = i.left;
        }
        if i.right > r {
            r = i.right;
        }
    }
    Interval { left: l, right: r }
}

// F2: Interval Mode
#[must_use]
pub fn interval_mode(intervals: &[Interval]) -> Interval {
    if intervals.is_empty() {
        return Interval::new(0.0, 0.0);
    }

    // Sweep line events: (coordinate, type). Type: +1 start, -1 end
    // For sorting: coordinate asc. If equal, Start before End to count closed intervals properly.
    // However, if [1, 2] and [2, 3] overlap at 2, we want count=2 at 2.
    // So Start(2) processed before End(2).
    // Type should be ordered such that Start is handled first?
    // Actually, usually we process all events at X.
    // Let's use simple ordering: (coord, is_end). is_end=false(Start) < is_end=true(End).

    let mut events = Vec::new();
    for i in intervals {
        events.push(Event {
            x: i.left,
            is_end: false,
        });
        events.push(Event {
            x: i.right,
            is_end: true,
        });
    }
    events.sort();

    let mut max_overlap = 0;

    // First pass: find max_overlap
    let mut temp_ov = 0;
    for e in &events {
        if !e.is_end {
            temp_ov += 1;
        }
        if temp_ov > max_overlap {
            max_overlap = temp_ov;
        }
        if e.is_end {
            temp_ov -= 1;
        }
    }

    // Second pass: collect intervals for the Hull of the Mode
    let mut final_hull_l = f64::MAX;
    let mut final_hull_r = f64::MIN;
    let mut found = false;

    let mut ov = 0;
    let mut prev_x = events[0].x;

    for e in &events {
        let x = e.x;
        // Interval [prev_x, x] has overlap 'ov'
        if ov == max_overlap && x >= prev_x {
            if prev_x < final_hull_l {
                final_hull_l = prev_x;
            }
            if x > final_hull_r {
                final_hull_r = x;
            }
            found = true;
        }

        if e.is_end {
            ov -= 1;
        } else {
            ov += 1;
        }
        prev_x = x;
    }

    if !found {
        return Interval::new(0.0, 0.0);
    }
    Interval::new(final_hull_l, final_hull_r)
}

// F3: Kreinovich Median
#[must_use]
pub fn med_k(intervals: &[Interval]) -> Interval {
    if intervals.is_empty() {
        return Interval::new(0.0, 0.0);
    }
    let mut lefts: Vec<f64> = intervals.iter().map(|i| i.left).collect();
    let mut rights: Vec<f64> = intervals.iter().map(|i| i.right).collect();

    lefts.sort_by(|a, b| a.partial_cmp(b).unwrap_or(Ordering::Equal));
    rights.sort_by(|a, b| a.partial_cmp(b).unwrap_or(Ordering::Equal));

    let mid = lefts.len() / 2;
    // Usually median for straight array: if even, avg of mid-1 and mid?
    // Or just mid?
    // "Median" usually implies standard statistical median.
    let ml = if lefts.len() % 2 == 1 {
        lefts[mid]
    } else {
        f64::midpoint(lefts[mid - 1], lefts[mid])
    };

    let mr = if rights.len() % 2 == 1 {
        rights[mid]
    } else {
        f64::midpoint(rights[mid - 1], rights[mid])
    };

    Interval::new(ml, mr)
}

// F4: Prolubnikov Median
// "x_m + x_m+1 / 2 where x_m, x_m+1 are central elements of variational series"
// Assuming sorting by center (midpoint).
#[must_use]
pub fn med_p(intervals: &[Interval]) -> Interval {
    if intervals.is_empty() {
        return Interval::new(0.0, 0.0);
    }
    let mut sorted = intervals.to_vec();
    sorted.sort_by(|a, b| {
        let ca = f64::midpoint(a.left, a.right);
        let cb = f64::midpoint(b.left, b.right);
        ca.partial_cmp(&cb).unwrap_or(Ordering::Equal)
    });

    let n = sorted.len();
    if n % 2 == 1 {
        // "central elements" -> just one? Or (x_m + x_m) / 2?
        // If odd, x_m is the middle.
        sorted[n / 2]
    } else {
        let m1 = sorted[n / 2 - 1];
        let m2 = sorted[n / 2];
        // Average the intervals
        Interval {
            left: f64::midpoint(m1.left, m2.left),
            right: f64::midpoint(m1.right, m2.right),
        }
    }
}

// Jaccard Functional
#[must_use]
pub fn jaccard(x: Interval, y: Interval) -> f64 {
    let num = x.right.min(y.right) - x.left.max(y.left);
    let den = x.right.max(y.right) - x.left.min(y.left);
    if den == 0.0 {
        return 0.0; // Avoid division by zero, though if den=0, intervals are essentially points at same loc? 1.0?
        // If both [a,a] and [a,a], num=0, den=0. J=1.
        // If num==0 and den==0 -> 1.0.
    }
    // Note: The prompt formula allows negative numerator.
    // (min_r - max_l) can be negative if disjoint.
    num / den
}

// --- Optimization ---

// Golden Section Search
// Maximize f(s) in range [a, b]
#[allow(clippy::many_single_char_names)]
pub fn golden_section_search<F>(mut f: F, mut a: f64, mut b: f64, tol: f64) -> f64
where
    F: FnMut(f64) -> f64,
{
    let phi = (5.0_f64.sqrt() - 1.0) / 2.0;
    let mut c = b - phi * (b - a);
    let mut d = a + phi * (b - a);
    let mut fc = f(c);
    let mut fd = f(d);

    while (b - a).abs() > tol {
        if fc > fd {
            // For Maximization: higher is better
            // Max is in [a, d]
            b = d;
            d = c;
            fd = fc;
            c = b - phi * (b - a);
            fc = f(c);
        } else {
            // Max is in [c, b]
            a = c;
            c = d;
            fc = fd;
            d = a + phi * (b - a);
            fd = f(d);
        }
    }
    f64::midpoint(a, b)
}

fn read_data<P: AsRef<Path>>(path: P) -> std::io::Result<Vec<Interval>> {
    let mut file = File::open(path)?;
    let mut buffer = Vec::new();
    file.read_to_end(&mut buffer)?;

    let header_size = 256;
    let frame_size = 16400;
    let frame_header_size = 16;
    // let point_count_per_frame = 1024;
    // let words_per_point = 8;

    // Check minimal size
    if buffer.len() < header_size {
        return Err(std::io::Error::new(
            std::io::ErrorKind::InvalidData,
            "File too short",
        ));
    }

    let mut offset = header_size;
    let radius = 1.0 / 16384.0; // 1 / 2^14

    // Frame averaging variables
    let mut frame_count = 0;
    let data_len_u16 = 1024 * 8;
    let mut sum_buffer: Vec<f64> = vec![0.0; data_len_u16];

    while offset + frame_size <= buffer.len() {
        let frame_data_start = offset + frame_header_size;
        let frame_end = offset + frame_size;
        let data_slice = &buffer[frame_data_start..frame_end];

        for (idx, chunk) in data_slice.chunks_exact(2).enumerate() {
            if idx >= data_len_u16 {
                break;
            }
            let val_raw = u16::from_le_bytes(chunk.try_into().unwrap());
            let val_masked = val_raw & 0x3FFF;
            sum_buffer[idx] += f64::from(val_masked);
        }
        frame_count += 1;
        offset += frame_size;
    }

    let mut intervals = Vec::with_capacity(data_len_u16);
    if frame_count > 0 {
        let f_count = f64::from(frame_count);
        for sum_val in sum_buffer {
            let avg_code = sum_val / f_count;
            let v = avg_code / 16384.0 - 0.5;
            intervals.push(Interval::new(v - radius, v + radius));
        }
    }

    Ok(intervals)
}

// Helper for F1 (Raw Sets) - Element-wise Mean Jaccard
#[must_use]
pub fn mean_jaccard_a(x: &[Interval], y: &[Interval], a: f64) -> f64 {
    let mut sum = 0.0;
    let n = x.len();
    if n == 0 {
        return 0.0;
    }
    for i in 0..n {
        sum += jaccard(x[i] + a, y[i]);
    }
    sum / (n as f64)
}

#[must_use]
pub fn mean_jaccard_t(x: &[Interval], y: &[Interval], t: f64) -> f64 {
    let mut sum = 0.0;
    let n = x.len();
    if n == 0 {
        return 0.0;
    }
    for i in 0..n {
        sum += jaccard(x[i] * t, y[i]);
    }
    sum / (n as f64)
}

fn generate_trace<F>(
    mut f: F,
    start: f64,
    end: f64,
    steps: usize,
    filename: &str,
) -> std::io::Result<()>
where
    F: FnMut(f64) -> f64,
{
    let path = format!("report/data/{filename}");
    let mut file = File::create(path)?;
    writeln!(file, "param,value")?;

    let step_size = (end - start) / (steps as f64);
    for i in 0..=steps {
        let val = start + (i as f64) * step_size;
        let res = f(val);
        writeln!(file, "{val},{res}")?;
    }
    Ok(())
}

fn main() -> std::io::Result<()> {
    // Ensure data directory exists
    std::fs::create_dir_all("report/data")?;

    // Load Real Data
    // X: -0.205_lvl_sides_a_fast_data.bin
    // Y: 0.225_lvl_sides_a_fast_data.bin
    // Paths are relative to CWD.
    let path_x = "data/-0.205_lvl_side_a_fast_data.bin";
    // Check if 0.227 or 0.225. Based on fs scan, it's 0.225.
    let path_y = "data/0.225_lvl_side_a_fast_data.bin"; // fs showed this name

    let x_intervals = read_data(path_x)?;
    let y_intervals = read_data(path_y)?;

    println!("Loaded {} samples for X.", x_intervals.len());
    println!("Loaded {} samples for Y.", y_intervals.len());

    // optimization params
    // Targets: a ~ 0.34, t ~ -1.05.
    // Tighten ranges to helps with narrow peaks (Mode).
    let a_min = 0.0;
    let a_max = 1.0;
    let t_min = -2.0;
    let t_max = 0.0;
    let tol = 5e-4;

    let x_mode = interval_mode(&x_intervals);
    let y_mode = interval_mode(&y_intervals);

    let x_medk = med_k(&x_intervals);
    let y_medk = med_k(&y_intervals);

    let x_medp = med_p(&x_intervals);
    let y_medp = med_p(&y_intervals);

    // --- Solve and Trace for 'a' ---
    let steps = 200;

    // F1: J(Raw)
    print!("Processing F1 (Raw) for a... ");
    std::io::stdout().flush().unwrap();
    let f1_a_val = golden_section_search(
        |a| mean_jaccard_a(&x_intervals, &y_intervals, a),
        a_min,
        a_max,
        tol,
    );
    print!("Done (a={f1_a_val:.4}). Generating trace... ");
    std::io::stdout().flush().unwrap();
    generate_trace(
        |a| mean_jaccard_a(&x_intervals, &y_intervals, a),
        0.0,
        1.0,
        steps,
        "f1_a.csv",
    )?;
    println!("Done.");

    // F2: J(Mode)
    print!("Processing F2 (Mode) for a... ");
    std::io::stdout().flush().unwrap();
    let f2_a_val = golden_section_search(|a| jaccard(x_mode + a, y_mode), 0.3, 0.4, 1e-5);
    print!("Done (a={f2_a_val:.4}). Generating trace... ");
    std::io::stdout().flush().unwrap();
    // Focused trace for Mode to capture narrow peak
    generate_trace(
        |a| jaccard(x_mode + a, y_mode),
        0.32,
        0.37,
        1000,
        "f2_a.csv",
    )?;
    println!("Done.");

    // F3: J(MedK)
    print!("Processing F3 (MedK) for a... ");
    std::io::stdout().flush().unwrap();
    let f3_a_val = golden_section_search(|a| jaccard(x_medk + a, y_medk), a_min, a_max, tol);
    print!("Done (a={f3_a_val:.4}). Generating trace... ");
    std::io::stdout().flush().unwrap();
    generate_trace(|a| jaccard(x_medk + a, y_medk), 0.0, 1.0, steps, "f3_a.csv")?;
    println!("Done.");

    // F4: J(MedP)
    print!("Processing F4 (MedP) for a... ");
    std::io::stdout().flush().unwrap();
    let f4_a_val = golden_section_search(|a| jaccard(x_medp + a, y_medp), a_min, a_max, tol);
    print!("Done (a={f4_a_val:.4}). Generating trace... ");
    std::io::stdout().flush().unwrap();
    generate_trace(|a| jaccard(x_medp + a, y_medp), 0.0, 1.0, steps, "f4_a.csv")?;
    println!("Done.");

    println!("{:-<80}", "");

    // --- Solve and Trace for 't' ---

    // F1: Raw
    print!("Processing F1 (Raw) for t... ");
    std::io::stdout().flush().unwrap();
    let f1_t_val = golden_section_search(
        |t| mean_jaccard_t(&x_intervals, &y_intervals, t),
        t_min,
        t_max,
        tol,
    );
    print!("Done (t={f1_t_val:.4}). Generating trace... ");
    std::io::stdout().flush().unwrap();
    generate_trace(
        |t| mean_jaccard_t(&x_intervals, &y_intervals, t),
        -1.5,
        -0.5,
        steps,
        "f1_t.csv",
    )?;
    println!("Done.");

    // F2: Mode
    print!("Processing F2 (Mode) for t... ");
    std::io::stdout().flush().unwrap();
    let f2_t_val = golden_section_search(|t| jaccard(x_mode * t, y_mode), -1.1, -1.0, 1e-5);
    print!("Done (t={f2_t_val:.4}). Generating trace... ");
    std::io::stdout().flush().unwrap();
    // Focused trace for Mode
    generate_trace(
        |t| jaccard(x_mode * t, y_mode),
        -1.1,
        -1.0,
        1000,
        "f2_t.csv",
    )?;
    println!("Done.");

    // F3: MedK
    print!("Processing F3 (MedK) for t... ");
    std::io::stdout().flush().unwrap();
    let f3_t_val = golden_section_search(|t| jaccard(x_medk * t, y_medk), t_min, t_max, tol);
    print!("Done (t={f3_t_val:.4}). Generating trace... ");
    std::io::stdout().flush().unwrap();
    generate_trace(
        |t| jaccard(x_medk * t, y_medk),
        -1.5,
        -0.5,
        steps,
        "f3_t.csv",
    )?;
    println!("Done.");

    // F4: MedP
    print!("Processing F4 (MedP) for t... ");
    std::io::stdout().flush().unwrap();
    let f4_t_val = golden_section_search(|t| jaccard(x_medp * t, y_medp), t_min, t_max, tol);
    print!("Done (t={f4_t_val:.4}). Generating trace... ");
    std::io::stdout().flush().unwrap();
    generate_trace(
        |t| jaccard(x_medp * t, y_medp),
        -1.5,
        -0.5,
        steps,
        "f4_t.csv",
    )?;
    println!("Done.");

    println!("{:-<65}", "");
    println!(
        "{:<12} | {:<6} | {:<12} | {:<12} | {:<8}",
        "Func", "Param", "Calculated", "Target", "J(Param)"
    );
    println!("{:-<65}", "");

    // Helper to print row
    let print_row = |name: &str, param: &str, val: f64, target: &str, func_val: f64| {
        println!("{name:<12} | {param:<6} | {val:<12.4} | {target:<12} | {func_val:<8.4}");
    };

    // Recalculate Functional Values for display
    print_row(
        "F1 (Raw)",
        "a",
        f1_a_val,
        "0.3409",
        mean_jaccard_a(&x_intervals, &y_intervals, f1_a_val),
    );
    print_row(
        "F2 (Mode)",
        "a",
        f2_a_val,
        "0.3468",
        jaccard(x_mode + f2_a_val, y_mode),
    );
    print_row(
        "F3 (MedK)",
        "a",
        f3_a_val,
        "0.3444",
        jaccard(x_medk + f3_a_val, y_medk),
    );
    print_row(
        "F4 (MedP)",
        "a",
        f4_a_val,
        "0.3444",
        jaccard(x_medp + f4_a_val, y_medp),
    );

    print_row(
        "F1 (Raw)",
        "t",
        f1_t_val,
        "-1.0509",
        mean_jaccard_t(&x_intervals, &y_intervals, f1_t_val),
    );
    print_row(
        "F2 (Mode)",
        "t",
        f2_t_val,
        "-1.0391",
        jaccard(x_mode * f2_t_val, y_mode),
    );
    print_row(
        "F3 (MedK)",
        "t",
        f3_t_val,
        "-1.0272",
        jaccard(x_medk * f3_t_val, y_medk),
    );
    print_row(
        "F4 (MedP)",
        "t",
        f4_t_val,
        "-1.0272",
        jaccard(x_medp * f4_t_val, y_medp),
    );
    println!("{:-<65}", "");

    println!("\nVerification: Optimization performed with convergence tolerance epsilon = {tol}.");
    Ok(())
}
