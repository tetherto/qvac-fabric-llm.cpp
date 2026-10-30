use std::fs;
use std::io::{self, Write};

use turbovec::codebook;
use turbovec::pack;
use turbovec::rotation;
use turbovec::TurboQuantIndex;

const DEFAULT_DIM: usize = 128;
const N_DB: usize = 8;
const N_QUERY: usize = 3;
const K: usize = 3;

fn normalize_rows(values: &mut [f32], dim: usize) {
    for row in values.chunks_exact_mut(dim) {
        let norm = row
            .iter()
            .map(|v| (*v as f64) * (*v as f64))
            .sum::<f64>()
            .sqrt();
        if norm > 0.0 {
            for value in row {
                *value = (*value as f64 / norm) as f32;
            }
        }
    }
}

fn fill_value(row: usize, col: usize, phase: f64) -> f32 {
    let x = (row + 1) as f64;
    let y = (col + 1) as f64;
    (0.63 * (0.017 * x * y + phase).sin()
        + 0.31 * (0.041 * (x + 3.0) * (y + 1.0)).cos()
        + 0.06 * (0.097 * (x + y)).sin()) as f32
}

fn print_f32_array(name: &str, values: &[f32]) {
    println!("static const float {}[] = {{", name);
    for chunk in values.chunks(8) {
        print!("   ");
        for value in chunk {
            print!(" {:.9e}f,", value);
        }
        println!();
    }
    println!("}};");
}

fn print_u8_array(name: &str, values: &[u8]) {
    println!("static const uint8_t {}[] = {{", name);
    for chunk in values.chunks(16) {
        print!("   ");
        for value in chunk {
            print!(" 0x{:02x},", value);
        }
        println!();
    }
    println!("}};");
}

fn read_u32_le(bytes: &[u8]) -> u32 {
    u32::from_le_bytes(bytes.try_into().unwrap())
}

fn hash_f32_le(values: &[f32]) -> u64 {
    let mut hash = 0xcbf29ce484222325u64;
    for value in values {
        for byte in value.to_le_bytes() {
            hash ^= byte as u64;
            hash = hash.wrapping_mul(0x100000001b3);
        }
    }
    hash
}

fn hash_bytes(values: &[u8]) -> u64 {
    let mut hash = 0xcbf29ce484222325u64;
    for &byte in values {
        hash ^= byte as u64;
        hash = hash.wrapping_mul(0x100000001b3);
    }
    hash
}

fn build_query_lut(
    query: &[f32],
    centroids: &[f32],
    bits: usize,
    dim: usize,
) -> (Vec<u8>, f32, f32) {
    let codes_per_byte = 8 / bits;
    let codes_per_nibble = codes_per_byte / 2;
    let n_byte_groups = dim / codes_per_byte;
    let code_mask = (1u16 << bits) - 1;
    let mut lut = vec![0u8; n_byte_groups * 32];
    let mut values = vec![0.0f32; n_byte_groups * 32];
    let mut mins = vec![0.0f32; n_byte_groups * 2];
    let mut max_span = 0.0f32;
    let mut bias = 0.0f32;
    for group in 0..n_byte_groups {
        let dim_start = group * codes_per_byte;
        let mut lo_min = f32::MAX;
        let mut lo_max = f32::MIN;
        let mut hi_min = f32::MAX;
        let mut hi_max = f32::MIN;
        for nibble in 0u16..16 {
            let mut lo = 0.0f32;
            let mut hi = 0.0f32;
            for coordinate in 0..codes_per_nibble {
                let shift = (codes_per_nibble - 1 - coordinate) * bits;
                let code = ((nibble >> shift) & code_mask) as usize;
                lo += query[dim_start + coordinate] * centroids[code];
                hi += query[dim_start + codes_per_nibble + coordinate] * centroids[code];
            }
            values[group * 32 + nibble as usize] = lo;
            values[group * 32 + 16 + nibble as usize] = hi;
            lo_min = lo_min.min(lo);
            lo_max = lo_max.max(lo);
            hi_min = hi_min.min(hi);
            hi_max = hi_max.max(hi);
        }
        mins[group * 2] = lo_min;
        mins[group * 2 + 1] = hi_min;
        bias += lo_min + hi_min;
        max_span = max_span.max(lo_max - lo_min).max(hi_max - hi_min);
    }
    let scale = if max_span > 1e-10 {
        max_span / 127.0
    } else {
        1.0
    };
    let inverse_scale = 1.0 / scale;
    for group in 0..n_byte_groups {
        for entry in 0..16 {
            let lo = group * 32 + entry;
            let hi = lo + 16;
            lut[lo] = ((values[lo] - mins[group * 2]) * inverse_scale)
                .round()
                .clamp(0.0, 127.0) as u8;
            lut[hi] = ((values[hi] - mins[group * 2 + 1]) * inverse_scale)
                .round()
                .clamp(0.0, 127.0) as u8;
        }
    }
    (lut, scale, bias)
}

fn arg_value<'a>(args: &'a [String], flag: &str) -> Option<&'a str> {
    args.windows(2)
        .find_map(|pair| (pair[0] == flag).then_some(pair[1].as_str()))
}

fn main() {
    let args: Vec<String> = std::env::args().collect();
    let bits = args
        .get(1)
        .map(|s| s.parse::<usize>().unwrap())
        .unwrap_or(4);
    assert!(bits == 2 || bits == 4);
    let dim = arg_value(&args, "--dim")
        .map(|s| s.parse::<usize>().unwrap())
        .unwrap_or(DEFAULT_DIM);
    let prefix = arg_value(&args, "--prefix")
        .map(str::to_owned)
        .unwrap_or_else(|| {
            if bits == 2 && dim == DEFAULT_DIM {
                "kTurboVecGoldenQ2".to_string()
            } else if bits == 4 && dim == DEFAULT_DIM {
                "kTurboVecGolden".to_string()
            } else {
                format!("kTurboVecGoldenDim{}Q{}", dim, bits)
            }
        });
    if args.iter().any(|arg| arg == "--rotation-hash-only") {
        let rotation = rotation::make_rotation_matrix(dim);
        println!("0x{:016x}", hash_f32_le(&rotation));
        return;
    }
    if args.iter().any(|arg| arg == "--tqplus-hashes") {
        const N_TQPLUS: usize = 1000;
        let mut db = vec![0.0f32; N_TQPLUS * dim];
        for row in 0..N_TQPLUS {
            for col in 0..dim {
                db[row * dim + col] = fill_value(row, col, 0.47);
            }
        }
        let mut index = TurboQuantIndex::new(dim, bits).unwrap();
        index.add(&db);
        index.prepare();
        let mut queries = vec![0.0f32; N_QUERY * dim];
        for row in 0..N_QUERY {
            for col in 0..dim {
                queries[row * dim + col] = fill_value(row * 2, col, 0.29);
            }
        }
        let rotation = rotation::make_rotation_matrix(dim);
        let mut rotated_queries = vec![0.0f32; N_QUERY * dim];
        {
            let q_ref = faer::mat::from_row_major_slice::<f32, _, _>(&queries, N_QUERY, dim);
            let r_ref = faer::mat::from_row_major_slice::<f32, _, _>(&rotation, dim, dim);
            let out_mut = faer::mat::from_row_major_slice_mut::<f32, _, _>(
                &mut rotated_queries,
                N_QUERY,
                dim,
            );
            faer::linalg::matmul::matmul(
                out_mut,
                q_ref,
                r_ref.transpose(),
                None,
                1.0_f32,
                faer::Parallelism::Rayon(0),
            );
        }
        let results = index.search(&queries, K);
        let tv_path = std::env::current_dir().unwrap().join(format!(
            "tests/turbovec-golden-gen/turbovec-dim{}-q{}-tqplus.tmp.tv",
            dim, bits
        ));
        index.write(&tv_path).unwrap();
        let tv_bytes = fs::read(&tv_path).unwrap();
        fs::remove_file(&tv_path).unwrap();
        let packed_offset = 14;
        let packed_len = N_TQPLUS * bits * (dim / 8);
        let scales_offset = packed_offset + packed_len;
        let calib_count_offset = scales_offset + N_TQPLUS * 4;
        let calib_count =
            read_u32_le(&tv_bytes[calib_count_offset..calib_count_offset + 4]) as usize;
        assert_eq!(calib_count, dim);
        let shift_offset = calib_count_offset + 4;
        let tq_scale_offset = shift_offset + dim * 4;
        let (blocked_codes, _) = pack::repack(
            &tv_bytes[packed_offset..packed_offset + packed_len],
            N_TQPLUS,
            bits,
            dim,
        );
        let shifts: Vec<f32> = tv_bytes[shift_offset..shift_offset + dim * 4]
            .chunks_exact(4)
            .map(|bytes| f32::from_le_bytes(bytes.try_into().unwrap()))
            .collect();
        let tq_scales: Vec<f32> = tv_bytes[tq_scale_offset..tq_scale_offset + dim * 4]
            .chunks_exact(4)
            .map(|bytes| f32::from_le_bytes(bytes.try_into().unwrap()))
            .collect();
        let mut calibrated_query = vec![0.0f32; dim];
        let mut bias_correction = 0.0f64;
        for coordinate in 0..dim {
            calibrated_query[coordinate] = rotated_queries[coordinate] / tq_scales[coordinate];
            bias_correction -= (rotated_queries[coordinate] as f64) * (shifts[coordinate] as f64);
        }
        let (_, centroids) = codebook::codebook(bits, dim);
        let (query_lut, query_lut_scale, mut query_lut_bias) =
            build_query_lut(&calibrated_query, &centroids, bits, dim);
        let base_lut_bias = query_lut_bias;
        let bias_correction_f32 = bias_correction as f32;
        query_lut_bias += bias_correction_f32;
        println!(
            "bits={} codes=0x{:016x} blocked=0x{:016x} scales=0x{:016x} shift=0x{:016x} tqscale=0x{:016x} qrot=0x{:016x}",
            bits,
            hash_bytes(&tv_bytes[packed_offset..packed_offset + packed_len]),
            hash_bytes(&blocked_codes),
            hash_bytes(&tv_bytes[scales_offset..scales_offset + N_TQPLUS * 4]),
            hash_bytes(&tv_bytes[shift_offset..shift_offset + dim * 4]),
            hash_bytes(&tv_bytes[tq_scale_offset..tq_scale_offset + dim * 4]),
            hash_f32_le(&rotated_queries),
        );
        println!(
            "lut=0x{:016x} lut_scale={:08x} lut_bias={:08x} lut_base_bias={:08x} bias_correction={:08x} centroids=0x{:016x}",
            hash_bytes(&query_lut),
            query_lut_scale.to_bits(),
            query_lut_bias.to_bits(),
            base_lut_bias.to_bits(),
            bias_correction_f32.to_bits(),
            hash_f32_le(&centroids),
        );
        print!("scores:");
        for score in &results.scores {
            print!(" {:.9e}", score);
        }
        println!();
        print!("score_bits:");
        for score in &results.scores {
            print!(" {:08x}", score.to_bits());
        }
        println!();
        print!("indices:");
        for index in &results.indices {
            print!(" {}", index);
        }
        println!();
        return;
    }

    let mut db = vec![0.0f32; N_DB * dim];
    let mut queries = vec![0.0f32; N_QUERY * dim];
    for row in 0..N_DB {
        for col in 0..dim {
            db[row * dim + col] = fill_value(row, col, 0.13);
        }
    }
    for row in 0..N_QUERY {
        for col in 0..dim {
            queries[row * dim + col] = fill_value(row * 2, col, 0.29);
        }
    }
    normalize_rows(&mut db, dim);
    normalize_rows(&mut queries, dim);

    let mut index = TurboQuantIndex::new(dim, bits).unwrap();
    index.add(&db);
    index.prepare();
    let results = index.search(&queries, K);
    let (_, centroids) = codebook::codebook(bits, dim);
    let rotation = rotation::make_rotation_matrix(dim);
    let rotation_hash = hash_f32_le(&rotation);
    let tv_path = std::env::current_dir().unwrap().join(format!(
        "tests/turbovec-golden-gen/turbovec-dim{}-q{}.tmp.tv",
        dim, bits
    ));
    index.write(&tv_path).unwrap();
    let tv_bytes = fs::read(&tv_path).unwrap();
    fs::remove_file(&tv_path).unwrap();

    let core_offset = 5;
    assert_eq!(&tv_bytes[0..4], b"TVPI");
    assert_eq!(tv_bytes[4], 3);
    assert_eq!(tv_bytes[core_offset], bits as u8);
    assert_eq!(
        read_u32_le(&tv_bytes[core_offset + 1..core_offset + 5]) as usize,
        dim
    );
    assert_eq!(
        read_u32_le(&tv_bytes[core_offset + 5..core_offset + 9]) as usize,
        N_DB
    );
    let packed_offset = core_offset + 9;
    let packed_len = N_DB * bits * (dim / 8);
    let scales_offset = packed_offset + packed_len;
    let scales_len = N_DB;
    let calib_count_offset = scales_offset + scales_len * 4;
    let calib_count = read_u32_le(&tv_bytes[calib_count_offset..calib_count_offset + 4]) as usize;
    let packed_codes = &tv_bytes[packed_offset..packed_offset + packed_len];
    let scale_bytes = &tv_bytes[scales_offset..scales_offset + scales_len * 4];
    let rust_scales: Vec<f32> = scale_bytes
        .chunks_exact(4)
        .map(|b| f32::from_le_bytes(b.try_into().unwrap()))
        .collect();

    println!(
        "// Generated by tests/turbovec-golden-gen with turbovec 0.9, bits={}, dim={}.",
        bits, dim
    );
    print!("// Rust centroids:");
    for centroid in centroids {
        print!(" {:.9e}", centroid);
    }
    println!();
    println!("static constexpr int {}Dim = {};", prefix, dim);
    println!("static constexpr int {}NDb = {};", prefix, N_DB);
    println!("static constexpr int {}NQuery = {};", prefix, N_QUERY);
    println!("static constexpr int {}K = {};", prefix, K);
    println!("static constexpr int {}RustTvVersion = 3;", prefix);
    println!(
        "static constexpr int {}RustPackedBytes = {};",
        prefix, packed_len
    );
    println!(
        "static constexpr int {}RustScaleCount = {};",
        prefix, scales_len
    );
    println!(
        "static constexpr int {}RustCalibCount = {};",
        prefix, calib_count
    );
    println!(
        "static constexpr int {}RustTvBytesLen = {};",
        prefix,
        tv_bytes.len()
    );
    println!(
        "static constexpr uint64_t {}RustRotationHash = UINT64_C(0x{:016x});",
        prefix, rotation_hash
    );
    print_f32_array(&format!("{}Db", prefix), &db);
    print_f32_array(&format!("{}Queries", prefix), &queries);
    print_f32_array(&format!("{}RustScores", prefix), &results.scores);
    print_f32_array(&format!("{}RustScales", prefix), &rust_scales);
    print_u8_array(&format!("{}RustPackedCodes", prefix), packed_codes);
    print_u8_array(&format!("{}RustTvBytes", prefix), &tv_bytes);
    println!("static const int {}TopK[] = {{", prefix);
    for qi in 0..N_QUERY {
        print!("   ");
        for index in results.indices_for_query(qi) {
            print!(" {},", index);
        }
        println!();
    }
    println!("}};");

    io::stdout().flush().unwrap();
}
