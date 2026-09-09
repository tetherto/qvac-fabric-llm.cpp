#include "mtmd-image.h"

#include <algorithm>
#include <cstring>
#include <cmath>
#include <limits>
#include <vector>

void mtmd_image_preproc_out::append(const clip_hparams & hparams, const clip_image_u8 & img, bool normalized) {
    clip_image_f32 dst;
    dst.from_u8(img);
    if (normalized) {
        dst.normalize(hparams.image_mean, hparams.image_std);
    }
    entries.push_back(std::move(dst));
}

void mtmd_image_preproc_out::append(const clip_hparams & hparams, const std::vector<clip_image_u8> & imgs, bool normalized) {
    for (const auto & img : imgs) {
        append(hparams, img, normalized);
    }
}

void mtmd_image_preproc_out::append(const clip_hparams & hparams, clip_image_f32 & img, bool normalized) {
    if (normalized) {
        img.normalize(hparams.image_mean, hparams.image_std);
    }
    entries.push_back(std::move(img));
}

void mtmd_image_preproc_out::append_overview(const clip_hparams & hparams, const clip_image_u8 & img, bool normalized) {
    overview.from_u8(img);
    if (normalized) {
        overview.normalize(hparams.image_mean, hparams.image_std);
    }
}

// set of tools to manipulate images
// in the future, we can have HW acceleration by allowing this struct to access 3rd party lib like imagick or opencv
struct img_tool {
    static void resize(
            const clip_image_u8 & src,
            clip_image_u8 & dst,
            const clip_image_size & target_resolution,
            resize_algo algo,
            pad_style padding = PAD_CEIL,
            std::array<uint8_t, 3> pad_color = {0, 0, 0}) {
        dst.set_size(target_resolution, src.is_placeholder());

        if (src.is_placeholder()) {
            // no-op for placeholder image, just set the size and return
            return;
        }

        if (dst.get_size() == src.get_size()) {
            // no resize needed, simple copy
            dst.cpy_buf(src.get_ro_buf());
            return;
        }

        if (padding == PAD_NONE) {
            // direct resize
            switch (algo) {
                case RESIZE_ALGO_BILINEAR:
                    resize_bilinear(src, dst, target_resolution.width, target_resolution.height);
                    break;
                case RESIZE_ALGO_BICUBIC:
                    resize_bicubic(src, dst, target_resolution.width, target_resolution.height);
                    break;
                case RESIZE_ALGO_BICUBIC_PILLOW:
                    resize_bicubic_pillow(src, dst, target_resolution.width, target_resolution.height);
                    break;
                case RESIZE_ALGO_LANCZOS:
                    resize_lanczos_pillow(src, dst, target_resolution.width, target_resolution.height);
                    break;
                default:
                    throw std::runtime_error("Unsupported resize algorithm");
            }
        } else {
            // resize with padding
            clip_image_u8 resized_image;
            float scale_w = static_cast<float>(target_resolution.width) / src.get_size().width;
            float scale_h = static_cast<float>(target_resolution.height) / src.get_size().height;
            float scale = std::min(scale_w, scale_h);

            int new_width, new_height;
            if (padding == PAD_NEAREST) {
                new_width  = std::min(static_cast<int>(std::round(src.get_size().width * scale)), target_resolution.width);
                new_height = std::min(static_cast<int>(std::round(src.get_size().height * scale)), target_resolution.height);
            } else {
                new_width  = std::min(static_cast<int>(std::ceil(src.get_size().width * scale)), target_resolution.width);
                new_height = std::min(static_cast<int>(std::ceil(src.get_size().height * scale)), target_resolution.height);
            }

            switch (algo) {
                case RESIZE_ALGO_BILINEAR:
                    resize_bilinear(src, resized_image, new_width, new_height);
                    break;
                case RESIZE_ALGO_BICUBIC:
                    resize_bicubic(src, resized_image, new_width, new_height);
                    break;
                case RESIZE_ALGO_BICUBIC_PILLOW:
                    resize_bicubic_pillow(src, resized_image, new_width, new_height);
                    break;
                case RESIZE_ALGO_LANCZOS:
                    resize_lanczos_pillow(src, resized_image, new_width, new_height);
                    break;
                default:
                    throw std::runtime_error("Unsupported resize algorithm");
            }

            // fill dst with pad_color
            fill(dst, pad_color);

            int offset_x, offset_y;
            if (padding == PAD_NEAREST) {
                offset_x = static_cast<int>(std::round((target_resolution.width  - new_width)  / 2.0f));
                offset_y = static_cast<int>(std::round((target_resolution.height - new_height) / 2.0f));
            } else {
                offset_x = (target_resolution.width  - new_width)  / 2;
                offset_y = (target_resolution.height - new_height) / 2;
            }
            composite(dst, resized_image, offset_x, offset_y);
        }
    }

    static void crop(const clip_image_u8 & image, clip_image_u8 & dst, int x, int y, int w, int h) {
        GGML_ASSERT(x >= 0 && y >= 0 && w > 0 && h > 0);
        GGML_ASSERT(x + w <= image.get_size().width && y + h <= image.get_size().height);
        dst.set_size({w, h}, image.is_placeholder());

        if (image.is_placeholder()) {
            // no-op for placeholder image, just set the size and return
            return;
        }

        for (int i = 0; i < h; ++i) {
            for (int j = 0; j < w; ++j) {
                dst.set_pixel(j, i, image.get_pixel(x + j, y + i));
            }
        }
    }

    struct calc_size_opt {
        int align_size = 1;
        int min_pixels = 0;   // 0 = disabled
        int max_pixels = 0;   // 0 = disabled
        // applied before min/max_pixels, so min_pixels can push an edge back above longest_edge
        int longest_edge = 0; // 0 = disabled
    };

    // calculate the size of the **resized** image, while preserving the aspect ratio and
    // aligning to the nearest multiple of align_size ("smart_resize" in transformers code)
    static clip_image_size calc_size_preserved_ratio(const clip_image_size & inp_size, const calc_size_opt & opts) {
        GGML_ASSERT(opts.align_size > 0);
        const int width  = inp_size.width;
        const int height = inp_size.height;
        if (width <= 0 || height <= 0) {
            return {0, 0};
        }

        auto round_by_factor = [f = opts.align_size](float x) { return static_cast<int>(std::round(x / static_cast<float>(f))) * f; };
        auto ceil_by_factor  = [f = opts.align_size](double x) { return static_cast<int>(std::ceil(x / static_cast<double>(f))) * f; };
        auto floor_by_factor = [f = opts.align_size](float x) { return static_cast<int>(std::floor(x / static_cast<float>(f))) * f; };

        int w_bar, h_bar;
        if (opts.longest_edge > 0) {
            // double, not float: the reference processors do this in Python floats, and where the
            // true product lands on a multiple of align_size, float32 rounding pushes it just past
            // and the ceil buys a whole extra row or column of slices. 960x720 at align 512 and
            // longest_edge 2048 is the common case: 1536 in double, 2048 in float32, so a 4x3 grid
            // becomes 4x4 and the image gains 4 slices the reference never emits.
            const double scale = std::min(static_cast<double>(opts.longest_edge) / width,
                                          static_cast<double>(opts.longest_edge) / height);
            w_bar = ceil_by_factor(width  * scale);
            h_bar = ceil_by_factor(height * scale);
        } else {
            // always align up first
            w_bar = std::max(opts.align_size, round_by_factor(width));
            h_bar = std::max(opts.align_size, round_by_factor(height));
        }

        if (opts.max_pixels > 0 && h_bar * w_bar > opts.max_pixels) {
            const auto beta = std::sqrt(static_cast<float>(height) * width / opts.max_pixels);
            h_bar = std::max(opts.align_size, floor_by_factor(height / beta));
            w_bar = std::max(opts.align_size, floor_by_factor(width  / beta));
        } else if (opts.min_pixels > 0 && h_bar * w_bar < opts.min_pixels) {
            const auto beta = std::sqrt(static_cast<float>(opts.min_pixels) / (static_cast<float>(height) * width));
            h_bar = ceil_by_factor(height * beta);
            w_bar = ceil_by_factor(width * beta);
        }

        return {w_bar, h_bar};
    }

    // draw src image into dst image at offset (offset_x, offset_y)
    static void composite(clip_image_u8 & dst, const clip_image_u8 & src, int offset_x, int offset_y) {
        if (src.is_placeholder()) {
            // no-op for placeholder image
            return;
        }

        const auto src_size = src.get_size();
        const auto dst_size = dst.get_size();
        for (int y = 0; y < src_size.height; ++y) {
            for (int x = 0; x < src_size.width; ++x) {
                int dx = x + offset_x;
                int dy = y + offset_y;
                // skip pixels that would be out of bounds in the destination
                if (dx < 0 || dy < 0 || dx >= dst_size.width || dy >= dst_size.height) {
                    continue;
                }
                dst.set_pixel(dx, dy, src.get_pixel(x, y));
            }
        }
    }

    // fill the image with a solid color
    static void fill(clip_image_u8 & img, const std::array<uint8_t, 3> & color) {
        if (img.is_placeholder()) {
            // no-op for placeholder image
            return;
        }

        const auto size = img.get_size();
        for (int y = 0; y < size.height; ++y) {
            for (int x = 0; x < size.width; ++x) {
                img.set_pixel(x, y, color);
            }
        }
    }

private:
    // Bilinear resize function
    static void resize_bilinear(const clip_image_u8 & src, clip_image_u8 & dst, int target_width, int target_height) {
        const auto src_size = src.get_size();
        if (src_size.width == 0 || src_size.height == 0) { dst.set_size({0, 0}, false); return; }
        if (target_width  <= 0) target_width  = 1;
        if (target_height <= 0) target_height = 1;

        dst.set_size({target_width, target_height}, false);

        if (src.is_placeholder()) {
            // no-op for placeholder image, just set the size and return
            return;
        }

        float x_ratio = target_width  > 1 ? static_cast<float>(src_size.width  - 1) / (target_width  - 1) : 0.0f;
        float y_ratio = target_height > 1 ? static_cast<float>(src_size.height - 1) / (target_height - 1) : 0.0f;

        for (int y = 0; y < target_height; ++y) {
            for (int x = 0; x < target_width; ++x) {
                float px = x * x_ratio;
                float py = y * y_ratio;

                int x0 = std::min(static_cast<int>(px), src_size.width  - 1);
                int y0 = std::min(static_cast<int>(py), src_size.height - 1);
                int x1 = std::min(x0 + 1, src_size.width  - 1);
                int y1 = std::min(y0 + 1, src_size.height - 1);

                float xf = px - x0;
                float yf = py - y0;

                const auto p00 = src.get_pixel(x0, y0);
                const auto p10 = src.get_pixel(x1, y0);
                const auto p01 = src.get_pixel(x0, y1);
                const auto p11 = src.get_pixel(x1, y1);

                std::array<uint8_t, 3> pixel;
                for (int c = 0; c < 3; ++c) {
                    float top    = lerp(static_cast<float>(p00[c]), static_cast<float>(p10[c]), xf);
                    float bottom = lerp(static_cast<float>(p01[c]), static_cast<float>(p11[c]), xf);
                    pixel[c] = static_cast<uint8_t>(lerp(top, bottom, yf));
                }
                dst.set_pixel(x, y, pixel);
            }
        }
    }

    // Bicubic resize function
    // part of image will be cropped if the aspect ratio is different
    static void resize_bicubic(const clip_image_u8 & img, clip_image_u8 & dst, int target_width, int target_height) {
        const auto img_size = img.get_size();
        const int nx = img_size.width;
        const int ny = img_size.height;

        dst.set_size({target_width, target_height}, false);

        if (img.is_placeholder()) {
            // no-op for placeholder image, just set the size and return
            return;
        }

        float Cc;
        float C[5] = {};
        float d0, d2, d3, a0, a1, a2, a3;
        int i, j, k, jj;
        int x, y;
        float dx, dy;
        float tx, ty;

        tx = (float)nx / (float)target_width;
        ty = (float)ny / (float)target_height;

        // Bicubic interpolation; adapted from ViT.cpp, inspired from :
        //    -> https://github.com/yglukhov/bicubic-interpolation-image-processing/blob/master/libimage.c#L36
        //    -> https://en.wikipedia.org/wiki/Bicubic_interpolation

        for (i = 0; i < target_height; i++) {
            for (j = 0; j < target_width; j++) {
                x = (int)(tx * j);
                y = (int)(ty * i);

                dx = tx * j - x;
                dy = ty * i - y;

                std::array<uint8_t, 3> pixel;
                for (k = 0; k < 3; k++) {
                    for (jj = 0; jj <= 3; jj++) {
                        d0 = img.get_pixel(clip(x - 1, 0, nx - 1), clip(y - 1 + jj, 0, ny - 1))[k] - img.get_pixel(clip(x, 0, nx - 1), clip(y - 1 + jj, 0, ny - 1))[k];
                        d2 = img.get_pixel(clip(x + 1, 0, nx - 1), clip(y - 1 + jj, 0, ny - 1))[k] - img.get_pixel(clip(x, 0, nx - 1), clip(y - 1 + jj, 0, ny - 1))[k];
                        d3 = img.get_pixel(clip(x + 2, 0, nx - 1), clip(y - 1 + jj, 0, ny - 1))[k] - img.get_pixel(clip(x, 0, nx - 1), clip(y - 1 + jj, 0, ny - 1))[k];
                        a0 = img.get_pixel(clip(x, 0, nx - 1), clip(y - 1 + jj, 0, ny - 1))[k];

                        a1 = -1.0 / 3 * d0 + d2 - 1.0 / 6 * d3;
                        a2 =  1.0 / 2 * d0 +      1.0 / 2 * d2;
                        a3 = -1.0 / 6 * d0 -      1.0 / 2 * d2 + 1.0 / 6 * d3;

                        C[jj] = a0 + a1 * dx + a2 * dx * dx + a3 * dx * dx * dx;

                        d0 = C[0] - C[1];
                        d2 = C[2] - C[1];
                        d3 = C[3] - C[1];
                        a0 = C[1];
                        a1 = -1.0 / 3 * d0 + d2 - 1.0 / 6 * d3;
                        a2 =  1.0 / 2 * d0 +      1.0 / 2 * d2;
                        a3 = -1.0 / 6 * d0 -      1.0 / 2 * d2 + 1.0 / 6 * d3;
                        Cc = a0 + a1 * dy + a2 * dy * dy + a3 * dy * dy * dy;

                        const uint8_t Cc2 = std::min(std::max(std::round(Cc), 0.0f), 255.0f);
                        pixel[k] = Cc2;
                    }
                }
                dst.set_pixel(j, i, pixel);
            }
        }
    }

    // Pillow-compatible separable resampling (Bicubic and Lanczos)
    // Adapted from https://github.com/python-pillow/Pillow/blob/main/src/libImaging/Resample.c
    //
    // Key properties:
    // 1. Separable filtering: horizontal pass followed by vertical pass
    // 2. Pre-computes normalized filter coefficients for each output pixel
    // 3. Fixed-point integer arithmetic (22 fractional bits) for speed and determinism
    static bool resize_bicubic_pillow(const clip_image_u8 & img, clip_image_u8 & dst, int target_width, int target_height) {
        return resize_pillow(img, dst, target_width, target_height, /*use_lanczos=*/false);
    }

    // Lanczos-3 (support radius 3), matches Pillow's Image.LANCZOS
    static bool resize_lanczos_pillow(const clip_image_u8 & img, clip_image_u8 & dst, int target_width, int target_height) {
        return resize_pillow(img, dst, target_width, target_height, /*use_lanczos=*/true);
    }

    static bool resize_pillow(
            const clip_image_u8 & img,
            clip_image_u8 & dst,
            int target_width,
            int target_height,
            bool use_lanczos) {
        // Fixed-point precision: 22 bits = 32 (int32_t) - 8 (uint8_t pixels) - 2 (headroom for accumulation)
        // This allows encoding fractional weights as integers: weight * 2^22
        const int PRECISION_BITS = 32 - 8 - 2;

        // Resample filter: Lanczos-3 (support [-3, 3]) or bicubic with a = -0.5 (support [-2, 2])
        // Note: GGML/PyTorch bicubic uses a = -0.75, Pillow uses a = -0.5
        // Returns filter weight for distance x from pixel center
        auto resample_filter = [use_lanczos](double x) -> double {
            if (use_lanczos) {
                if (-3.0 <= x && x < 3.0) {
                    auto sinc = [](double v) {
                        if (v == 0.0) {
                            return 1.0;
                        }
                        const double pi_v = v * 3.141592653589793238462643383279502884;
                        return std::sin(pi_v) / pi_v;
                    };
                    return sinc(x) * sinc(x / 3.0);
                }
                return 0.0;
            }

            constexpr double a = -0.5;
            if (x < 0.0) {
                x = -x;
            }
            if (x < 1.0) {
                return ((a + 2.0) * x - (a + 3.0)) * x * x + 1;
            }
            if (x < 2.0) {
                return (((x - 5) * x + 8) * x - 4) * a;
            }
            return 0.0;  // Zero outside [-2, 2]
        };

        // Filter support radius: 2 for bicubic, 3 for lanczos
        const double filter_support = use_lanczos ? 3.0 : 2.0;

        // Clipping function for 8-bit values
        auto clip8 = [](int val) -> uint8_t {
            if (val < 0) return 0;
            if (val > 255) return 255;
            return static_cast<uint8_t>(val);
        };

        // Precompute filter coefficients for ONE dimension (horizontal or vertical)
        //
        // Parameters:
        //   inSize  - Number of pixels in input dimension (e.g., src_width or src_height)
        //   outSize - Number of pixels in output dimension (e.g., target_width or target_height)
        //   bounds  - [OUTPUT] Array of size outSize*2 storing input pixel ranges:
        //             bounds[xx*2+0] = first input pixel index for output pixel xx (xmin)
        //             bounds[xx*2+1] = number of input pixels for output pixel xx (xcnt)
        //   weights - [OUTPUT] Array of size outSize*ksize storing fixed-point filter weights:
        //             kk[xx*ksize + x] = weight for input pixel x contributing to output pixel xx
        //
        // Returns: kernel size (ksize) - number of input pixels that contribute to each output pixel
        auto precompute_weights = [&](int inSize, int outSize,
                                     std::vector<int> & bounds, std::vector<int32_t> & weights) -> int {
            GGML_ASSERT(inSize > 0 && outSize > 0);
            double support, scale, filterscale;
            double center, ww, ss;
            int xx, x, ksize, xmin, xmax;

            // Calculate scaling factor: ratio of input range to output size
            filterscale = scale = static_cast<double>(inSize) / outSize;
            // For upsampling (scale < 1), keep filterscale = 1 to maintain filter sharpness
            // For downsampling (scale > 1), widen filter to prevent aliasing
            if (filterscale < 1.0) {
                filterscale = 1.0;
            }

            // Determine filter support radius and kernel size
            support = filter_support * filterscale;  // Widen filter when downsampling
            ksize = static_cast<int>(std::ceil(support)) * 2 + 1;  // Total pixels in kernel

            std::vector<double> pre_weights(outSize * ksize);  // Temporary weights
            bounds.resize(outSize * 2);


            // For each output pixel, compute its filter coefficients
            for (xx = 0; xx < outSize; xx++) {
                // Calculate the center position in input space (pixel-center convention: +0.5)
                center = (xx + 0.5) * scale;
                ww = 0.0;  // Sum of weights for normalization
                ss = 1.0 / filterscale;  // Scale factor for filter function

                // Determine the range of input pixels that contribute to this output pixel
                xmin = static_cast<int>(center - support + 0.5);
                if (xmin < 0) {
                    xmin = 0;
                }

                xmax = static_cast<int>(center + support + 0.5);
                if (xmax > inSize) {
                    xmax = inSize;
                }

                xmax -= xmin;

                // Compute filter weights for each contributing input pixel
                for (x = 0; x < xmax; x++) {
                    // Distance from input pixel center to output pixel center in input space
                    double w = resample_filter((x + xmin - center + 0.5) * ss);
                    pre_weights[xx * ksize + x] = w;
                    ww += w;  // Accumulate for normalization
                }

                // Normalize weights to sum to 1.0 (preserves brightness)
                for (x = 0; x < xmax; x++) {
                    if (ww != 0.0) {
                        pre_weights[xx * ksize + x] /= ww;
                    }
                }

                // Zero-pad remaining kernel positions
                for (; x < ksize; x++) {
                    pre_weights[xx * ksize + x] = 0;
                }

                // Store input pixel range for this output pixel
                bounds[xx * 2 + 0] = xmin;
                bounds[xx * 2 + 1] = xmax;
            }

            // Convert floating-point coefficients to fixed-point integers
            // Formula: int32 = round(float * 2^PRECISION_BITS)
            weights.resize(outSize * ksize);

            const double fxp_scale = std::ldexp(1.0, PRECISION_BITS); // 1.0 * 2^PRECISION_BITS

            for (int i = 0; i < outSize * ksize; i++) {
                if (use_lanczos) {
                    // Pillow adds +/- 0.5 then truncates toward zero; std::round would round twice
                    const double rounded = pre_weights[i] * fxp_scale + (pre_weights[i] < 0 ? -0.5 : 0.5);
                    weights[i] = static_cast<int32_t>(rounded);
                    continue;
                }
                double tmp_val = pre_weights[i] * fxp_scale;
                if (pre_weights[i] < 0) {
                    tmp_val -= 0.5;
                } else {
                    tmp_val += 0.5;
                }
                tmp_val = std::round(tmp_val);
                tmp_val = std::clamp(tmp_val,
                                     static_cast<double>(std::numeric_limits<int32_t>::min()),
                                     static_cast<double>(std::numeric_limits<int32_t>::max()));
                weights[i] = static_cast<int32_t>(tmp_val);
            }

            return ksize;
        };

        // Horizontal resampling pass
        // Resizes width from imIn to out_nx, preserving height
        auto resample_horizontal = [&](const clip_image_u8 & imIn, clip_image_u8 & imOut,
                                       int out_nx,
                                       int ksize, const std::vector<int> & bounds, const std::vector<int32_t> & weights) {
            const int in_ny = imIn.get_size().height;
            imOut.set_size({out_nx, in_ny}, false);

            // Process each row independently
            for (int yy = 0; yy < in_ny; yy++) {
                // For each output pixel in this row
                for (int xx = 0; xx < out_nx; xx++) {
                    // Get the range of input pixels and filter coefficients
                    int xmin = bounds[xx * 2 + 0];  // First input pixel index
                    int xcnt = bounds[xx * 2 + 1];  // Number of input pixels

                    // Initialize accumulators for RGB channels with rounding bias (0.5 in fixed-point)
                    int32_t ss0 = 1 << (PRECISION_BITS - 1);
                    int32_t ss1 = 1 << (PRECISION_BITS - 1);
                    int32_t ss2 = 1 << (PRECISION_BITS - 1);

                    // Convolve: sum weighted input pixels
                    for (int x = 0; x < xcnt; x++) {
                        const auto src_px = imIn.get_pixel(x + xmin, yy);
                        ss0 += src_px[0] * weights[xx * ksize + x];  // R channel
                        ss1 += src_px[1] * weights[xx * ksize + x];  // G channel
                        ss2 += src_px[2] * weights[xx * ksize + x];  // B channel
                    }

                    // Convert back from fixed-point (divide by 2^PRECISION_BITS) and clamp to [0,255]
                    imOut.set_pixel(xx, yy, {clip8(ss0 >> PRECISION_BITS),
                                             clip8(ss1 >> PRECISION_BITS),
                                             clip8(ss2 >> PRECISION_BITS)});
                }
            }
        };

        // Vertical resampling pass
        // Resizes height from imIn to out_ny, preserving width
        auto resample_vertical = [&](const clip_image_u8 & imIn, clip_image_u8 & imOut,
                                     int out_ny,
                                     int ksize, const std::vector<int> & bounds, const std::vector<int32_t> & weight) {
            const int in_nx = imIn.get_size().width;
            imOut.set_size({in_nx, out_ny}, false);

            // For each output row
            for (int yy = 0; yy < out_ny; yy++) {
                // Get the range of input rows and filter coefficients
                int ymin = bounds[yy * 2 + 0];  // First input row index
                int ycnt = bounds[yy * 2 + 1];  // Number of input rows

                // Process each column in this output row
                for (int xx = 0; xx < in_nx; xx++) {
                    // Initialize accumulators for RGB channels with rounding bias
                    int32_t ss0 = 1 << (PRECISION_BITS - 1);
                    int32_t ss1 = 1 << (PRECISION_BITS - 1);
                    int32_t ss2 = 1 << (PRECISION_BITS - 1);

                    // Convolve: sum weighted input pixels vertically
                    for (int y = 0; y < ycnt; y++) {
                        const auto src_px = imIn.get_pixel(xx, y + ymin);
                        ss0 += src_px[0] * weight[yy * ksize + y];  // R channel
                        ss1 += src_px[1] * weight[yy * ksize + y];  // G channel
                        ss2 += src_px[2] * weight[yy * ksize + y];  // B channel
                    }

                    // Convert back from fixed-point and clamp to [0,255]
                    imOut.set_pixel(xx, yy, {clip8(ss0 >> PRECISION_BITS),
                                             clip8(ss1 >> PRECISION_BITS),
                                             clip8(ss2 >> PRECISION_BITS)});
                }
            }
        };

        // Main resampling logic using separable two-pass approach
        const int src_width  = img.get_size().width;
        const int src_height = img.get_size().height;

        bool need_horizontal = (target_width != src_width);
        bool need_vertical = (target_height != src_height);

        // Precompute filter coefficients for both dimensions
        std::vector<int> bounds_horiz, bounds_vert;
        std::vector<int32_t> weights_horiz, weights_vert;
        int ksize_horiz = 0, ksize_vert = 0;

        if (need_horizontal) {
            ksize_horiz = precompute_weights(src_width, target_width, bounds_horiz, weights_horiz);
        }

        if (need_vertical) {
            ksize_vert = precompute_weights(src_height, target_height, bounds_vert, weights_vert);
        }

        // Perform two-pass resampling
        if (need_horizontal && need_vertical) {
            // Both horizontal and vertical
            clip_image_u8 temp;
            resample_horizontal(img, temp, target_width, ksize_horiz, bounds_horiz, weights_horiz);
            resample_vertical(temp, dst, target_height, ksize_vert, bounds_vert, weights_vert);
        } else if (need_horizontal) {
            // Only horizontal
            resample_horizontal(img, dst, target_width, ksize_horiz, bounds_horiz, weights_horiz);
        } else if (need_vertical) {
            // Only vertical
            resample_vertical(img, dst, target_height, ksize_vert, bounds_vert, weights_vert);
        } else {
            // No resizing needed - direct copy
            dst.set_size(img.get_size(), img.is_placeholder());
            if (!img.is_placeholder()) {
                dst.cpy_buf(img.get_ro_buf());
            }
        }

        return true;
    }

    static inline int clip(int x, int lower, int upper) {
        return std::max(lower, std::min(x, upper));
    }

    // Linear interpolation between two points
    static inline float lerp(float s, float e, float t) {
        return s + (e - s) * t;
    }
};


//
// mtmd_image_preprocessor_llava_uhd
//

mtmd_image_preproc_out mtmd_image_preprocessor_llava_uhd::preprocess(const clip_image_u8 & img) {
    const clip_image_size original_size = img.get_size();
    auto const inst = get_slice_instructions(original_size);
    auto sliced = slice_image(img, inst);

    mtmd_image_preproc_out output;
    output.append_overview(hparams, sliced.overview, true);
    output.append(hparams, sliced.slices, true);
    output.grid_x = inst.grid_size.width;
    output.grid_y = inst.grid_size.height;

    return output;
}

mtmd_image_preprocessor_llava_uhd::slice_instructions mtmd_image_preprocessor_llava_uhd::get_slice_instructions(const clip_image_size & original_size) {
    mtmd_image_preprocessor_llava_uhd::slice_instructions res;
    // align slices by patch_size * n_merge so an integer number of merger output tokens fits per slice
    const int n_merge         = hparams.n_merge;
    const int patch_size      = hparams.patch_size * n_merge;
    const int slice_size      = hparams.image_size;
    const int original_width  = original_size.width;
    const int original_height = original_size.height;

    const bool has_slices    = original_size.width > slice_size || original_size.height > slice_size;
    const bool has_pinpoints = !hparams.image_res_candidates.empty();

    if (!has_slices) {
        // skip slicing logic
        res.overview_size = clip_image_size{slice_size, slice_size};
        res.refined_size  = clip_image_size{0, 0};
        res.grid_size     = clip_image_size{0, 0};

        return res;
    }

    if (has_pinpoints) {
        // has pinpoints, use them to calculate the grid size (e.g. llava-1.6)
        auto refine_size = select_best_resolution(
            original_size,
            hparams.image_res_candidates);
        res.overview_size         = clip_image_size{slice_size, slice_size};
        res.refined_size          = refine_size;
        res.grid_size             = clip_image_size{0, 0};

        LOG_DBG("%s: using pinpoints for slicing\n", __func__);
        LOG_DBG("%s: original size: %d x %d, overview size: %d x %d, refined size: %d x %d\n",
                __func__, original_width, original_height,
                res.overview_size.width, res.overview_size.height,
                res.refined_size.width,  res.refined_size.height);

        for (int y = 0; y < refine_size.height; y += slice_size) {
            for (int x = 0; x < refine_size.width; x += slice_size) {
                slice_coordinates slice;
                slice.x = x;
                slice.y = y;
                slice.size.width  = std::min(slice_size, refine_size.width  - x);
                slice.size.height = std::min(slice_size, refine_size.height - y);
                res.slices.push_back(slice);
                LOG_DBG("%s: slice %d: x=%d, y=%d, size=%dx%d\n",
                        __func__, (int)res.slices.size() - 1,
                        slice.x, slice.y, slice.size.width, slice.size.height);
            }
        }

        res.grid_size.height = refine_size.height / slice_size;
        res.grid_size.width  = refine_size.width  / slice_size;
        LOG_DBG("%s: grid size: %d x %d\n", __func__, res.grid_size.width, res.grid_size.height);

        return res;
    }

    // no pinpoints, dynamically calculate the grid size (e.g. minicpmv)

    auto best_size    = get_best_resize(original_size, slice_size, patch_size, !has_slices);
    res.overview_size = best_size;

    {
        const int max_slice_nums = 9; // TODO: this is only used by minicpmv, maybe remove it
        const float log_ratio = log((float)original_width / original_height);
        const float ratio = (float)original_width * original_height / (slice_size * slice_size);
        const int multiple = fmin(ceil(ratio), max_slice_nums);

        auto best_grid   = get_best_grid(max_slice_nums, multiple, log_ratio);
        auto refine_size = get_refine_size(original_size, best_grid, slice_size, patch_size, true);
        res.grid_size    = best_grid;
        res.refined_size = refine_size;

        LOG_DBG("%s: original size: %d x %d, overview size: %d x %d, refined size: %d x %d, grid size: %d x %d\n",
                __func__, original_width, original_height,
                res.overview_size.width, res.overview_size.height,
                res.refined_size.width, res.refined_size.height,
                res.grid_size.width, res.grid_size.height);

        int width  = refine_size.width;
        int height = refine_size.height;
        int grid_x = int(width  / best_grid.width);
        int grid_y = int(height / best_grid.height);
        for (int patches_y = 0,                    ic = 0;
                patches_y < refine_size.height && ic < best_grid.height;
                patches_y += grid_y,              ic += 1) {
            for (int patches_x = 0,                   jc = 0;
                    patches_x < refine_size.width && jc < best_grid.width;
                    patches_x += grid_x,             jc += 1) {
                slice_coordinates slice;
                slice.x = patches_x;
                slice.y = patches_y;
                slice.size.width  = grid_x;
                slice.size.height = grid_y;
                res.slices.push_back(slice);
                LOG_DBG("%s: slice %d: x=%d, y=%d, size=%dx%d\n",
                        __func__, (int)res.slices.size() - 1,
                        slice.x, slice.y, slice.size.width, slice.size.height);
            }
        }
    }

    return res;
}

mtmd_image_preprocessor_llava_uhd::slice_output mtmd_image_preprocessor_llava_uhd::slice_image(const clip_image_u8 & img, const mtmd_image_preprocessor_llava_uhd::slice_instructions & inst) {
    slice_output output;

    // resize to overview size
    img_tool::resize(img, output.overview, inst.overview_size, hparams.image_resize_algo_ov,
                        hparams.image_pad_ov, hparams.image_pad_color_ov);

    if (inst.slices.empty()) {
        // no slices, just return the overview image
        return output;
    }

    // resize to refined size
    clip_image_u8 refined_img;
    img_tool::resize(img, refined_img, inst.refined_size, hparams.image_resize_algo_rf,
                        hparams.image_pad_rf, hparams.image_pad_color_rf);

    // create slices
    for (const auto & slice : inst.slices) {
        int x = slice.x;
        int y = slice.y;
        int w = slice.size.width;
        int h = slice.size.height;

        clip_image_u8 img_slice;
        img_tool::crop(refined_img, img_slice, x, y, w, h);
        output.slices.push_back(std::move(img_slice));
    }

    return output;
}

clip_image_size mtmd_image_preprocessor_llava_uhd::get_best_resize(const clip_image_size & original_size, int scale_resolution, int patch_size, bool allow_upscale) {
    int width  = original_size.width;
    int height = original_size.height;
    if ((width * height > scale_resolution * scale_resolution) || allow_upscale) {
        float r = static_cast<float>(width) / height;
        height  = static_cast<int>(scale_resolution / std::sqrt(r));
        width   = static_cast<int>(height * r);
    }
    clip_image_size res;
    res.width  = ensure_divide(width,  patch_size);
    res.height = ensure_divide(height, patch_size);
    return res;
}

clip_image_size mtmd_image_preprocessor_llava_uhd::resize_maintain_aspect_ratio(const clip_image_size & orig, const clip_image_size & target_max) {
    float scale_width  = static_cast<float>(target_max.width)  / orig.width;
    float scale_height = static_cast<float>(target_max.height) / orig.height;
    float scale = std::min(scale_width, scale_height);
    return clip_image_size{
        static_cast<int>(orig.width  * scale),
        static_cast<int>(orig.height * scale),
    };
}

clip_image_size mtmd_image_preprocessor_llava_uhd::select_best_resolution(const clip_image_size & original_size, const std::vector<clip_image_size> & possible_resolutions) {
    clip_image_size best_fit;
    int min_wasted_area = std::numeric_limits<int>::max();
    int max_effective_resolution = 0;

    for (const clip_image_size & candidate : possible_resolutions) {
        auto target_size = resize_maintain_aspect_ratio(original_size, candidate);
        int effective_resolution = std::min(
            target_size.width * target_size.height,
            original_size.width * original_size.height);
        int wasted_area = (candidate.width * candidate.height) - effective_resolution;

        if (effective_resolution > max_effective_resolution || (effective_resolution == max_effective_resolution && wasted_area < min_wasted_area)) {
            max_effective_resolution = effective_resolution;
            min_wasted_area = wasted_area;
            best_fit = candidate;
        }

        LOG_DBG("%s: candidate: %d x %d, target: %d x %d, wasted: %d, effective: %d\n", __func__, candidate.width, candidate.height, target_size.width, target_size.height, wasted_area, effective_resolution);
    }

    return best_fit;
}

int mtmd_image_preprocessor_llava_uhd::ensure_divide(int length, int patch_size) {
    return std::max(static_cast<int>(std::round(static_cast<float>(length) / patch_size) * patch_size), patch_size);
}

clip_image_size mtmd_image_preprocessor_llava_uhd::get_refine_size(const clip_image_size & original_size, const clip_image_size & grid, int scale_resolution, int patch_size, bool allow_upscale) {
    int width  = original_size.width;
    int height = original_size.height;
    int grid_x = grid.width;
    int grid_y = grid.height;

    int refine_width  = ensure_divide(width, grid_x);
    int refine_height = ensure_divide(height, grid_y);

    clip_image_size grid_size;
    grid_size.width  = refine_width  / grid_x;
    grid_size.height = refine_height / grid_y;

    auto best_grid_size  = get_best_resize(grid_size, scale_resolution, patch_size, allow_upscale);
    int best_grid_width  = best_grid_size.width;
    int best_grid_height = best_grid_size.height;

    clip_image_size refine_size;
    refine_size.width  = best_grid_width  * grid_x;
    refine_size.height = best_grid_height * grid_y;
    return refine_size;
}

// Pick the grid from `candidates` whose log(width/height) is closest to `target_log_ratio`.
// Strict-less keeps the first candidate on ties, so callers control tie-breaking via ordering.
// Shared by the log-ratio tilers (llava_uhd, qwen3vl); lfm2 deliberately uses a different
// (linear ratio diff + area) metric and is not routed through here.
static clip_image_size pick_grid_by_log_ratio(const std::vector<clip_image_size> & candidates, const float target_log_ratio) {
    clip_image_size best_grid{1, 1};
    float min_error = std::numeric_limits<float>::infinity();
    for (const auto & grid : candidates) {
        const float error = std::abs(target_log_ratio - std::log(1.0f * grid.width / grid.height));
        if (error < min_error) {
            min_error = error;
            best_grid = grid;
        }
    }
    return best_grid;
}

clip_image_size mtmd_image_preprocessor_llava_uhd::get_best_grid(const int max_slice_nums, const int multiple, const float log_ratio) {
    std::vector<int> candidate_split_grids_nums;
    for (int i : {multiple - 1, multiple, multiple + 1}) {
        if (i == 1 || i > max_slice_nums) {
            continue;
        }
        candidate_split_grids_nums.push_back(i);
    }

    std::vector<clip_image_size> candidate_grids;
    for (int split_grids_nums : candidate_split_grids_nums) {
        int m = 1;
        while (m <= split_grids_nums) {
            if (split_grids_nums % m == 0) {
                candidate_grids.push_back(clip_image_size{m, split_grids_nums / m});
            }
            ++m;
        }
    }

    return pick_grid_by_log_ratio(candidate_grids, log_ratio);
}

//
// mtmd_image_preprocessor_fixed_size
//

mtmd_image_preproc_out mtmd_image_preprocessor_fixed_size::preprocess(const clip_image_u8 & img) {
    clip_image_u8 resized_image;
    int sz = hparams.image_size;
    img_tool::resize(img, resized_image, {sz, sz},
                        hparams.image_resize_algo,
                        hparams.image_resize_pad,
                        hparams.image_pad_color);
    mtmd_image_preproc_out output;
    output.append(hparams, resized_image, true);
    return output;
}

// Resize to [min_pixels, max_pixels] with aspect ratio preserved, aligned to `align_px`, single f32 output.
static mtmd_image_preproc_out preprocess_dyn_size_aligned(
        const clip_image_u8 & img,
        const clip_hparams & hparams,
        const int align_px) {
    GGML_ASSERT(hparams.image_min_pixels > 0 && hparams.image_max_pixels > 0);
    GGML_ASSERT(align_px > 0);
    clip_image_u8 resized_image;
    const clip_image_size img_size = img.get_size();
    const clip_image_size target_size = img_tool::calc_size_preserved_ratio(
        img_size,
        {
            /* align_size   */ align_px,
            /* min_pixels   */ hparams.image_min_pixels,
            /* max_pixels   */ hparams.image_max_pixels,
            /* longest_edge */ 0,
        });
    img_tool::resize(img, resized_image, target_size,
                     hparams.image_resize_algo,
                     hparams.image_resize_pad,
                     hparams.image_pad_color);
    mtmd_image_preproc_out output;
    output.append(hparams, resized_image, true);
    return output;
}

static int dyn_size_align_px(const clip_hparams & hparams) {
    // the original pixtral model doesn't have n_merge
    const int cur_merge = hparams.n_merge == 0 ? 1 : hparams.n_merge;
    return hparams.patch_size * cur_merge;
}

//
// mtmd_image_preprocessor_dyn_size
//

mtmd_image_preproc_out mtmd_image_preprocessor_dyn_size::preprocess(const clip_image_u8 & img) {
    return preprocess_dyn_size_aligned(img, hparams, dyn_size_align_px(hparams));
}

//
// mtmd_image_preprocessor_qwen3vl
//

mtmd_image_preproc_out mtmd_image_preprocessor_qwen3vl::preprocess(const clip_image_u8 & img) {
    mtmd_image_preproc_out output;

    // Tile size comes from the model's image_size (e.g. 768 for Qwen3VL-2B, patch_size=16).
    const int tile_px = hparams.image_size;
    GGML_ASSERT(tile_px > 0 && "image_size not set in model hparams");

    GGML_ASSERT(hparams.patch_size > 0);
    // Guard the log-ratio division below; the public API asserts this in mtmd.cpp, but a direct
    // caller (fuzz harness, unit test, malformed upload reaching this path) could reach here with
    // a zero-dimension image. Fail this one image gracefully instead of aborting the whole process.
    const clip_image_size img_size = img.get_size();
    if (img_size.width <= 0 || img_size.height <= 0) {
        LOG_ERR("%s: image has zero dimension (%dx%d)\n", __func__, img_size.width, img_size.height);
        return output;
    }

    // Select the grid (n_col × n_row, n_col*n_row <= max_tiles, excluding 1×1) that:
    //   1. downscales the image in both dimensions (upscaling gains no detail)
    //   2. maximises tile count (more tiles = more spatial detail preserved)
    //   3. among equal tile counts, minimises log-ratio error (less stretch distortion)
    // If no grid downscales, fall back to dyn_size (single-tile at native resolution).
    const float img_log_ratio = std::log((float)img_size.width / (float)img_size.height);

    struct grid_cand { int col, row; float ratio_err; };
    std::vector<grid_cand> fitting;
    // candidate count is sum_{col} floor(max_tiles/col) ≈ max_tiles*ln(max_tiles), not max_tiles²;
    // a small reserve avoids the early reallocations without over-allocating at large max_tiles.
    fitting.reserve((size_t)(max_tiles * std::log((float)std::max(max_tiles, 2))));
    for (int col = 1; col <= max_tiles; col++) {
        for (int row = 1; col * row <= max_tiles; row++) {
            if (col == 1 && row == 1) { continue; } // 1×1 == dyn_size below
            if (col * tile_px <= img_size.width && row * tile_px <= img_size.height) {
                const float err = std::abs(img_log_ratio - std::log((float)col / (float)row));
                fitting.push_back({col, row, err});
            }
        }
    }
    if (fitting.empty()) {
        LOG_INF("%s: %dx%d fits no tile grid (tile_px=%d, max_tiles=%d) — using dyn_size\n",
                __func__, img_size.width, img_size.height, tile_px, max_tiles);
        output = preprocess_dyn_size_aligned(img, hparams, dyn_size_align_px(hparams));
        // grid_x/grid_y left at 0 → single-tile path; tokenizer treats this as 1×1
        return output;
    }

    // Tolerance band: find the best (minimum) ratio error, then allow any candidate within
    // TOLERANCE of it to compete on tile count. This prevents a high-tile-count grid from
    // winning when a near-perfect-ratio lower-tile-count grid exists (e.g. 2×2 over 2×1
    // for a 2:1 image, which would squash the image 2× in one dimension).
    static constexpr float RATIO_ERR_TOLERANCE = 0.2f;
    const float best_ratio_err = std::min_element(fitting.begin(), fitting.end(),
        [](const grid_cand & a, const grid_cand & b) { return a.ratio_err < b.ratio_err; })->ratio_err;
    const float eligible_threshold = best_ratio_err + RATIO_ERR_TOLERANCE;

    auto best = std::min_element(fitting.begin(), fitting.end(), [eligible_threshold](const grid_cand & a, const grid_cand & b) {
        const bool ea = a.ratio_err <= eligible_threshold;
        const bool eb = b.ratio_err <= eligible_threshold;
        if (ea != eb) { return ea; }        // eligible always beats ineligible
        const int ta = a.col * a.row;
        const int tb = b.col * b.row;
        if (ta != tb) { return ta > tb; }   // more tiles first
        return a.ratio_err < b.ratio_err;   // then closer ratio
    });

    const int use_col = best->col;
    const int use_row = best->row;
    LOG_INF("%s: %dx%d — selected %dx%d grid (%d tiles, ratio_err=%.3f)\n",
            __func__, img_size.width, img_size.height, use_col, use_row, use_col * use_row, best->ratio_err);
    const int target_w = use_col * tile_px;
    const int target_h = use_row * tile_px;

    // Resize to the tile canvas by stretching. With aspect-ratio-aware grid selection the chosen
    // grid's ratio is close to the image's ratio, so the residual stretch is small.
    // pad=false avoids black bars that confuse the model in tile inputs.
    clip_image_u8 resized;
    img_tool::resize(img, resized, {target_w, target_h},
                     hparams.image_resize_algo,
                     /* pad */ PAD_NONE,
                     hparams.image_pad_color);

    // Split into use_col × use_row tiles, packed row-major into the batch.
    clip_image_u8 tile_u8;

    for (int tr = 0; tr < use_row; tr++) {
        for (int tc = 0; tc < use_col; tc++) {
            const int src_x0 = tc * tile_px;
            const int src_y0 = tr * tile_px;
            img_tool::crop(resized, tile_u8, src_x0, src_y0, tile_px, tile_px);

            clip_image_f32 tile_f32;
            tile_f32.from_u8(tile_u8);
            tile_f32.normalize(hparams.image_mean, hparams.image_std);
            output.entries.push_back(std::move(tile_f32));
        }
    }

    // LLaVA-style global overview (thumbnail): a downscaled full image (tile_px x tile_px) prepended
    // as entries[0]. It carries the whole image uncut, so text split across a tile seam stays
    // readable; on DocVQA this recovers seam-truncated text for a sizeable ANLS gain. Encoded as a
    // separate 1x1 chunk in mtmd.cpp; the tiles are untouched (no resolution cost to them).
    {
        clip_image_u8 thumb_u8;
        img_tool::resize(img, thumb_u8, {tile_px, tile_px},
                         hparams.image_resize_algo,
                         /* pad */ PAD_NONE,
                         hparams.image_pad_color);
        clip_image_f32 thumb_f32;
        thumb_f32.from_u8(thumb_u8);
        thumb_f32.normalize(hparams.image_mean, hparams.image_std);
        output.entries.insert(output.entries.begin(), std::move(thumb_f32));
        output.overview_in_entries = true;
    }

    output.grid_x = use_col;
    output.grid_y = use_row;
    return output;
}
//
// mtmd_image_preprocessor_longest_edge
//

mtmd_image_preproc_out mtmd_image_preprocessor_longest_edge::preprocess(const clip_image_u8 & img) {
    GGML_ASSERT(hparams.image_longest_edge > 0);
    clip_image_u8 resized_image;
    const clip_image_size original_size = img.get_size();
    // the original pixtral model doesn't have n_merge
    const int cur_merge = hparams.n_merge == 0 ? 1 : hparams.n_merge;
    const clip_image_size target_size = img_tool::calc_size_preserved_ratio(
        original_size,
        {
            /* align_size   */ hparams.patch_size * cur_merge,
            /* min_pixels   */ std::max(0, hparams.image_min_pixels),
            /* max_pixels   */ std::max(0, hparams.image_max_pixels),
            /* longest_edge */ hparams.image_longest_edge,
        });
    img_tool::resize(img, resized_image, target_size,
                        hparams.image_resize_algo,
                        hparams.image_resize_pad,
                        hparams.image_pad_color);
    mtmd_image_preproc_out output;
    output.append(hparams, resized_image, true);
    return output;
}

//
// mtmd_image_preprocessor_minicpmv
//

mtmd_image_preprocessor_llava_uhd::slice_instructions mtmd_image_preprocessor_minicpmv::get_slice_instructions(const clip_image_size & original_size) {
    if (hparams.n_merge == 2) {
        const int   slice_size = hparams.image_size;
        const float ratio      = (float)original_size.width * original_size.height / (slice_size * slice_size);
        if (ratio <= 1.0f) {
            mtmd_image_preprocessor_llava_uhd::slice_instructions inst;
            const int patch_size = hparams.patch_size * hparams.n_merge;
            inst.overview_size = get_best_resize(original_size, slice_size, patch_size, true);
            inst.refined_size  = clip_image_size{0, 0};
            inst.grid_size     = clip_image_size{0, 0};
            return inst;
        }
    }
    return mtmd_image_preprocessor_llava_uhd::get_slice_instructions(original_size);
}

//
// mtmd_image_preprocessor_lfm2
//

mtmd_image_preproc_out mtmd_image_preprocessor_lfm2::preprocess(const clip_image_u8 & img) {
    auto const inst = get_slice_instructions(img.get_size());
    if (!inst.slices.empty()) {
        return mtmd_image_preprocessor_llava_uhd::preprocess(img);
    }

    // single tile: no thumbnail
    // note: not using output.overview here because it will emit <|img_thumbnail|> token, which we don't want in this case
    auto sliced = slice_image(img, inst);
    mtmd_image_preproc_out output;
    output.append(hparams, sliced.overview, true);
    return output;
}

bool mtmd_image_preprocessor_lfm2::should_tile(
        const clip_hparams & hparams,
        const clip_image_size & original_size) {
    const int align_size = hparams.patch_size * hparams.n_merge;

    const auto round_by_factor = [align_size](float x) {
        // see https://github.com/ggml-org/llama.cpp/pull/27057#discussion_r3796264887
        return static_cast<int>(std::nearbyint(static_cast<double>(x) / align_size)) * align_size;
    };

    const int h_bar = std::max(hparams.patch_size, round_by_factor(original_size.height));
    const int w_bar = std::max(hparams.patch_size, round_by_factor(original_size.width));

    return static_cast<double>(h_bar) * static_cast<double>(w_bar) >
           static_cast<double>(hparams.image_max_pixels) * max_pixels_tolerance;
}

mtmd_image_preprocessor_llava_uhd::slice_instructions mtmd_image_preprocessor_lfm2::get_slice_instructions(const clip_image_size & original_size) {
    mtmd_image_preprocessor_llava_uhd::slice_instructions inst;
    const int align_size = hparams.patch_size * hparams.n_merge;
    inst.overview_size = img_tool::calc_size_preserved_ratio(
                            original_size,
                            { align_size, hparams.image_min_pixels, hparams.image_max_pixels, 0 });

    const bool needs_tiling = should_tile(hparams, original_size);

    if (!needs_tiling) {
        inst.refined_size = clip_image_size{0, 0};
        inst.grid_size    = clip_image_size{0, 0};
        return inst;
    }

    const clip_image_size grid = get_grid_layout(original_size.height, original_size.width);

    inst.grid_size    = grid;
    inst.refined_size = clip_image_size{tile_size * grid.width, tile_size * grid.height};

    LOG_DBG("%s: original size: %d x %d, overview size: %d x %d, refined size: %d x %d, grid size: %d x %d\n",
            __func__,
            original_size.width, original_size.height,
            inst.overview_size.width, inst.overview_size.height,
            inst.refined_size.width, inst.refined_size.height,
            grid.width, grid.height);

    for (int row = 0; row < grid.height; row++) {
        for (int col = 0; col < grid.width; col++) {
            mtmd_image_preprocessor_llava_uhd::slice_coordinates slice;
            slice.x    = col * tile_size;
            slice.y    = row * tile_size;
            slice.size = clip_image_size{tile_size, tile_size};
            inst.slices.push_back(slice);
            LOG_DBG("%s: slice %d: x=%d, y=%d, size=%d x %d\n",
                    __func__, (int)inst.slices.size() - 1,
                    slice.x, slice.y, slice.size.width, slice.size.height);
        }
    }

    return inst;
}

clip_image_size mtmd_image_preprocessor_lfm2::find_closest_aspect_ratio(
        float aspect_ratio,
        const std::vector<clip_image_size> & target_ratios,
        int width, int height) {
    float best_ratio_diff = std::numeric_limits<float>::max();
    clip_image_size best_ratio = {1, 1};
    const float area = static_cast<float>(width * height);

    for (const auto & ratio : target_ratios) {
        const float target_aspect_ratio = static_cast<float>(ratio.width) / ratio.height;
        const float ratio_diff = std::abs(aspect_ratio - target_aspect_ratio);
        if (ratio_diff < best_ratio_diff) {
            best_ratio_diff = ratio_diff;
            best_ratio = ratio;
        } else if (ratio_diff == best_ratio_diff) {
            const float target_area = static_cast<float>(tile_size * tile_size * ratio.width * ratio.height);
            if (area > 0.5f * target_area) {
                best_ratio = ratio;
            }
        }
    }
    return best_ratio;
}

std::vector<clip_image_size> mtmd_image_preprocessor_lfm2::get_target_ratios() {
    std::vector<clip_image_size> ratios;
    for (int n = min_tiles; n <= max_tiles; n++) {
        for (int w = 1; w <= n; w++) {
            for (int h = 1; h <= n; h++) {
                if (w * h >= min_tiles && w * h <= max_tiles) {
                    bool found = false;
                    for (const auto & r : ratios) {
                        if (r.width == w && r.height == h) {
                            found = true;
                            break;
                        }
                    }
                    if (!found) {
                        ratios.push_back({w, h});
                    }
                }
            }
        }
    }
    std::sort(ratios.begin(), ratios.end(), [](const clip_image_size & a, const clip_image_size & b) {
        return a.width * a.height < b.width * b.height;
    });
    return ratios;
}

clip_image_size mtmd_image_preprocessor_lfm2::get_grid_layout(int height, int width) {
    const float aspect_ratio = static_cast<float>(width) / height;
    const auto ratios = get_target_ratios();
    return find_closest_aspect_ratio(aspect_ratio, ratios, width, height);
}

//
// mtmd_image_preprocessor_idefics3
//

// Sizing rule for the no-upscale variant, a direct port of DynamicResize._get_new_hw() with
// resize_to_max_side_len=False from the reference processor. `p` is the slice size
// (vit_img_size) and `m` the cap (max_img_size); the reference's min_side_len is
// FLASH_MIN_SIDE_LEN, which is the same 512 as vit_img_size, so it is not a separate knob.
//
// CITE: https://huggingface.co/qvac/VisionPsy-Nano-460M-Flash/blob/main/custom_transforms.py
//
// Note both branches return multiples of `p`, exactly like the upscaling path: the reference
// splitter hard-errors on anything else. So this changes HOW MANY slices an image becomes,
// never their size.
static clip_image_size calc_size_no_upscale(const clip_image_size & inp_size, int p, int m) {
    GGML_ASSERT(p > 0 && m > 0);
    if (inp_size.width <= 0 || inp_size.height <= 0) {
        return {0, 0};
    }
    // integer ceil, mirroring the reference's -(-a // b). Stays in int64_t all the way
    // through the multiply-back-by-p: an extreme aspect ratio (e.g. a very wide, very
    // thin image) can push that product past INT32_MAX, which would silently wrap
    // negative in 32-bit arithmetic and defeat the std::min/std::max clamps below.
    auto ceil_div = [](int64_t a, int64_t b) { return (a + b - 1) / b; };

    const int64_t long_side  = std::max(inp_size.width, inp_size.height);
    const int64_t short_side = std::min(inp_size.width, inp_size.height);
    const int64_t p64 = p;
    const int64_t m64 = m;

    int64_t target_long;
    int64_t target_short;
    if (short_side < p64) {
        // Short side below one slice: pin it to exactly p and give the long side however many
        // whole slices the aspect ratio asks for. This is the one case that DOES enlarge.
        const int64_t den = short_side * p64;
        target_long  = std::min(m64, ceil_div(long_side * p64, den) * p64);
        target_short = std::max(ceil_div(short_side * p64, den) * p64, p64);
    } else {
        target_long = std::min(m64, ceil_div(long_side, p64) * p64);
        // double, not float, and deliberately not exact integer arithmetic: the reference is
        // `math.ceil(short * scale / p)` on Python floats, and its rounding is part of the
        // rule the model was trained under. Where the true quotient lands on an integer the
        // three disagree by a whole slice row. Over every size pair up to 3000x3000, float32
        // differs from the reference in 171 cases and exact integers in 92; double in none.
        const double scale = (double) target_long / (double) long_side;
        target_short       = std::max((int64_t) std::ceil((double) short_side * scale / (double) p64) * p64, p64);
    }

    // Defensive clamp: both branches are designed to land in [p, m], but this guards the
    // narrowing cast below against ever producing a value outside that range.
    target_long  = std::clamp(target_long, p64, m64);
    target_short = std::clamp(target_short, p64, m64);

    return inp_size.width >= inp_size.height
        ? clip_image_size{(int) target_long, (int) target_short}
        : clip_image_size{(int) target_short, (int) target_long};
}

// The refined size has two steps:
// 1. Resize w/ aspect-ratio preserving such that the longer side is
//      the preprocessor longest size
// 2. Resize w/out preserving aspect ratio such that both sides are
//      multiples of image_size (always rounding up)
//
// CITE: https://github.com/huggingface/transformers/blob/main/src/transformers/models/idefics3/image_processing_idefics3.py#L737
//
// no_upscale replaces step 1 only; step 2 is shared, so both variants land on a
// multiple of image_size in each dimension.
mtmd_idefics3_sizing mtmd_calc_idefics3_sizing(const clip_image_size & original_size,
                                               int image_size,
                                               int image_longest_edge,
                                               bool no_upscale,
                                               int min_pixels,
                                               int max_pixels) {
    mtmd_idefics3_sizing out;
    out.refined_size = no_upscale
        ? calc_size_no_upscale(original_size, image_size, image_longest_edge)
        : img_tool::calc_size_preserved_ratio(
            original_size,
            { image_size, std::max(0, min_pixels), std::max(0, max_pixels), image_longest_edge });
    out.grid_size = clip_image_size{
        static_cast<int>(std::ceil(static_cast<float>(out.refined_size.width) / image_size)),
        static_cast<int>(std::ceil(static_cast<float>(out.refined_size.height) / image_size)),
    };
    out.n_slices = out.grid_size.width * out.grid_size.height;
    return out;
}

mtmd_image_preproc_out mtmd_image_preprocessor_idefics3::preprocess(const clip_image_u8 & img) {
    const clip_image_size original_size = img.get_size();
    const mtmd_idefics3_sizing sizing = mtmd_calc_idefics3_sizing(
        original_size, hparams.image_size, hparams.image_longest_edge, hparams.image_no_upscale,
        hparams.image_min_pixels, hparams.image_max_pixels);
    const clip_image_size refined_size = sizing.refined_size;
    // LOG_INF("%s: original size: %d x %d, refined size: %d x %d\n",
    //         __func__, original_size.width, original_size.height,
    //         refined_size.width, refined_size.height);

    mtmd_image_preprocessor_llava_uhd::slice_instructions instructions;
    instructions.refined_size = refined_size;
    instructions.grid_size = sizing.grid_size;
    // Square, in both variants: the reference resizes the global image to (p, p) regardless
    // of no_upscale (GlobalAndSplitImages in custom_transforms.py).
    instructions.overview_size = clip_image_size{hparams.image_size, hparams.image_size};
    for (int y = 0; y < refined_size.height; y += hparams.image_size) {
        for (int x = 0; x < refined_size.width; x += hparams.image_size) {
            // LOG_INF("%s: adding slice at x=%d, y=%d\n", __func__, x, y);
            instructions.slices.push_back(mtmd_image_preprocessor_llava_uhd::slice_coordinates{
                /* x    */x,
                /* y    */y,
                /* size */clip_image_size{
                    std::min(hparams.image_size, refined_size.width - x),
                    std::min(hparams.image_size, refined_size.height - y)
                }
            });
        }
    }
    auto sliced = slice_image(img, instructions);

    mtmd_image_preproc_out output;
    output.append_overview(hparams, sliced.overview, true);
    output.append(hparams, sliced.slices, true);
    output.grid_x = instructions.grid_size.width;
    output.grid_y = instructions.grid_size.height;
    return output;
}

//
// mtmd_image_preprocessor_internvl
//

mtmd_image_preproc_out mtmd_image_preprocessor_internvl::preprocess(const clip_image_u8 & img) {
    GGML_ASSERT(!hparams.image_res_candidates.empty());
    const clip_image_size original_size = img.get_size();
    auto const inst = get_slice_instructions(original_size);
    auto sliced = slice_image(img, inst);

    mtmd_image_preproc_out output;
    // InternVL: slices first, then overview
    output.append(hparams, sliced.slices, true);
    output.append_overview(hparams, sliced.overview, true);
    output.grid_x = inst.grid_size.width;
    output.grid_y = inst.grid_size.height;
    return output;
}

//
// mtmd_image_preprocessor_deepseekocr
//

std::vector<clip_image_size> mtmd_image_preprocessor_deepseekocr::get_target_ratios() const {
    std::vector<clip_image_size> ratios;
    for (int n = min_tiles; n <= max_tiles; n++) {
        for (int w = 1; w <= n; w++) {
            for (int h = 1; h <= n; h++) {
                if (w * h < min_tiles || w * h > max_tiles) {
                    continue;
                }
                bool found = false;
                for (const auto & r : ratios) {
                    if (r.width == w && r.height == h) {
                        found = true;
                        break;
                    }
                }
                if (!found) {
                    ratios.push_back({ w, h });
                }
            }
        }
    }
    std::sort(ratios.begin(), ratios.end(), [](const clip_image_size & a, const clip_image_size & b) {
        return a.width * a.height < b.width * b.height;
    });
    return ratios;
}

clip_image_size mtmd_image_preprocessor_deepseekocr::find_closest_aspect_ratio(
    float                                aspect_ratio,
    const std::vector<clip_image_size> & target_ratios,
    int                                  width,
    int                                  height) const {
    float           best_ratio_diff = std::numeric_limits<float>::max();
    clip_image_size best_ratio      = { 1, 1 };
    const float     area            = static_cast<float>(width * height);

    for (const auto & ratio : target_ratios) {
        const float target_aspect_ratio = static_cast<float>(ratio.width) / ratio.height;
        const float ratio_diff          = std::abs(aspect_ratio - target_aspect_ratio);
        if (ratio_diff < best_ratio_diff) {
            best_ratio_diff = ratio_diff;
            best_ratio      = ratio;
        } else if (ratio_diff == best_ratio_diff) {
            const float target_area = static_cast<float>(tile_size * tile_size * ratio.width * ratio.height);
            if (area > 0.5f * target_area) {
                best_ratio = ratio;
            }
        }
    }
    return best_ratio;
}

mtmd_image_preproc_out mtmd_image_preprocessor_deepseekocr::preprocess(const clip_image_u8 & img) {
    mtmd_image_preproc_out output;
    int grid_w = 0;
    int grid_h = 0;
    const auto img_size = img.get_size();

    // global view: aspect-preserving fit-and-pad to base_size
    clip_image_u8 padded;
    img_tool::resize(img, padded,
                    { base_size, base_size },
                    RESIZE_ALGO_BICUBIC_PILLOW,
                    PAD_NEAREST,
                    hparams.image_pad_color);
    output.append_overview(hparams, padded, true);
    output.overview.add_viewsep = true;

    // if this condition doesn't hold, the output is overview only, no tiles
    if (img_size.width > tile_size || img_size.height > tile_size) {
        const float           aspect_ratio  = static_cast<float>(img_size.width) / img_size.height;
        const auto            target_ratios = get_target_ratios();
        const clip_image_size grid =
            find_closest_aspect_ratio(aspect_ratio, target_ratios, img_size.width, img_size.height);
        grid_w = grid.width;
        grid_h = grid.height;

        clip_image_u8 refined;
        img_tool::resize(img, refined, { tile_size * grid_w, tile_size * grid_h }, RESIZE_ALGO_BICUBIC_PILLOW,
                         PAD_NONE);

        for (int row = 0; row < grid_h; row++) {
            if (fuse_row) {
                // concat all tiles in this row into a single image, along the H axis
                // output image size: w = tile_size, h = tile_size * grid_w
                // this is to ensure the whole row is always processed together
                clip_image_u8 row_img;
                row_img.set_size({tile_size, tile_size * grid_w}, false);
                for (int col = 0; col < grid_w; col++) {
                    for (int py = 0; py < tile_size; py++) {
                        for (int px = 0; px < tile_size; px++) {
                            row_img.set_pixel(px, col * tile_size + py,
                                              refined.get_pixel(col * tile_size + px, row * tile_size + py));
                        }
                    }
                }
                output.append(hparams, row_img, true);
            } else {
                for (int col = 0; col < grid_w; col++) {
                    clip_image_u8 tile;
                    img_tool::crop(refined, tile, col * tile_size, row * tile_size, tile_size, tile_size);
                    output.append(hparams, tile, true);
                }
            }
        }
        if (fuse_row) {
            grid_w = 1; // each fused row is one image; a single output column
        }
    }

    LOG_DBG("%s: grid size: %d x %d (%d tiles) + global view\n", __func__, grid_w, grid_h, grid_w * grid_h);
    LOG_DBG("%s: overview size: %d x %d\n", __func__, padded.get_size().width, padded.get_size().height);

    output.grid_x = grid_w;
    output.grid_y = grid_h;
    return output;
}

//
// mtmd_image_preprocessor_step3vl
//

void mtmd_image_preprocessor_step3vl::img_u8_resize_bilinear_to_f32(
        const clip_image_u8 & src,
        clip_image_f32 & dst,
        int target_width,
        int target_height,
        const float mean[3],
        const float std[3]) {
    const auto src_size = src.get_size();
    if (src_size.width == target_width && src_size.height == target_height) {
        dst.from_u8(src);
        dst.normalize(mean, std);
        return;
    }

    dst.set_size({target_width, target_height}, false, false);

    if (src.is_placeholder()) {
        // no-op for placeholder image, just set the size and return
        return;
    }

    const float scale_x = static_cast<float>(src_size.width)  / target_width;
    const float scale_y = static_cast<float>(src_size.height) / target_height;

    std::vector<float> local_buf((size_t) 3 * (size_t) target_width * (size_t) target_height);

    for (int y = 0; y < target_height; ++y) {
        const float src_y = (static_cast<float>(y) + 0.5f) * scale_y - 0.5f;
        const int y0_floor = static_cast<int>(std::floor(src_y));
        const int y0 = std::max(0, std::min(y0_floor,     src_size.height - 1));
        const int y1 = std::max(0, std::min(y0_floor + 1, src_size.height - 1));
        const float ly = src_y - y0_floor;

        for (int x = 0; x < target_width; ++x) {
            const float src_x = (static_cast<float>(x) + 0.5f) * scale_x - 0.5f;
            const int x0_floor = static_cast<int>(std::floor(src_x));
            const int x0 = std::max(0, std::min(x0_floor,     src_size.width - 1));
            const int x1 = std::max(0, std::min(x0_floor + 1, src_size.width - 1));
            const float lx = src_x - x0_floor;

            const auto p00 = src.get_pixel(x0, y0);
            const auto p01 = src.get_pixel(x1, y0);
            const auto p10 = src.get_pixel(x0, y1);
            const auto p11 = src.get_pixel(x1, y1);

            const size_t idx_dst = (size_t) 3 * ((size_t) y * (size_t) target_width + (size_t) x);
            for (int c = 0; c < 3; ++c) {
                const float v00 = (static_cast<float>(p00[c]) / 255.0f - mean[c]) / std[c];
                const float v01 = (static_cast<float>(p01[c]) / 255.0f - mean[c]) / std[c];
                const float v10 = (static_cast<float>(p10[c]) / 255.0f - mean[c]) / std[c];
                const float v11 = (static_cast<float>(p11[c]) / 255.0f - mean[c]) / std[c];

                const float top = v00 + (v01 - v00) * lx;
                const float bot = v10 + (v11 - v10) * lx;
                local_buf[idx_dst + c] = top + (bot - top) * ly;
            }
        }
    }
    dst.cpy_buf(local_buf);
}

int mtmd_image_preprocessor_step3vl::get_image_longest_edge(const clip_hparams & params) {
    return params.image_longest_edge > 0 ? params.image_longest_edge : default_image_longest_edge;
}

int mtmd_image_preprocessor_step3vl::determine_window_size(const clip_hparams & params, int longer, int shorter) {
    const int image_size = params.image_size;
    const int crop_size  = default_image_crop_size;
    const float aspect_ratio = static_cast<float>(longer) / shorter;

    if (longer <= image_size) {
        return aspect_ratio > small_aspect_ratio_limit ? shorter : 0;
    }

    return aspect_ratio > wide_aspect_ratio_limit ? std::min(shorter, crop_size) : crop_size;
}

int mtmd_image_preprocessor_step3vl::calc_crop_extent(int length, int window_size) {
    const float ratio = static_cast<float>(length) / window_size;
    if (ratio < 1.0f) {
        return length;
    }

    const float decimal = ratio - std::floor(ratio);
    const int rounded = decimal > crop_rounding_threshold
        ? static_cast<int>(std::floor(ratio)) + 1
        : static_cast<int>(std::floor(ratio));
    return window_size * rounded;
}

std::vector<int> mtmd_image_preprocessor_step3vl::calc_grid(int length, int window_size) {
    const int n = length <= window_size
        ? 1
        : static_cast<int>(std::ceil(static_cast<float>(length - window_size) / window_size + 1.0f));
    std::vector<int> starts(n);

    for (int i = 0; i < n; ++i) {
        starts[i] = window_size * i;
    }

    if (n > 1 && starts.back() + window_size > length) {
        starts.back() = length - window_size;
    }

    return starts;
}

clip_image_u8 mtmd_image_preprocessor_step3vl::prepare_image(const clip_image_u8 & img, const clip_hparams & params) {
    clip_image_u8 resized = img;
    const auto img_size = img.get_size();
    const float aspect_ratio = img_size.height > 0 ? static_cast<float>(img_size.width) / img_size.height : 1.0f;
    if (std::min(img_size.width, img_size.height) < 32 &&
        (aspect_ratio > wide_aspect_ratio_limit ||
         aspect_ratio < 1.0f / wide_aspect_ratio_limit)) {
        const int square_size = std::max(img_size.width, img_size.height);
        clip_image_u8 padded;
        padded.set_size({square_size, square_size}, false);
        img_tool::fill(padded, {0, 0, 0});
        img_tool::composite(padded, img, 0, 0);
        resized = std::move(padded);
    }

    const int max_image_size = get_image_longest_edge(params);
    const auto resized_size = resized.get_size();
    if (std::max(resized_size.width, resized_size.height) > max_image_size) {
        const float scale = static_cast<float>(max_image_size) / std::max(resized_size.width, resized_size.height);
        const clip_image_size new_size = {
            std::max(1, static_cast<int>(std::floor(resized_size.width  * scale))),
            std::max(1, static_cast<int>(std::floor(resized_size.height * scale))),
        };
        clip_image_u8 scaled;
        img_tool::resize(resized, scaled, new_size, RESIZE_ALGO_BILINEAR, PAD_NONE);
        resized = std::move(scaled);
    }

    return resized;
}

clip_image_u8 mtmd_image_preprocessor_step3vl::crop_with_black_padding(const clip_image_u8 & image, int x, int y, int w, int h) {
    clip_image_u8 dst;
    dst.set_size({w, h}, false);
    img_tool::fill(dst, {0, 0, 0});

    const auto img_size = image.get_size();
    const int src_x0 = std::max(0, x);
    const int src_y0 = std::max(0, y);
    const int src_x1 = std::min(img_size.width,  x + w);
    const int src_y1 = std::min(img_size.height, y + h);

    if (src_x0 >= src_x1 || src_y0 >= src_y1) {
        return dst;
    }

    const int dst_x0 = src_x0 - x;
    const int dst_y0 = src_y0 - y;

    for (int yy = 0; yy < src_y1 - src_y0; ++yy) {
        for (int xx = 0; xx < src_x1 - src_x0; ++xx) {
            dst.set_pixel(dst_x0 + xx, dst_y0 + yy, image.get_pixel(src_x0 + xx, src_y0 + yy));
        }
    }

    return dst;
}

mtmd_image_preprocessor_step3vl::slice_instructions mtmd_image_preprocessor_step3vl::build_slice_instructions(
        const clip_hparams & params,
        const clip_image_size & prepared_size) {
    slice_instructions instructions;
    instructions.overview_size = prepared_size;

    const int window_size = determine_window_size(
        params,
        std::max(prepared_size.width, prepared_size.height),
        std::min(prepared_size.width, prepared_size.height));
    if (window_size <= 0) {
        instructions.refined_size = clip_image_size{0, 0};
        instructions.grid_size    = clip_image_size{0, 0};
        return instructions;
    }

    const int crop_width  = calc_crop_extent(prepared_size.width,  window_size);
    const int crop_height = calc_crop_extent(prepared_size.height, window_size);
    instructions.refined_size = clip_image_size{crop_width, crop_height};

    const auto xs = calc_grid(crop_width,  window_size);
    const auto ys = calc_grid(crop_height, window_size);
    instructions.grid_size = clip_image_size{
        static_cast<int>(xs.size()),
        static_cast<int>(ys.size()),
    };

    for (int y : ys) {
        for (int x : xs) {
            instructions.slices.push_back(slice_coordinates{
                /* x    */ x,
                /* y    */ y,
                /* size */ clip_image_size{window_size, window_size},
            });
        }
    }

    return instructions;
}

mtmd_image_preproc_out mtmd_image_preprocessor_step3vl::preprocess(const clip_image_u8 & img) {
    clip_image_u8 prepared = prepare_image(img, hparams);
    const auto instructions = build_slice_instructions(hparams, prepared.get_size());

    mtmd_image_preproc_out output;
    // overview (normalized f32, already includes mean/std)
    img_u8_resize_bilinear_to_f32(
        prepared,
        output.overview,
        hparams.image_size,
        hparams.image_size,
        hparams.image_mean,
        hparams.image_std);

    if (instructions.slices.empty()) {
        output.grid_x = 0;
        output.grid_y = 0;
        return output;
    }

    clip_image_u8 img_for_crop = prepared;
    const auto prepared_size = prepared.get_size();
    if (instructions.refined_size.width != prepared_size.width || instructions.refined_size.height != prepared_size.height) {
        clip_image_u8 refined;
        img_tool::resize(prepared, refined, instructions.refined_size, RESIZE_ALGO_BILINEAR, PAD_NONE);
        img_for_crop = std::move(refined);
    }

    const int crop_size = default_image_crop_size;
    for (const auto & slice : instructions.slices) {
        // If the requested patch extends past the source image, pad the out-of-bounds area with black.
        clip_image_u8 patch = crop_with_black_padding(img_for_crop, slice.x, slice.y, slice.size.width, slice.size.height);

        clip_image_f32 patch_f32;
        img_u8_resize_bilinear_to_f32(
            patch,
            patch_f32,
            crop_size,
            crop_size,
            hparams.image_mean,
            hparams.image_std);
        output.append(hparams, patch_f32, false);
    }

    output.grid_x = instructions.grid_size.width;
    output.grid_y = instructions.grid_size.height;

    return output;
}

//
// mtmd_image_preprocessor_youtuvl
//

mtmd_image_preproc_out mtmd_image_preprocessor_youtuvl::preprocess(const clip_image_u8 & img) {
    const int patch_size = hparams.patch_size;   // typically 16
    const int merge_size = hparams.n_merge;      // typically 2
    const int align_size = patch_size * merge_size;  // 32

    const int max_num_patches = hparams.image_max_pixels > 0 ?
        hparams.image_max_pixels / (patch_size * patch_size) : 256;

    // Linear search for optimal scale to fit within max_num_patches
    const auto img_size = img.get_size();
    float scale = 1.0f;
    int target_height = img_size.height;
    int target_width  = img_size.width;

    auto get_scaled_image_size = [align_size](float scale, int size) -> int {
        float scaled_size = size * scale;
        // Round up to nearest multiple of align_size
        int aligned = static_cast<int>(std::ceil(scaled_size / align_size)) * align_size;
        // Ensure at least one patch
        return std::max(align_size, aligned);
    };

    // Linear search with 0.02 step size
    while (scale > 0.0f) {
        target_height = get_scaled_image_size(scale, img_size.height);
        target_width  = get_scaled_image_size(scale, img_size.width);

        int num_patches_h = target_height / patch_size;
        int num_patches_w = target_width / patch_size;
        int num_patches = num_patches_h * num_patches_w;

        if (num_patches > max_num_patches) {
            scale -= 0.02f;
        } else {
            break;
        }
    }

    clip_image_size new_size = {target_width, target_height};

    // Resize the image
    clip_image_u8 resized;
    img_tool::resize(img, resized, new_size, hparams.image_resize_algo, hparams.image_resize_pad);

    mtmd_image_preproc_out output;
    output.append(hparams, resized, true);
    return output;
}

mtmd_image_preproc_out mtmd_image_preprocessor_granite::preprocess(const clip_image_u8 & img) {
    GGML_ASSERT(!hparams.image_res_candidates.empty());

    const clip_image_size orig_size = img.get_size();
    const int             tile_size = hparams.image_size;
    GGML_ASSERT(tile_size > 0);

    // llava-next always encodes an overview plus a grid of tiles, even for small images
    const clip_image_size refined_size = select_best_resolution(orig_size, hparams.image_res_candidates);
    const int             grid_x       = refined_size.width  / tile_size;
    const int             grid_y       = refined_size.height / tile_size;

    // the tiles are stacked on the Y axis, a big grid overflows the stacked image height
    GGML_ASSERT(grid_x >= 0 && grid_x <= 1024 && grid_y >= 0 && grid_y <= 1024);

    clip_image_u8 overview;
    img_tool::resize(img, overview, {tile_size, tile_size}, hparams.image_resize_algo_ov,
                        hparams.image_pad_ov, hparams.image_pad_color_ov);

    clip_image_u8 refined;
    img_tool::resize(img, refined, refined_size, hparams.image_resize_algo_rf,
                        hparams.image_pad_rf, hparams.image_pad_color_rf);

    // stack the overview and the tiles on the Y axis, so the whole grid goes through one graph
    clip_image_u8 stacked;
    stacked.set_size({tile_size, tile_size * (1 + grid_x * grid_y)}, false);
    auto copy_tile = [&](const clip_image_u8 & src, int src_x, int src_y, int dst_idx) {
        for (int py = 0; py < tile_size; py++) {
            for (int px = 0; px < tile_size; px++) {
                stacked.set_pixel(px, dst_idx * tile_size + py, src.get_pixel(src_x + px, src_y + py));
            }
        }
    };
    copy_tile(overview, 0, 0, 0);
    for (int ty = 0; ty < grid_y; ty++) {
        for (int tx = 0; tx < grid_x; tx++) {
            copy_tile(refined, tx * tile_size, ty * tile_size, 1 + ty * grid_x + tx);
        }
    }

    LOG_DBG("%s: grid size: %d x %d (%d tiles) + overview\n", __func__, grid_x, grid_y, grid_x * grid_y);

    mtmd_image_preproc_out output;
    output.append(hparams, stacked, true);
    auto & entry = output.entries.back();
    entry.anyres.grid_x  = grid_x;
    entry.anyres.grid_y  = grid_y;
    entry.anyres.orig_nx = orig_size.width;
    entry.anyres.orig_ny = orig_size.height;
    return output;
}

//
// mtmd_image_preprocessor_muse_glimmer
//

// Replicates transformers' get_aspect_ratio_preserving_size
static clip_image_size muse_glimmer_grid_size(int img_w, int img_h, int patch_hw, int max_tokens) {
    double i_nph = (double) img_h / patch_hw;
    double i_npw = (double) img_w / patch_hw;
    const double ratio = i_nph > 0.0 ? i_npw / i_nph : 1.0;
    if (i_nph * i_npw > (double) max_tokens) {
        i_nph = std::sqrt((double) max_tokens / ratio);
        i_npw = i_nph * ratio;
    }
    const int hs[2] = { (int) std::floor(i_nph), (int) std::ceil(i_nph) };
    const int ws[2] = { (int) std::floor(i_npw), (int) std::ceil(i_npw) };
    const double target_ar = (double) img_h / (double) img_w;
    int    best_nph = -1;
    int    best_npw = -1;
    double best_d   = 0.0;
    for (int a = 0; a < 2; ++a) {
        for (int b = 0; b < 2; ++b) {
            const int nph = hs[a];
            const int npw = ws[b];
            if (nph < 1 || npw < 1 || nph * npw > max_tokens) {
                continue;
            }
            const double d = std::fabs((double) nph / (double) npw - target_ar);
            const int n_tokens      = nph * npw;
            const int best_n_tokens = best_nph * best_npw;
            if (best_nph < 0 || d < best_d || (d == best_d && n_tokens > best_n_tokens)) {
                best_nph = nph;
                best_npw = npw;
                best_d   = d;
            }
        }
    }
    if (best_nph < 0) { // no candidate fit under the cap: round and clamp
        best_nph = std::max(1, (int) std::lround(i_nph));
        best_npw = std::max(1, (int) std::lround(i_npw));
    }
    return clip_image_size{ best_npw * patch_hw, best_nph * patch_hw };
}

mtmd_image_preproc_out mtmd_image_preprocessor_muse_glimmer::preprocess(const clip_image_u8 & img) {
    const int patch_hw   = hparams.patch_size * hparams.n_merge;
    const int patch_area = hparams.patch_size * hparams.patch_size * hparams.n_merge * hparams.n_merge;
    GGML_ASSERT(patch_area > 0 && hparams.image_max_pixels > 0);
    const int max_tokens = hparams.image_max_pixels / patch_area;

    const clip_image_size original_size = img.get_size();
    const clip_image_size target_size   = muse_glimmer_grid_size(
        original_size.width, original_size.height, patch_hw, max_tokens);

    // PIL resizes directly to (target_w, target_h) -- a stretch, no padding.
    clip_image_u8 resized_image;
    img_tool::resize(img, resized_image, target_size, hparams.image_resize_algo, PAD_NONE);

    mtmd_image_preproc_out output;
    output.append(hparams, resized_image, true);
    return output;
}
