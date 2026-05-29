#include <algorithm>
#include <array>
#include <atomic>
#include <cctype>
#include <cerrno>
#include <cmath>
#include <csignal>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <ctime>
#include <condition_variable>
#include <memory>
#include <mutex>
#include <opencv2/imgcodecs.hpp>
#include <opencv2/imgproc.hpp>
#include <pthread.h>
#include <string>
#include <sys/mman.h>
#include <unistd.h>
#include <vector>

#include <nncase/runtime/runtime_op_utility.h>

#include "person_fifo_writer.h"
#include "yolo_detect.h"

extern "C" {
#include "k_sys_comm.h"
#include "k_type.h"
#include "k_vb_comm.h"
#include "k_vicap_comm.h"
#include "k_video_comm.h"
#include "mpi_sys_api.h"
#include "mpi_vb_api.h"
#include "mpi_vicap_api.h"
}

namespace {

/** Mosaic width × height (OpenCV: cols × rows). vstack 348×213 per camera (y=0,213,426). */
constexpr int kMosaicCols = 348;
constexpr int kMosaicRows = 639;
/** Band `cv::resize`: INTER_NEAREST is fastest; INTER_LINEAR if quality matters more than preprocess CPU. */
constexpr int kMosaicBandResizeInterp = cv::INTER_NEAREST;
constexpr int kCamCount = 3;
/** Coordinate remap: detections are mapped back to VICAP chn1 plane (ISP scaler out from sensor). */
constexpr int kSrcW = 640;
constexpr int kSrcH = 360;

/** NV12 @ chn1: ISP scales 1920×1080 sensor capture → 640×360 before NV12 convert and mosaic bands. */
constexpr int kIspChn1Height = 360;
constexpr int kIspChn1Width = 640;
constexpr int kIspInputWidth = 1920;
constexpr int kIspInputHeight = 1080;
constexpr int kVicapInputBufNum = 4;
constexpr int kVicapOutputBufNum = 5;
constexpr int kMosaicPipelineSlots = 2;

/** VICAP chn1 NV12 buffer size (matches vicap_init chn_attr.buffer_size). */
constexpr size_t kChn1Nv12BufBytes =
    VICAP_ALIGN_UP(static_cast<size_t>(kIspChn1Height * kIspChn1Width * 3 / 2), VICAP_ALIGN_1K);

/** Rolling window size for mosaic-inference FPS (avoids printf every frame). */
constexpr int kFpsYoloWindowFrames = 60;

constexpr k_vicap_mirror kVicapMirrorAll = VICAP_MIRROR_BOTH;

std::atomic<bool> g_running{true};
std::mutex g_print_mtx;

/** Three stacked strips (cam0 top → cam2 bottom); vstack heights 213+213+213 @ width 348. */
constexpr std::array<int, kCamCount> kMosaicBandH{{213, 213, 213}};
constexpr std::array<int, kCamCount> kMosaicBandY{{0, 213, 426}};
static_assert(kMosaicBandH[0] + kMosaicBandH[1] + kMosaicBandH[2] == kMosaicRows, "mosaic bands must fill canvas");

/** Letterbox fit of kSrcW×kSrcH inside each kMosaicCols × strip_h band (uniform scale; precomputed). */
constexpr std::array<int, kCamCount> kMosaicBandInnerW{{348, 348, 348}};
constexpr std::array<int, kCamCount> kMosaicBandInnerH{{195, 195, 195}};
constexpr std::array<int, kCamCount> kMosaicBandXPad{{0, 0, 0}};
constexpr std::array<int, kCamCount> kMosaicBandYPadInBand{{9, 9, 9}};
/** Mosaic px per chn1 px after letterbox (inner_h / kSrcH; equals inner_w / kSrcW within FP noise). */
constexpr std::array<float, kCamCount> kMosaicBandScale{
    {195.F / 360.F, 195.F / 360.F, 195.F / 360.F}};

struct MosaicLayout {
    std::array<int, kCamCount> band_y{};
    std::array<int, kCamCount> band_h{};
    std::array<float, kCamCount> scale{};
    std::array<int, kCamCount> x_pad{};
    std::array<int, kCamCount> y_pad_in_band{};
};

struct Nv12MmapGrab {
    k_video_frame_info dump_info{};
    k_vicap_dev dev{};
    void *mmap_y = nullptr;
    size_t mmap_y_bytes = 0;
    void *mmap_uv = nullptr;
    size_t mmap_uv_bytes = 0;
    void *mmap_packed = nullptr;
    size_t mmap_packed_bytes = 0;
    const uint8_t *y_plane = nullptr;
    const uint8_t *uv_plane = nullptr;
    k_u32 y_stride = 0;
    k_u32 uv_stride = 0;
    int width = 0;
    int height = 0;
    k_pixel_format pixel_format = PIXEL_FORMAT_BUTT;
    bool ok = false;
};

struct Nv12FrameLayout {
    k_u32 y_stride = 0;
    k_u32 uv_stride = 0;
    size_t y_bytes = 0;
    size_t uv_bytes = 0;
    size_t buf_bytes = 0;
    size_t uv_offset = 0;
};

/** Derive Y/UV byte layout from VICAP frame metadata (same rules as triple_cam_yolo_infer). */
static bool nv12_frame_layout(const k_video_frame &frame, size_t width, size_t height, Nv12FrameLayout *out)
{
    if (out == nullptr || width == 0U || height == 0U) {
        return false;
    }

    const k_u32 y_stride = frame.stride[0] >= width ? frame.stride[0] : static_cast<k_u32>(width);
    k_u32 uv_stride = y_stride;
    if (frame.stride[1] >= width) {
        uv_stride = frame.stride[1];
    }

    out->y_stride = y_stride;
    out->uv_stride = uv_stride;
    out->y_bytes = static_cast<size_t>(y_stride) * height;
    out->uv_bytes = static_cast<size_t>(uv_stride) * height / 2U;
    out->buf_bytes = kChn1Nv12BufBytes;
    out->uv_offset = out->y_bytes;
    return true;
}

static bool frame_uses_nv21_chroma(k_pixel_format fmt)
{
    return fmt == PIXEL_FORMAT_YVU_SEMIPLANAR_420;
}

static bool vicap_attach_nv12_mmap(k_vicap_dev dev, const k_video_frame_info &dump, Nv12MmapGrab *g)
{
    std::memset(g, 0, sizeof(*g));
    g->dump_info = dump;
    g->dev = dev;

    const k_video_frame &frame = dump.v_frame;
    const size_t width = frame.width > 0 ? frame.width : kIspChn1Width;
    const size_t height = frame.height > 0 ? frame.height : kIspChn1Height;
    if (width != kIspChn1Width || height != kIspChn1Height) {
        std::printf("unexpected frame size %ux%u format=%d (expected chn1 %dx%d)\n", frame.width, frame.height,
            frame.pixel_format, kIspChn1Width, kIspChn1Height);
        return false;
    }

    Nv12FrameLayout layout{};
    if (!nv12_frame_layout(frame, width, height, &layout)) {
        return false;
    }

    static std::array<bool, kCamCount> printed_meta{{false, false, false}};
    const int cam_idx = static_cast<int>(dev);
    if (cam_idx >= 0 && cam_idx < kCamCount && !printed_meta[static_cast<size_t>(cam_idx)]) {
        printed_meta[static_cast<size_t>(cam_idx)] = true;
        std::printf(
            "dev=%d fmt=%d dual_uv=%d y_phys=%llx uv_phys=%llx "
            "stride0=%u stride1=%u y_stride=%u uv_stride=%u width=%u height=%u\n",
            dev,
            frame.pixel_format,
            frame.phys_addr[1] != 0U ? 1 : 0,
            static_cast<unsigned long long>(frame.phys_addr[0]),
            static_cast<unsigned long long>(frame.phys_addr[1]),
            frame.stride[0],
            frame.stride[1],
            layout.y_stride,
            layout.uv_stride,
            frame.width,
            frame.height);
    }

    g->width = static_cast<int>(width);
    g->height = static_cast<int>(height);
    g->y_stride = layout.y_stride;
    g->uv_stride = layout.uv_stride;
    g->pixel_format = frame.pixel_format;

    /* Trust BSP phys_addr[]: separate Y/UV mmap when uv plane address is provided. */
    if (frame.phys_addr[1] != 0U) {
        void *y_buf = kd_mpi_sys_mmap(frame.phys_addr[0], layout.y_bytes);
        if (y_buf == nullptr || y_buf == MAP_FAILED) {
            std::printf("mmap Y plane failed dev=%d phys=0x%llx size=%zu errno=%d\n",
                dev, static_cast<unsigned long long>(frame.phys_addr[0]), layout.y_bytes, errno);
            return false;
        }
        void *uv_buf = kd_mpi_sys_mmap(frame.phys_addr[1], layout.uv_bytes);
        if (uv_buf == nullptr || uv_buf == MAP_FAILED) {
            kd_mpi_sys_munmap(y_buf, layout.y_bytes);
            std::printf("mmap UV plane failed dev=%d phys=0x%llx size=%zu errno=%d\n",
                dev, static_cast<unsigned long long>(frame.phys_addr[1]), layout.uv_bytes, errno);
            return false;
        }
        g->mmap_y = y_buf;
        g->mmap_y_bytes = layout.y_bytes;
        g->mmap_uv = uv_buf;
        g->mmap_uv_bytes = layout.uv_bytes;
        g->y_plane = static_cast<const uint8_t *>(y_buf);
        g->uv_plane = static_cast<const uint8_t *>(uv_buf);
        g->ok = true;
        return true;
    }

    void *buf = kd_mpi_sys_mmap(frame.phys_addr[0], layout.buf_bytes);
    if (buf == nullptr || buf == MAP_FAILED) {
        std::printf("mmap packed NV12 failed dev=%d phys=0x%llx size=%zu errno=%d\n",
            dev, static_cast<unsigned long long>(frame.phys_addr[0]), layout.buf_bytes, errno);
        return false;
    }
    g->mmap_packed = buf;
    g->mmap_packed_bytes = layout.buf_bytes;
    g->y_plane = static_cast<const uint8_t *>(buf);
    g->uv_plane = g->y_plane + static_cast<ptrdiff_t>(layout.uv_offset);
    g->ok = true;
    return true;
}

static bool nv12_grab_release(Nv12MmapGrab *g)
{
    if (g == nullptr || !g->ok) {
        return true;
    }
    if (g->mmap_packed != nullptr && g->mmap_packed_bytes > 0U) {
        kd_mpi_sys_munmap(g->mmap_packed, g->mmap_packed_bytes);
        g->mmap_packed = nullptr;
        g->mmap_packed_bytes = 0;
    } else {
        if (g->mmap_uv != nullptr && g->mmap_uv_bytes > 0U) {
            kd_mpi_sys_munmap(g->mmap_uv, g->mmap_uv_bytes);
            g->mmap_uv = nullptr;
            g->mmap_uv_bytes = 0;
        }
        if (g->mmap_y != nullptr && g->mmap_y_bytes > 0U) {
            kd_mpi_sys_munmap(g->mmap_y, g->mmap_y_bytes);
            g->mmap_y = nullptr;
            g->mmap_y_bytes = 0;
        }
    }
    g->y_plane = nullptr;
    g->uv_plane = nullptr;
    const k_s32 ret = kd_mpi_vicap_dump_release(g->dev, VICAP_CHN_ID_1, &g->dump_info);
    g->ok = false;
    return ret == K_SUCCESS;
}

static bool yuv_grab_to_rgb_two_plane(const Nv12MmapGrab &g, cv::Mat *out_rgb, bool nv21)
{
    if (!g.ok || out_rgb == nullptr || g.y_plane == nullptr || g.uv_plane == nullptr) {
        return false;
    }
    const int w = g.width;
    const int h = g.height;
    const cv::Mat y_mat(h, w, CV_8UC1, const_cast<uint8_t *>(g.y_plane), static_cast<size_t>(g.y_stride));
    const cv::Mat uv_mat(h / 2, w / 2, CV_8UC2, const_cast<uint8_t *>(g.uv_plane), static_cast<size_t>(g.uv_stride));
    const int cvt = nv21 ? cv::COLOR_YUV2RGB_NV21 : cv::COLOR_YUV2RGB_NV12;
    cv::cvtColorTwoPlane(y_mat, uv_mat, *out_rgb, cvt);
    return true;
}

static bool nv12_grab_to_bgr(const Nv12MmapGrab &g, cv::Mat *out_bgr)
{
    cv::Mat rgb;
    if (!yuv_grab_to_rgb_two_plane(g, &rgb, false)) {
        return false;
    }
    cv::cvtColor(rgb, *out_bgr, cv::COLOR_RGB2BGR);
    return true;
}

static bool nv12_grab_to_rgb(const Nv12MmapGrab &g, cv::Mat *out_rgb)
{
    return yuv_grab_to_rgb_two_plane(g, out_rgb, false);
}

static bool nv21_grab_to_rgb(const Nv12MmapGrab &g, cv::Mat *out_rgb)
{
    return yuv_grab_to_rgb_two_plane(g, out_rgb, true);
}

/** Convert grab to RGB (NV12/NV21 from frame pixel_format). */
static bool yuv_grab_to_rgb(const Nv12MmapGrab &g, cv::Mat *out_rgb)
{
    if (frame_uses_nv21_chroma(g.pixel_format)) {
        return nv21_grab_to_rgb(g, out_rgb);
    }
    return nv12_grab_to_rgb(g, out_rgb);
}

/** NV12→RGB per camera stripe (640×360 @ chn1); stitched mosaic is RGB for YoloDetect::pre_process (no extra BGR→RGB).
 *  @param cam_rgb_reuse optional persistent 640×360×3 buffer (created/resized here once). */
void build_mosaic_cv_resize(const std::array<Nv12MmapGrab, kCamCount> &grabs, cv::Mat *mosaic_rgb, const MosaicLayout &layout,
    cv::Mat *cam_rgb_reuse = nullptr)
{
    if (mosaic_rgb->empty() || mosaic_rgb->rows != kMosaicRows || mosaic_rgb->cols != kMosaicCols
        || mosaic_rgb->type() != CV_8UC3) {
        mosaic_rgb->create(kMosaicRows, kMosaicCols, CV_8UC3);
    }
    mosaic_rgb->setTo(cv::Scalar(114, 114, 114));

    cv::Mat cam_rgb_stack;
    cv::Mat *const cam_rgb = cam_rgb_reuse != nullptr ? cam_rgb_reuse : &cam_rgb_stack;
    if (cam_rgb->empty() || cam_rgb->rows != kIspChn1Height || cam_rgb->cols != kIspChn1Width
        || cam_rgb->type() != CV_8UC3) {
        cam_rgb->create(kIspChn1Height, kIspChn1Width, CV_8UC3);
    }
    for (int i = 0; i < kCamCount; ++i) {
        const Nv12MmapGrab &g = grabs[static_cast<size_t>(i)];

        if (!yuv_grab_to_rgb(g, cam_rgb)) {
            continue;
        }

        const int strip_y = layout.band_y[static_cast<size_t>(i)];
        const int xp = layout.x_pad[static_cast<size_t>(i)];
        const int ypb = layout.y_pad_in_band[static_cast<size_t>(i)];
        const int inner_w = kMosaicBandInnerW[static_cast<size_t>(i)];
        const int inner_h = kMosaicBandInnerH[static_cast<size_t>(i)];
        cv::resize(*cam_rgb, (*mosaic_rgb)(cv::Rect(xp, strip_y + ypb, inner_w, inner_h)),
            cv::Size(inner_w, inner_h), 0., 0., kMosaicBandResizeInterp);
    }
}

static bool nv12_grab_is_ai2d_packed_nv12(const Nv12MmapGrab &g)
{
    return g.ok && g.mmap_packed != nullptr && g.width == kIspChn1Width && g.height == kIspChn1Height && g.y_stride > 0;
}

/** Three separate AI2D invocations (one per camera band): packed NV12 + phys → resized RGB planar strip,
 *  then stitched into the KPU input tensor rows. nncase exposes one src per `invoke`; this matches the
 *  Kendryte NV12 path (YUV420_NV12 → NCHW) using `runtime_tensor` physical_address wrapping. */
struct MosaicNv12Ai2dMosaic {
    typecode_t out_tc_ = typecode_t::dt_float32;
    k_u32 nv12_stride_ = 0;
    bool built_ = false;
    std::array<std::unique_ptr<ai2d_builder>, kCamCount> builders_{};
    std::array<runtime_tensor, kCamCount> band_out_{};

    void reset()
    {
        for (auto &b : builders_) {
            b.reset();
        }
        for (auto &t : band_out_) {
            t.reset();
        }
        built_ = false;
        nv12_stride_ = 0;
    }

    bool ensure_built(typecode_t out_tc, char *err_buf, size_t err_len)
    {
        out_tc_ = out_tc;
        if (built_) {
            return true;
        }
        if (err_buf != nullptr && err_len > 0) {
            err_buf[0] = '\0';
        }
        if (out_tc_ != typecode_t::dt_float32 && out_tc_ != typecode_t::dt_uint8) {
            if (err_buf != nullptr && err_len > 0) {
                std::snprintf(err_buf, err_len, "MosaicNv12Ai2dMosaic: kmodel input must be f32 or u8 (got type %u)",
                    static_cast<unsigned>(out_tc_));
            }
            return false;
        }

        const k_u32 y_stride = static_cast<k_u32>(kIspChn1Width);
        nv12_stride_ = y_stride;
        const size_t nv12_tensor_h = static_cast<size_t>(kIspChn1Height + kIspChn1Height / 2);

        ai2d_shift_param_t shift_param{false, 0};
        ai2d_affine_param_t affine_param{false, ai2d_interp_method::cv2_bilinear, 0, 0, 127, 1,
            {0.5F, 0.1F, 0.0F, 0.1F, 0.5F, 0.0F}};
        ai2d_resize_param_t resize_param{true, ai2d_interp_method::tf_bilinear, ai2d_interp_mode::half_pixel};

        for (int i = 0; i < kCamCount; ++i) {
            const int strip_h = kMosaicBandH[static_cast<size_t>(i)];
            const int xp = kMosaicBandXPad[static_cast<size_t>(i)];
            const int inner_w = kMosaicBandInnerW[static_cast<size_t>(i)];
            const int pad_l = xp;
            const int pad_r = kMosaicCols - xp - inner_w;
            if (pad_l < 0 || pad_r < 0 || strip_h <= 0 || inner_w <= 0) {
                if (err_buf != nullptr && err_len > 0) {
                    std::snprintf(err_buf, err_len, "MosaicNv12Ai2dMosaic: invalid band %d geometry", i);
                }
                reset();
                return false;
            }

            dims_t in_shape{1, 1, nv12_tensor_h, static_cast<size_t>(nv12_stride_)};
            dims_t out_shape{1, 3, static_cast<size_t>(strip_h), static_cast<size_t>(kMosaicCols)};

            ai2d_datatype_t ai2d_dtype{ai2d_format::YUV420_NV12, ai2d_format::NCHW_FMT, typecode_t::dt_uint8, out_tc_};
            ai2d_crop_param_t crop_param{false, 0, 0, 0, 0};
            ai2d_pad_param_t pad_param{true, {{0, 0}, {0, 0}, {0, 0}, {pad_l, pad_r}}, ai2d_pad_mode::constant,
                {114, 114, 114}};

            builders_[static_cast<size_t>(i)].reset(
                new ai2d_builder(in_shape, out_shape, ai2d_dtype, crop_param, shift_param, pad_param, resize_param,
                    affine_param));
            builders_[static_cast<size_t>(i)]->build_schedule().expect("ai2d build_schedule failed (mosaic band)");
            band_out_[static_cast<size_t>(i)] =
                hrt::create(out_tc_, out_shape, hrt::pool_shared).expect("create mosaic band tensor failed");
        }
        built_ = true;
        return true;
    }

    static void stitch_band_nchw_uint8(uint8_t *dst_chw, int mosaic_h, int mosaic_w, int y0, int band_h, int band_w,
        const uint8_t *src_chw)
    {
        const size_t plane = static_cast<size_t>(mosaic_h) * static_cast<size_t>(mosaic_w);
        for (int c = 0; c < 3; ++c) {
            uint8_t *const dst_plane = dst_chw + static_cast<size_t>(c) * plane;
            const uint8_t *const src_plane = src_chw + static_cast<size_t>(c) * static_cast<size_t>(band_h) * static_cast<size_t>(band_w);
            for (int r = 0; r < band_h; ++r) {
                std::memcpy(dst_plane + static_cast<size_t>(y0 + r) * static_cast<size_t>(mosaic_w),
                    src_plane + static_cast<size_t>(r) * static_cast<size_t>(band_w), static_cast<size_t>(band_w));
            }
        }
    }

    static void stitch_band_nchw_float(float *dst_chw, int mosaic_h, int mosaic_w, int y0, int band_h, int band_w,
        const float *src_chw)
    {
        const size_t plane = static_cast<size_t>(mosaic_h) * static_cast<size_t>(mosaic_w);
        for (int c = 0; c < 3; ++c) {
            float *const dst_plane = dst_chw + static_cast<size_t>(c) * plane;
            const float *const src_plane = src_chw + static_cast<size_t>(c) * static_cast<size_t>(band_h) * static_cast<size_t>(band_w);
            for (int r = 0; r < band_h; ++r) {
                std::memcpy(dst_plane + static_cast<size_t>(y0 + r) * static_cast<size_t>(mosaic_w),
                    src_plane + static_cast<size_t>(r) * static_cast<size_t>(band_w),
                    static_cast<size_t>(band_w) * sizeof(float));
            }
        }
    }

    bool fill_kmodel_input(runtime_tensor &kmodel_in, const std::array<Nv12MmapGrab, kCamCount> &grabs,
        const MosaicLayout &layout, char *err_buf, size_t err_len)
    {
        if (err_buf != nullptr && err_len > 0) {
            err_buf[0] = '\0';
        }
        for (int i = 0; i < kCamCount; ++i) {
            if (!nv12_grab_is_ai2d_packed_nv12(grabs[static_cast<size_t>(i)])) {
                if (err_buf != nullptr && err_len > 0) {
                    std::snprintf(err_buf, err_len,
                        "cam%d: need packed NV12 mmap (phys_addr[1]==0) and stride=w for AI2D mosaic", i);
                }
                return false;
            }
            if (static_cast<k_u32>(grabs[static_cast<size_t>(i)].y_stride) != nv12_stride_) {
                if (err_buf != nullptr && err_len > 0) {
                    std::snprintf(err_buf, err_len, "cam%d y_stride=%u != built stride %u (rebuild not implemented)", i,
                        static_cast<unsigned>(grabs[static_cast<size_t>(i)].y_stride), static_cast<unsigned>(nv12_stride_));
                }
                return false;
            }
        }

        auto dst_wrap = kmodel_in.impl()->to_host().unwrap()->buffer().as_host().unwrap()
                            .map(map_access_::map_write)
                            .unwrap()
                            .buffer();
        uint8_t *const dst_u8 = reinterpret_cast<uint8_t *>(dst_wrap.data());
        float *const dst_f32 = reinterpret_cast<float *>(dst_wrap.data());

        for (int i = 0; i < kCamCount; ++i) {
            const Nv12MmapGrab &g = grabs[static_cast<size_t>(i)];
            const size_t packed_bytes = g.mmap_packed_bytes;
            dims_t in_shape{1, 1, static_cast<size_t>(g.height + g.height / 2), static_cast<size_t>(g.y_stride)};
            const gsl::span<gsl::byte> nv12_span(reinterpret_cast<gsl::byte *>(g.mmap_packed), packed_bytes);
            auto in_tensor = hrt::create(typecode_t::dt_uint8, in_shape, nv12_span, false, hrt::pool_shared,
                static_cast<uintptr_t>(g.dump_info.v_frame.phys_addr[0]))
                                  .expect("wrap nv12 phys for ai2d failed");
            hrt::sync(in_tensor, sync_op_t::sync_write_back, true).expect("nv12 tensor sync failed");

            builders_[static_cast<size_t>(i)]->invoke(in_tensor, band_out_[static_cast<size_t>(i)])
                .expect("ai2d invoke (mosaic band) failed");

            const int strip_h = kMosaicBandH[static_cast<size_t>(i)];
            const int y0 = layout.band_y[static_cast<size_t>(i)];

            auto src_wrap = band_out_[static_cast<size_t>(i)]
                                .impl()
                                ->to_host()
                                .unwrap()
                                ->buffer()
                                .as_host()
                                .unwrap()
                                .map(map_access_::map_read)
                                .unwrap()
                                .buffer();

            if (out_tc_ == typecode_t::dt_uint8) {
                stitch_band_nchw_uint8(dst_u8, kMosaicRows, kMosaicCols, y0, strip_h, kMosaicCols,
                    reinterpret_cast<const uint8_t *>(src_wrap.data()));
            } else {
                stitch_band_nchw_float(dst_f32, kMosaicRows, kMosaicCols, y0, strip_h, kMosaicCols,
                    reinterpret_cast<const float *>(src_wrap.data()));
            }
        }

        hrt::sync(kmodel_in, sync_op_t::sync_write_back, true).expect("kmodel input sync after mosaic stitch failed");
        return true;
    }
};

int band_from_mosaic_y(float y, const MosaicLayout &L)
{
    for (int i = 0; i < kCamCount; ++i) {
        const int y0 = L.band_y[static_cast<size_t>(i)];
        const int h = L.band_h[static_cast<size_t>(i)];
        if (y >= static_cast<float>(y0) && y < static_cast<float>(y0 + h)) {
            return i;
        }
    }
    return 0;
}

OutputDet det_mosaic_to_cam(const OutputDet &in, int cam, const MosaicLayout &L)
{
    const float sc = L.scale[static_cast<size_t>(cam)];
    if (sc <= 0.F) {
        return in;
    }
    const float inv = 1.F / sc;
    const int by = L.band_y[static_cast<size_t>(cam)];
    const int xp = L.x_pad[static_cast<size_t>(cam)];
    const int ypb = L.y_pad_in_band[static_cast<size_t>(cam)];

    float x1 = in.box.x;
    float y1 = in.box.y;
    float x2 = x1 + in.box.width;
    float y2 = y1 + in.box.height;

    float cx1 = (x1 - static_cast<float>(xp)) * inv;
    float cy1 = (y1 - static_cast<float>(by + ypb)) * inv;
    float cx2 = (x2 - static_cast<float>(xp)) * inv;
    float cy2 = (y2 - static_cast<float>(by + ypb)) * inv;

    const float lx = std::min(cx1, cx2);
    const float rx = std::max(cx1, cx2);
    const float ty = std::min(cy1, cy2);
    const float byy = std::max(cy1, cy2);

    OutputDet out = in;
    const float nx1 = std::max(0.F, std::min(lx, static_cast<float>(kSrcW)));
    const float ny1 = std::max(0.F, std::min(ty, static_cast<float>(kSrcH)));
    const float nx2 = std::max(0.F, std::min(rx, static_cast<float>(kSrcW)));
    const float ny2 = std::max(0.F, std::min(byy, static_cast<float>(kSrcH)));
    out.box.x = nx1;
    out.box.y = ny1;
    out.box.width = std::max(1.F, nx2 - nx1);
    out.box.height = std::max(1.F, ny2 - ny1);
    return out;
}

static uint64_t monotonic_us()
{
    struct timespec ts;
    if (clock_gettime(CLOCK_MONOTONIC, &ts) != 0) {
        return 0;
    }
    return static_cast<uint64_t>(ts.tv_sec) * 1000000ULL + static_cast<uint64_t>(ts.tv_nsec) / 1000ULL;
}

static int resolve_label_by_name(const std::vector<std::string> &labels, const char *name)
{
    for (size_t i = 0; i < labels.size(); ++i) {
        std::string lower = labels[i];
        for (char &c : lower) {
            c = static_cast<char>(std::tolower(static_cast<unsigned char>(c)));
        }
        if (lower == name) {
            return static_cast<int>(i);
        }
    }
    return -1;
}

struct DetectionClassIds {
    int person = 0;
    int car = -1;
    int vehicle = -1;
};

static bool try_env_cls_index(const char *name, int *out)
{
    const char *p = std::getenv(name);
    if (p == nullptr || p[0] == '\0') {
        return false;
    }
    *out = std::atoi(p);
    return true;
}

static bool labels_are_generic_cls012(const std::vector<std::string> &labels)
{
    if (labels.size() != 3U) {
        return false;
    }
    for (size_t i = 0; i < 3U; ++i) {
        if (labels[i].size() < 4U || labels[i].rfind("cls", 0) != 0U) {
            return false;
        }
    }
    return true;
}

static DetectionClassIds resolve_detection_class_ids(const std::vector<std::string> &labels)
{
    DetectionClassIds ids;
    ids.person = resolve_label_by_name(labels, "pedestrian");
    if (ids.person < 0) {
        ids.person = resolve_label_by_name(labels, "person");
    }
    ids.car = resolve_label_by_name(labels, "motorcycle");
    if (ids.car < 0) {
        ids.car = resolve_label_by_name(labels, "car");
    }
    ids.vehicle = resolve_label_by_name(labels, "vehicle");
    if (ids.vehicle < 0) {
        ids.vehicle = resolve_label_by_name(labels, "vehicles");
    }

    /** 3-class e2e cls0/cls1/cls2: motorcycle, pedestrian, vehicle. */
    if (labels_are_generic_cls012(labels)) {
        if (ids.car < 0) {
            ids.car = 0;
        }
        if (ids.person < 0) {
            ids.person = 1;
        }
        if (ids.vehicle < 0) {
            ids.vehicle = 2;
        }
    } else if (ids.person < 0) {
        ids.person = 0;
    }

    (void)try_env_cls_index("TRIPLE_CAM_MOSAIC_CLS_PERSON", &ids.person);
    (void)try_env_cls_index("TRIPLE_CAM_MOSAIC_CLS_CAR", &ids.car);
    (void)try_env_cls_index("TRIPLE_CAM_MOSAIC_CLS_VEHICLE", &ids.vehicle);
    return ids;
}

static void publish_detection_counts_fifo(const std::vector<OutputDet> &det, const DetectionClassIds &class_ids,
    const MosaicLayout &layout)
{
    if (!PersonFifoWriterReady()) {
        return;
    }
    uint32_t people[3] = {0, 0, 0};
    uint32_t cars[3] = {0, 0, 0};
    uint32_t vehicles[3] = {0, 0, 0};
    for (const OutputDet &d : det) {
        uint32_t *slot = nullptr;
        if (d.label == class_ids.person) {
            slot = people;
        } else if (class_ids.car >= 0 && d.label == class_ids.car) {
            slot = cars;
        } else if (class_ids.vehicle >= 0 && d.label == class_ids.vehicle) {
            slot = vehicles;
        } else {
            continue;
        }
        const float cy = d.box.y + d.box.height * 0.5F;
        const int cam = band_from_mosaic_y(cy, layout);
        if (cam >= 0 && cam < kCamCount) {
            ++slot[static_cast<size_t>(cam)];
        }
    }

    uint16_t band_y[3];
    uint16_t band_h[3];
    for (int i = 0; i < kCamCount; ++i) {
        band_y[static_cast<size_t>(i)] = static_cast<uint16_t>(layout.band_y[static_cast<size_t>(i)]);
        band_h[static_cast<size_t>(i)] = static_cast<uint16_t>(layout.band_h[static_cast<size_t>(i)]);
    }
    PersonFifoWriterPublish(people, cars, vehicles, band_y, band_h, monotonic_us());
}

static bool person_fifo_enabled_by_env()
{
    const char *const env = std::getenv("TRIPLE_CAM_MOSAIC_PERSON_FIFO");
    if (env == nullptr || env[0] == '\0') {
        return true;
    }
    return std::atoi(env) != 0;
}

void signal_handler(int sig)
{
    if (sig == SIGINT || sig == SIGTERM) {
        g_running.store(false);
    }
}

void print_usage(const char *prog)
{
    std::printf("Usage: %s <kmodel_path> [obj_thresh] [nms_thresh] [max_mosaics] [frame_interval_ms] [debug_mode] "
                "[capture_period_ms]\n",
        prog);
    std::printf(
        "  Triple VICAP @ %dx%d capture → chn1 %dx%d NV12 → NV12→RGB + letterboxed resize → %dx%d mosaic "
        "(aspect preserved). Use a kmodel with input NCHW [1,3,%d,%d] (348×639 vstack 213×348 per band).\n",
        kIspInputWidth, kIspInputHeight, kIspChn1Width, kIspChn1Height, kMosaicCols, kMosaicRows, kMosaicRows,
        kMosaicCols);
    std::printf(
        "  (legacy kmodels expecting 640×380 will still letterbox in pre_process if shapes differ).\n");
    std::printf("  max_mosaics: inferences (each uses fresh frames from cam0..2); <=0 run until Ctrl+C\n");
    std::printf("  Runtime uses a pipelined double buffer: stitch overlaps NPU; throughput ≈ max(capture, infer).\n");
    std::printf("  frame_interval_ms: sleep after each full mosaic infer (default 0)\n");
    std::printf("  capture_period_ms: min gap between dump_release on the same cam (0=off)\n");
    std::printf(
        "  debug_mode: 0 quiet (YOLO FPS line every %d inferences),\n"
        "               1 + capture/infer thread timeline (monotonic μs per stage),\n"
        "               2 + per-camera grab μs + per-detection dump\n",
        kFpsYoloWindowFrames);
    std::printf(
        "  Env TRIPLE_CAM_MOSAIC_QUIET_YOLO=1: mosaic pipeline traces + debug argv still active, but suppress\n"
        "    YoloDetect ScopedTiming lines (otherwise they mix with [pipeline] and reorder in the log).\n");
    std::printf(
        "  Env TRIPLE_CAM_MOSAIC_NV12_AI2D=1: packed NV12 (phys_addr[1]==0 mmap path) → 3× AI2D (YUV420_NV12→NCHW) + "
        "CPU stitch into kmodel input; skips OpenCV mosaic. Spatial dims must match mosaic (NCHW H×W = %d×%d). "
        "Requires stride=w and f32/u8 model input; experimental.\n",
        kMosaicRows, kMosaicCols);
    std::printf("  Timing keys (debug≥1 infer line): pre_infer_us=pre_process_and_infer (AI2D+NPU);\n");
    std::printf("    decode_nms_us=YoloDetect::post_process only; run_summary_log_us includes run-line printf+mux wait.\n");
    std::printf("  capture wait_free_buf_us grows when infer is slower than stitch (double-buffer back-pressure).\n");
    std::printf("  On exit: triple_cam_mosaic_snapshot.jpg + triple_cam_mosaic_last_cam[0-2].jpg\n");
}

/** One mosaic inference == one run; boxes are in mosaic pixels (348×639). */
void print_run_output(int run_index, const std::vector<OutputDet> &det, const std::vector<std::string> &labels)
{
    std::printf("--- run %d: %zu detections ---\n", run_index, det.size());
    for (size_t i = 0; i < det.size(); ++i) {
        const OutputDet &d = det[i];
        const char *name = "cls?";
        if (d.label >= 0 && d.label < static_cast<int>(labels.size())) {
            name = labels[static_cast<size_t>(d.label)].c_str();
        }
        const auto &b = d.box;
        std::printf("  [%zu] %s conf=%.4f xyxy=(%.1f,%.1f,%.1f,%.1f)\n", i, name, d.confidence, b.x, b.y,
            b.x + b.width, b.y + b.height);
    }
    std::printf("--- end run %d ---\n", run_index);
}

bool is_integer_arg(const char *arg)
{
    if (arg == nullptr || *arg == '\0') {
        return false;
    }
    if (*arg == '-' || *arg == '+') {
        ++arg;
    }
    if (*arg == '\0') {
        return false;
    }
    while (*arg != '\0') {
        if (!std::isdigit(static_cast<unsigned char>(*arg))) {
            return false;
        }
        ++arg;
    }
    return true;
}

bool save_bgr(const char *path, const cv::Mat &bgr)
{
    if (bgr.empty()) {
        return false;
    }
    const std::vector<int> params{cv::IMWRITE_JPEG_QUALITY, 100};
    if (!cv::imwrite(path, bgr, params)) {
        std::printf("failed to write %s\n", path);
        return false;
    }
    std::printf("wrote %s\n", path);
    return true;
}

bool dump_nv12_to_bgr(k_vicap_dev dev, const k_video_frame_info &dump, cv::Mat *out_bgr)
{
    if (out_bgr == nullptr) {
        return false;
    }
    const k_video_frame &frame = dump.v_frame;
    const size_t width = frame.width > 0 ? frame.width : kIspChn1Width;
    const size_t height = frame.height > 0 ? frame.height : kIspChn1Height;
    if (width != kIspChn1Width || height != kIspChn1Height) {
        std::printf("unexpected frame size %ux%u format=%d (expected chn1 %dx%d)\n", frame.width, frame.height,
            frame.pixel_format, kIspChn1Width, kIspChn1Height);
        return false;
    }

    Nv12FrameLayout layout{};
    if (!nv12_frame_layout(frame, width, height, &layout)) {
        return false;
    }

    const bool is_nv21 = frame_uses_nv21_chroma(frame.pixel_format);

    const uint8_t *y_plane = nullptr;
    const uint8_t *uv_plane = nullptr;
    void *y_buf = nullptr;
    void *uv_buf = nullptr;
    void *packed_buf = nullptr;

    if (frame.phys_addr[1] != 0U) {
        y_buf = kd_mpi_sys_mmap(frame.phys_addr[0], layout.y_bytes);
        if (y_buf == nullptr || y_buf == MAP_FAILED) {
            return false;
        }
        uv_buf = kd_mpi_sys_mmap(frame.phys_addr[1], layout.uv_bytes);
        if (uv_buf == nullptr || uv_buf == MAP_FAILED) {
            kd_mpi_sys_munmap(y_buf, layout.y_bytes);
            return false;
        }
        y_plane = static_cast<const uint8_t *>(y_buf);
        uv_plane = static_cast<const uint8_t *>(uv_buf);
    } else {
        packed_buf = kd_mpi_sys_mmap(frame.phys_addr[0], layout.buf_bytes);
        if (packed_buf == nullptr || packed_buf == MAP_FAILED) {
            return false;
        }
        y_plane = static_cast<const uint8_t *>(packed_buf);
        uv_plane = y_plane + static_cast<ptrdiff_t>(layout.uv_offset);
    }

    const cv::Mat y_mat(static_cast<int>(height), static_cast<int>(width), CV_8UC1,
        const_cast<uint8_t *>(y_plane), layout.y_stride);
    const cv::Mat uv_mat(static_cast<int>(height / 2U), static_cast<int>(width / 2U), CV_8UC2,
        const_cast<uint8_t *>(uv_plane), layout.uv_stride);
    const int cvt_code = is_nv21 ? cv::COLOR_YUV2BGR_NV21 : cv::COLOR_YUV2BGR_NV12;
    cv::cvtColorTwoPlane(y_mat, uv_mat, *out_bgr, cvt_code);

    if (uv_buf != nullptr) {
        kd_mpi_sys_munmap(uv_buf, layout.uv_bytes);
    }
    if (y_buf != nullptr) {
        kd_mpi_sys_munmap(y_buf, layout.y_bytes);
    }
    if (packed_buf != nullptr) {
        kd_mpi_sys_munmap(packed_buf, layout.buf_bytes);
    }
    return true;
}

struct CamGrabCtx {
    k_vicap_dev dev{};
    Nv12MmapGrab *grab = nullptr;
    uint64_t *last_release_slot = nullptr;
    int capture_period_ms = 0;
    int cam_index = 0;
};

/** One-shot grab + mmap NV12 planes (no full-res BGR); caller releases after mosaic build. */
static void cam_grab_once(CamGrabCtx *c)
{
    if (c->grab == nullptr) {
        return;
    }
    Nv12MmapGrab *const g = c->grab;
    g->ok = false;

    if (c->capture_period_ms > 0 && c->last_release_slot != nullptr && *c->last_release_slot != 0U) {
        const uint64_t now = monotonic_us();
        const uint64_t min_gap_us = static_cast<uint64_t>(c->capture_period_ms) * 1000ULL;
        const uint64_t last = *c->last_release_slot;
        if (now < last + min_gap_us) {
            const uint64_t sleep_us = (last + min_gap_us) - now;
            if (sleep_us > 0U && sleep_us < 2000000ULL) {
                usleep(static_cast<useconds_t>(sleep_us));
            }
        }
    }

    k_video_frame_info dump_info;
    std::memset(&dump_info, 0, sizeof(dump_info));
    const k_s32 ret = kd_mpi_vicap_dump_frame(c->dev, VICAP_CHN_ID_1, VICAP_DUMP_YUV, &dump_info, 1000);
    if (ret != K_SUCCESS) {
        std::lock_guard<std::mutex> lk(g_print_mtx);
        std::printf("cam%d dump_frame failed ret=%d\n", c->cam_index, ret);
        return;
    }

    if (!vicap_attach_nv12_mmap(c->dev, dump_info, g)) {
        const k_s32 rr = kd_mpi_vicap_dump_release(c->dev, VICAP_CHN_ID_1, &dump_info);
        if (rr != K_SUCCESS) {
            std::lock_guard<std::mutex> lk(g_print_mtx);
            std::printf("cam%d dump_release after mmap fail ret=%d\n", c->cam_index, rr);
        }
        return;
    }
}

/** One BGR frame via OpenCV (for exit JPEGs only). */
static bool grab_one_bgr_snapshot(k_vicap_dev dev, cv::Mat *bgr)
{
    if (bgr == nullptr) {
        return false;
    }
    k_video_frame_info dump_info;
    std::memset(&dump_info, 0, sizeof(dump_info));
    if (kd_mpi_vicap_dump_frame(dev, VICAP_CHN_ID_1, VICAP_DUMP_YUV, &dump_info, 1000) != K_SUCCESS) {
        return false;
    }
    const bool ok = dump_nv12_to_bgr(dev, dump_info, bgr);
    const k_s32 rr = kd_mpi_vicap_dump_release(dev, VICAP_CHN_ID_1, &dump_info);
    if (rr != K_SUCCESS) {
        std::printf("grab_one_bgr_snapshot release failed dev=%d ret=%d\n", dev, rr);
    }
    return ok && (rr == K_SUCCESS);
}

int vb_init()
{
    k_vb_config config;
    std::memset(&config, 0, sizeof(config));
    config.max_pool_cnt = 16;

    const k_u32 chn1_nv12 = VICAP_ALIGN_UP((kIspChn1Height * kIspChn1Width * 3 / 2), VICAP_ALIGN_1K);
    const k_u32 vicap_in = VICAP_ALIGN_UP((kIspInputWidth * kIspInputHeight * 3), VICAP_ALIGN_1K);

    for (int i = 0; i < kCamCount; ++i) {
        config.comm_pool[i].blk_cnt = kVicapOutputBufNum;
        config.comm_pool[i].mode = VB_REMAP_MODE_NOCACHE;
        config.comm_pool[i].blk_size = chn1_nv12;
    }
    for (int i = 0; i < kCamCount; ++i) {
        config.comm_pool[kCamCount + i].blk_cnt = kVicapInputBufNum;
        config.comm_pool[kCamCount + i].mode = VB_REMAP_MODE_NOCACHE;
        config.comm_pool[kCamCount + i].blk_size = vicap_in;
    }

    k_s32 ret = kd_mpi_vb_set_config(&config);
    if (ret != K_SUCCESS) {
        std::printf("vb_set_config failed ret=%d\n", ret);
        return ret;
    }

    k_vb_supplement_config supplement_config;
    std::memset(&supplement_config, 0, sizeof(supplement_config));
    supplement_config.supplement_config |= VB_SUPPLEMENT_JPEG_MASK;
    ret = kd_mpi_vb_set_supplement_config(&supplement_config);
    if (ret != K_SUCCESS) {
        std::printf("vb_set_supplement_config failed ret=%d\n", ret);
        return ret;
    }

    ret = kd_mpi_vb_init();
    if (ret != K_SUCCESS) {
        std::printf("vb_init failed ret=%d\n", ret);
    }
    return ret;
}

int vicap_init(k_vicap_dev vicap_dev, k_vicap_sensor_type sensor_type, k_vicap_mirror mirror)
{
    k_vicap_sensor_info sensor_info;
    std::memset(&sensor_info, 0, sizeof(sensor_info));
    k_s32 ret = kd_mpi_vicap_get_sensor_info(sensor_type, &sensor_info);
    if (ret != K_SUCCESS) {
        std::printf("vicap get_sensor_info failed dev=%d sensor=%d ret=%d\n", vicap_dev, sensor_type, ret);
        return ret;
    }

    k_vicap_dev_attr dev_attr;
    std::memset(&dev_attr, 0, sizeof(dev_attr));
    dev_attr.acq_win.h_start = 0;
    dev_attr.acq_win.v_start = 0;
    dev_attr.acq_win.width = kIspInputWidth;
    dev_attr.acq_win.height = kIspInputHeight;
    dev_attr.mode = VICAP_WORK_OFFLINE_MODE;
    dev_attr.buffer_num = kVicapInputBufNum;
    dev_attr.buffer_size = VICAP_ALIGN_UP((kIspInputWidth * kIspInputHeight * 2), VICAP_ALIGN_1K);
    dev_attr.pipe_ctrl.data = 0xFFFFFFFF;
    dev_attr.pipe_ctrl.bits.af_enable = 0;
    dev_attr.pipe_ctrl.bits.ahdr_enable = 0;
    dev_attr.pipe_ctrl.bits.dnr3_enable = 0;
    dev_attr.dw_enable = K_FALSE;
    dev_attr.cpature_frame = 0;
    dev_attr.mirror = mirror;
    std::memcpy(&dev_attr.sensor_info, &sensor_info, sizeof(k_vicap_sensor_info));

    ret = kd_mpi_vicap_set_dev_attr(vicap_dev, dev_attr);
    if (ret != K_SUCCESS) {
        std::printf("vicap set_dev_attr failed dev=%d ret=%d\n", vicap_dev, ret);
        return ret;
    }

    k_vicap_chn_attr chn_attr;
    std::memset(&chn_attr, 0, sizeof(chn_attr));
    chn_attr.out_win.h_start = 0;
    chn_attr.out_win.v_start = 0;
    chn_attr.out_win.width = kIspChn1Width;
    chn_attr.out_win.height = kIspChn1Height;
    chn_attr.crop_win = dev_attr.acq_win;
    chn_attr.scale_win = chn_attr.out_win;
    chn_attr.crop_enable = K_FALSE;
    /* Only enable the ISP scaler when output size differs from crop; 1:1 + scale on can soften detail.
     * For chn1 640x360 != acq 1920x1080 this evaluates K_TRUE, mirroring single_cam_yolo_infer so the
     * ISP performs the 1920x1080 -> 640x360 downscale in hardware (instead of leaving downsize to a
     * software path or, with scale_enable forced false, producing frames at the wrong size). */
    chn_attr.scale_enable =
        (chn_attr.out_win.width != chn_attr.crop_win.width || chn_attr.out_win.height != chn_attr.crop_win.height)
            ? K_TRUE
            : K_FALSE;
    chn_attr.chn_enable = K_TRUE;
    chn_attr.pix_format = PIXEL_FORMAT_YUV_SEMIPLANAR_420;
    chn_attr.buffer_num = kVicapOutputBufNum;
    chn_attr.buffer_size = VICAP_ALIGN_UP((kIspChn1Height * kIspChn1Width * 3 / 2), VICAP_ALIGN_1K);
    ret = kd_mpi_vicap_set_chn_attr(vicap_dev, VICAP_CHN_ID_1, chn_attr);
    if (ret != K_SUCCESS) {
        std::printf("vicap set chn1 attr failed dev=%d ret=%d\n", vicap_dev, ret);
        return ret;
    }

    ret = kd_mpi_vicap_set_database_parse_mode(vicap_dev, VICAP_DATABASE_PARSE_XML_JSON);
    if (ret != K_SUCCESS) {
        std::printf("vicap set_database_parse_mode failed dev=%d ret=%d\n", vicap_dev, ret);
        return ret;
    }

    ret = kd_mpi_vicap_init(vicap_dev);
    if (ret != K_SUCCESS) {
        std::printf("vicap init failed dev=%d ret=%d\n", vicap_dev, ret);
    }
    return ret;
}

int vicap_stream(k_vicap_dev vicap_dev, bool enable)
{
    k_s32 ret = enable ? kd_mpi_vicap_start_stream(vicap_dev) : kd_mpi_vicap_stop_stream(vicap_dev);
    if (ret != K_SUCCESS) {
        std::printf("vicap %s stream failed dev=%d ret=%d\n", enable ? "start" : "stop", vicap_dev, ret);
    }
    return ret;
}

/** Double-buffered pipeline: capture thread grab×3 + stitch while this thread runs NPU on the other buffer.
 *  Throughput ≈ max(T_capture, T_infer), not T_capture + T_infer.
 *
 *  You cannot safely stitch into the same cv::Mat the NPU is consuming; two slots alternate instead. */
struct MosaicPipelineCtx {
    /** RGB mosaic per slot (matches YoloDetect image-mode RGB expectation after stitch). */
    cv::Mat buf[kMosaicPipelineSlots];
    /** Capture-thread scratch for NV12→RGB (one 640×360 frame at a time); never read by infer thread. */
    cv::Mat cam_rgb_scratch;
    /** Per buffer-slot triple grab; infer releases NV12 dumps when use_nv12_ai2d is set. */
    std::array<std::array<Nv12MmapGrab, kCamCount>, kMosaicPipelineSlots> grabs_by_slot{};
    std::array<CamGrabCtx, kCamCount> grab_ctx{};
    int latest = -1;
    int inf_reading = -1;
    std::mutex mu;
    std::condition_variable cv_new;
    std::condition_variable cv_freed;
    const std::array<k_vicap_dev, kCamCount> *devs = nullptr;
    const MosaicLayout *layout = nullptr;
    int capture_period_ms = 0;
    std::array<uint64_t, kCamCount> last_release_us{{0, 0, 0}};
    std::atomic<bool> run{true};
    std::atomic<int> frames_captured{0};
    std::atomic<int> frames_dropped{0};
    /** Monotonic serial number for published mosaics (for correlating capture vs infer logs). */
    std::atomic<uint64_t> mosaic_serial_gen{0};
    /** Set together with `latest` under `mu` so the infer thread can print the matching id. */
    uint64_t last_published_serial = 0;
    int pipeline_debug = 0;
    /** If true: capture keeps NV12 mmap until infer runs packed-NV12 AI2D mosaic (see MosaicNv12Ai2dMosaic). */
    bool use_nv12_ai2d = false;
    MosaicNv12Ai2dMosaic *nv12_mosaic = nullptr;
};

static int mosaic_pipeline_free_slot_locked(const MosaicPipelineCtx &P)
{
    for (int s = 0; s < kMosaicPipelineSlots; ++s) {
        if (s != P.latest && s != P.inf_reading) {
            return s;
        }
    }
    return -1;
}

static int mosaic_pipeline_wait_latest(MosaicPipelineCtx &P, uint64_t *out_mosaic_serial = nullptr)
{
    std::unique_lock<std::mutex> lk(P.mu);
    P.cv_new.wait(lk, [&] {
        return P.latest != -1 || !P.run.load(std::memory_order_relaxed) || !g_running.load();
    });
    const int slot = P.latest;
    if (slot != -1) {
        if (out_mosaic_serial != nullptr) {
            *out_mosaic_serial = P.last_published_serial;
        }
        P.latest = -1;
        P.inf_reading = slot;
    }
    return slot;
}

static void mosaic_pipeline_release_infer_slot(MosaicPipelineCtx &P)
{
    std::lock_guard<std::mutex> lk(P.mu);
    P.inf_reading = -1;
    P.cv_freed.notify_one();
}

static void mosaic_pipeline_stop(MosaicPipelineCtx &P, pthread_t cap_thread)
{
    P.run.store(false, std::memory_order_relaxed);
    {
        std::lock_guard<std::mutex> lk(P.mu);
        P.cv_freed.notify_all();
        P.cv_new.notify_all();
    }
    pthread_join(cap_thread, nullptr);
}

/** Grabs NV12 mmap for cam0→2 on the capture thread only. Concurrent `vicap_dump_frame` from multiple pthreads has
 *  been observed to stall the BSP after the first successful triple grab.
 *  @param out_cam_grab_us optional; per-cam wall time for `cam_grab_once` (monotonic delta, μs). */
static void mosaic_capture_grab_three(MosaicPipelineCtx *P, int buf_slot, uint64_t *out_cam_grab_us = nullptr)
{
    for (int cam = 0; cam < kCamCount; ++cam) {
        const size_t idx = static_cast<size_t>(cam);
        CamGrabCtx *const ctx = &P->grab_ctx[idx];
        ctx->dev = (*P->devs)[idx];
        ctx->grab = &P->grabs_by_slot[static_cast<size_t>(buf_slot)][idx];
        ctx->last_release_slot = &P->last_release_us[idx];
        ctx->capture_period_ms = P->capture_period_ms;
        ctx->cam_index = cam;
        const uint64_t t0 = monotonic_us();
        cam_grab_once(ctx);
        if (out_cam_grab_us != nullptr) { 
            out_cam_grab_us[idx] = monotonic_us() - t0;
        }
    }
}

static void *mosaic_capture_thread_fn(void *arg)
{
    auto *P = static_cast<MosaicPipelineCtx *>(arg);
    std::array<uint64_t, kCamCount> cam_grab_us{};

    while (P->run.load(std::memory_order_relaxed) && g_running.load()) {
        const uint64_t t_iter0 = monotonic_us();
        int slot = -1;
        uint64_t wait_buf_us = 0;
        int snap_inf_reading = -2;
        int snap_latest_slot = -2;
        {
            std::unique_lock<std::mutex> lk(P->mu);
            const uint64_t t_wait0 = monotonic_us();
            P->cv_freed.wait(lk, [&] {
                slot = mosaic_pipeline_free_slot_locked(*P);
                return slot != -1 || !P->run.load(std::memory_order_relaxed) || !g_running.load();
            });
            wait_buf_us = monotonic_us() - t_wait0;
            snap_inf_reading = P->inf_reading;
            snap_latest_slot = P->latest;
        }

        if (P->pipeline_debug >= 1 && P->run.load(std::memory_order_relaxed) && g_running.load() && slot != -1) {
            std::lock_guard<std::mutex> plk(g_print_mtx);
            std::printf(
                "[pipeline][capture] pthread=%lu wait_free_buf_us=%llu inf_reading_slot=%d latest_slot=%d "
                "chosen_write_slot=%d\n",
                static_cast<unsigned long>(pthread_self()), static_cast<unsigned long long>(wait_buf_us),
                snap_inf_reading, snap_latest_slot, slot);
            std::fflush(stdout);
        }

        if (!P->run.load(std::memory_order_relaxed) || !g_running.load()) {
            break;
        }

        const uint64_t t_grab0 = monotonic_us();
        if (P->pipeline_debug >= 2) {
            mosaic_capture_grab_three(P, slot, cam_grab_us.data());
        } else {
            mosaic_capture_grab_three(P, slot, nullptr);
        }
        const uint64_t grab_three_us = monotonic_us() - t_grab0;

        bool all_ok = true;
        for (int i = 0; i < kCamCount && all_ok; ++i) {
            all_ok = P->grabs_by_slot[static_cast<size_t>(slot)][static_cast<size_t>(i)].ok;
        }

        if (!all_ok) {
            if (P->pipeline_debug >= 1) {
                std::lock_guard<std::mutex> plk(g_print_mtx);
                std::printf("[pipeline][capture] pthread=%lu grab_three failed (one or more cams); releasing, sleep 2ms\n",
                    static_cast<unsigned long>(pthread_self()));
                std::fflush(stdout);
            }
            for (int i = 0; i < kCamCount; ++i) {
                static_cast<void>(nv12_grab_release(&P->grabs_by_slot[static_cast<size_t>(slot)][static_cast<size_t>(i)]));
            }
            usleep(2000);
            continue;
        }

        const uint64_t t_stitch0 = monotonic_us();
        uint64_t stitch_and_release_us = 0;
        if (P->use_nv12_ai2d) {
            stitch_and_release_us = monotonic_us() - t_stitch0;
        } else {
            build_mosaic_cv_resize(P->grabs_by_slot[static_cast<size_t>(slot)], &P->buf[slot], *P->layout,
                &P->cam_rgb_scratch);

            for (int i = 0; i < kCamCount; ++i) {
                if (nv12_grab_release(&P->grabs_by_slot[static_cast<size_t>(slot)][static_cast<size_t>(i)])
                    && P->capture_period_ms > 0) {
                    P->last_release_us[static_cast<size_t>(i)] = monotonic_us();
                }
            }
            stitch_and_release_us = monotonic_us() - t_stitch0;
        }

        const uint64_t mosaic_id = 1 + P->mosaic_serial_gen.fetch_add(1ULL, std::memory_order_relaxed);

        bool dropped = false;
        {
            std::lock_guard<std::mutex> lk(P->mu);
            if (P->latest != -1) {
                P->frames_dropped.fetch_add(1, std::memory_order_relaxed);
                dropped = true;
            }
            P->last_published_serial = mosaic_id;
            P->latest = slot;
            P->cv_new.notify_one();
        }

        P->frames_captured.fetch_add(1, std::memory_order_relaxed);

        if (P->pipeline_debug >= 1) {
            const uint64_t total_us = monotonic_us() - t_iter0;
            std::lock_guard<std::mutex> plk(g_print_mtx);
            if (P->pipeline_debug >= 2) {
                std::printf(
                    "[pipeline][capture] pthread=%lu mosaic_id=%llu buf_slot=%d dropped_prev=%d grab3_us=%llu "
                    "cam_us=[%llu,%llu,%llu] stitch_release_us=%llu total_iter_us=%llu\n",
                    static_cast<unsigned long>(pthread_self()), static_cast<unsigned long long>(mosaic_id), slot,
                    dropped ? 1 : 0, static_cast<unsigned long long>(grab_three_us),
                    static_cast<unsigned long long>(cam_grab_us[0]),
                    static_cast<unsigned long long>(cam_grab_us[1]),
                    static_cast<unsigned long long>(cam_grab_us[2]),
                    static_cast<unsigned long long>(stitch_and_release_us), static_cast<unsigned long long>(total_us));
            } else {
                std::printf(
                    "[pipeline][capture] pthread=%lu mosaic_id=%llu buf_slot=%d dropped_prev=%d grab3_us=%llu "
                    "stitch_release_us=%llu total_iter_us=%llu\n",
                    static_cast<unsigned long>(pthread_self()), static_cast<unsigned long long>(mosaic_id), slot,
                    dropped ? 1 : 0, static_cast<unsigned long long>(grab_three_us),
                    static_cast<unsigned long long>(stitch_and_release_us), static_cast<unsigned long long>(total_us));
            }
            std::fflush(stdout);
        }
    }

    {
        std::lock_guard<std::mutex> lk(P->mu);
        P->cv_new.notify_all();
    }
    return nullptr;
}

void run_mosaic_loop_pipelined(const std::array<k_vicap_dev, kCamCount> &devs, YoloDetect *detector,
    const MosaicLayout &layout, int max_mosaics, int frame_interval_ms, int capture_period_ms, int debug_mode,
    const DetectionClassIds &class_ids)
{
    MosaicPipelineCtx P;
    for (auto &buf : P.buf) {
        buf.create(kMosaicRows, kMosaicCols, CV_8UC3);
        buf.setTo(cv::Scalar(114, 114, 114));
    }
    P.cam_rgb_scratch.create(kIspChn1Height, kIspChn1Width, CV_8UC3);
    P.devs = &devs;
    P.layout = &layout;
    P.capture_period_ms = capture_period_ms;
    P.pipeline_debug = debug_mode;

    MosaicNv12Ai2dMosaic nv12_mosaic;
    const char *const nv12_env = std::getenv("TRIPLE_CAM_MOSAIC_NV12_AI2D");
    if (detector != nullptr && nv12_env != nullptr && nv12_env[0] != '\0' && std::atoi(nv12_env) != 0) {
        char err[256];
        if (nv12_mosaic.ensure_built(detector->get_input_tensor(0).datatype(), err, sizeof err)) {
            P.use_nv12_ai2d = true;
            P.nv12_mosaic = &nv12_mosaic;
            std::printf(
                "TRIPLE_CAM_MOSAIC_NV12_AI2D=1: packed NV12 + phys → 3×AI2D (NV12→NCHW) + stitch into kmodel input "
                "(no OpenCV mosaic / full-frame color convert).\n");
        } else {
            std::printf("TRIPLE_CAM_MOSAIC_NV12_AI2D requested but AI2D mosaic init failed: %s — using CPU mosaic.\n", err);
        }
    }

    pthread_t cap_thread{};
    if (pthread_create(&cap_thread, nullptr, mosaic_capture_thread_fn, &P) != 0) {
        std::printf("mosaic pipeline: capture thread creation failed.\n");
        return;
    }
    if (debug_mode >= 1) {
        std::lock_guard<std::mutex> plk(g_print_mtx);
        std::printf(
            "[pipeline] capture thread spawned; infer runs on main pthread=%lu — enable debug≥1 timelines (μs, monotonic)\n",
            static_cast<unsigned long>(pthread_self()));
        std::fflush(stdout);
    }

    // Mosaic slots are RGB (NV12→RGB on capture thread); infer skips full-frame BGR→RGB before pre_process.

    std::vector<OutputDet> yolo_raw;
    yolo_raw.reserve(256);
    int infer_count = 0;
    int fps_window_yolo = 0;
    uint64_t fps_window_t0_us = 0;
    size_t fps_window_detections = 0;
    int last_snapshot_slot = -1;

    while (g_running.load()) {
        if (max_mosaics > 0 && infer_count >= max_mosaics) {
            break;
        }
        
        uint64_t mosaic_serial = 0;
        const uint64_t t_wait_infer0 = monotonic_us();
        const int slot = mosaic_pipeline_wait_latest(P, (debug_mode >= 1) ? &mosaic_serial : nullptr);
        const uint64_t wait_mosaic_us = monotonic_us() - t_wait_infer0;

        if (slot == -1) {
            if (debug_mode >= 1) {
                std::lock_guard<std::mutex> plk(g_print_mtx);
                std::printf("[pipeline][infer] main pthread=%lu exit_wait (shutdown) wait_mosaic_us=%llu\n",
                    static_cast<unsigned long>(pthread_self()), static_cast<unsigned long long>(wait_mosaic_us));
                std::fflush(stdout);
            }
            break;
        }

        uint64_t bgr2rgb_us = 0;
        uint64_t pre_infer_us = 0;
        uint64_t decode_nms_us = 0;
        uint64_t run_log_us = 0;

        if (detector != nullptr) {
            const uint64_t t_pi0 = monotonic_us();
            bool yolo_ran = false;
            if (P.use_nv12_ai2d && P.nv12_mosaic != nullptr) {
                char err[256];
                runtime_tensor k_in = detector->get_input_tensor(0);
                if (P.nv12_mosaic->fill_kmodel_input(k_in, P.grabs_by_slot[static_cast<size_t>(slot)], layout, err,
                        sizeof err)) {
                    detector->params = cv::Vec4d(1.0, 1.0, 0.0, 0.0);
                    detector->run();
                    detector->get_output();
                    yolo_ran = true;
                } else {
                    std::lock_guard<std::mutex> plk(g_print_mtx);
                    std::printf("NV12 AI2D mosaic failed: %s\n", err);
                }
                for (int i = 0; i < kCamCount; ++i) {
                    if (nv12_grab_release(&P.grabs_by_slot[static_cast<size_t>(slot)][static_cast<size_t>(i)])
                        && P.capture_period_ms > 0) {
                        P.last_release_us[static_cast<size_t>(i)] = monotonic_us();
                    }
                }
                pre_infer_us = monotonic_us() - t_pi0;
            } else {
                detector->pre_process_and_infer(P.buf[slot]);
                pre_infer_us = monotonic_us() - t_pi0;
                yolo_ran = true;
            }

            if (yolo_ran) {
                const uint64_t t_po0 = monotonic_us();
                detector->post_process(yolo_raw, detector->params);
                decode_nms_us = monotonic_us() - t_po0;
                publish_detection_counts_fifo(yolo_raw, class_ids, layout);

                ++infer_count;
                const uint64_t t_log0 = monotonic_us();
                if (debug_mode >= 2) {
                    std::lock_guard<std::mutex> lk(g_print_mtx);
                    print_run_output(infer_count, yolo_raw, detector->labels);
                    std::fflush(stdout);
                } else if (debug_mode >= 1) {
                    std::lock_guard<std::mutex> lk(g_print_mtx);
                    std::printf("run %d: %zu det  [candidates decode=%zu  after_cap=%zu  after_nms=%zu]\n", infer_count,
                        yolo_raw.size(), detector->stats_candidates_after_decode(),
                        detector->stats_candidates_after_cap(), detector->stats_detections_after_nms());
                    std::fflush(stdout);
                } else {
                    std::printf("run %d: %zu det  [candidates decode=%zu  after_cap=%zu  after_nms=%zu]\n", infer_count,
                        yolo_raw.size(), detector->stats_candidates_after_decode(),
                        detector->stats_candidates_after_cap(), detector->stats_detections_after_nms());
                }
                run_log_us = monotonic_us() - t_log0;

                if (fps_window_yolo == 0) {
                    fps_window_t0_us = monotonic_us();
                    fps_window_detections = 0;
                }
                ++fps_window_yolo;
                fps_window_detections += yolo_raw.size();
                if (fps_window_yolo >= kFpsYoloWindowFrames) {
                    const uint64_t dt_us = monotonic_us() - fps_window_t0_us;
                    if (dt_us > 0U) {
                        const double fps = static_cast<double>(fps_window_yolo) * 1000000.0 / static_cast<double>(dt_us);
                        const double avg_det =
                            static_cast<double>(fps_window_detections) / static_cast<double>(fps_window_yolo);

                        std::printf("YOLO FPS (last %d inferences): %.2f, detections: %zu total, %.2f avg\n",
                            fps_window_yolo, fps, fps_window_detections, avg_det);
                    }
                    fps_window_yolo = 0;
                    fps_window_detections = 0;
                }
            }
        }

        const uint64_t t_release0 = monotonic_us();
        mosaic_pipeline_release_infer_slot(P);
        const uint64_t release_notify_us = monotonic_us() - t_release0;

        if (debug_mode >= 1) {
            std::lock_guard<std::mutex> plk(g_print_mtx);
            if (detector != nullptr) {
                std::printf(
                    "[pipeline][infer] main pthread=%lu mosaic_id=%llu buf_slot=%d wait_mosaic_us=%llu bgr2rgb_us=%llu "
                    "pre_infer_us=%llu decode_nms_us=%llu run_summary_log_us=%llu release_notify_us=%llu\n",
                    static_cast<unsigned long>(pthread_self()), static_cast<unsigned long long>(mosaic_serial), slot,
                    static_cast<unsigned long long>(wait_mosaic_us), static_cast<unsigned long long>(bgr2rgb_us),
                    static_cast<unsigned long long>(pre_infer_us), static_cast<unsigned long long>(decode_nms_us),
                    static_cast<unsigned long long>(run_log_us), static_cast<unsigned long long>(release_notify_us));
            } else {
                std::printf(
                    "[pipeline][infer] main pthread=%lu mosaic_id=%llu buf_slot=%d wait_mosaic_us=%llu "
                    "release_notify_us=%llu (detector=null)\n",
                    static_cast<unsigned long>(pthread_self()), static_cast<unsigned long long>(mosaic_serial), slot,
                    static_cast<unsigned long long>(wait_mosaic_us),
                    static_cast<unsigned long long>(release_notify_us));
            }
            std::fflush(stdout);
        }

        last_snapshot_slot = slot;

        if (frame_interval_ms > 0 && g_running.load()) {
            usleep(static_cast<useconds_t>(frame_interval_ms) * 1000U);
        }
    }

    mosaic_pipeline_stop(P, cap_thread);

    std::printf("Pipeline exit: yolo_runs=%d  mosaic_captured=%d  mosaic_dropped=%d\n", infer_count,
        P.frames_captured.load(), P.frames_dropped.load());

    // Mosaic slots are RGB; convert for JPEG (imwrite expects BGR).
    if (!P.use_nv12_ai2d && last_snapshot_slot >= 0 && !P.buf[static_cast<size_t>(last_snapshot_slot)].empty()) {
        cv::Mat snap_bgr;
        cv::cvtColor(P.buf[static_cast<size_t>(last_snapshot_slot)], snap_bgr, cv::COLOR_RGB2BGR);
        save_bgr("triple_cam_mosaic_snapshot.jpg", snap_bgr);
    }
    std::array<cv::Mat, kCamCount> exit_bgr;
    for (int i = 0; i < kCamCount; ++i) {
        exit_bgr[static_cast<size_t>(i)].create(kIspChn1Height, kIspChn1Width, CV_8UC3);
        static_cast<void>(grab_one_bgr_snapshot(devs[static_cast<size_t>(i)], &exit_bgr[static_cast<size_t>(i)]));
    }
    for (int i = 0; i < kCamCount; ++i) {
        const std::string p = "triple_cam_mosaic_last_cam" + std::to_string(i) + ".jpg";
        if (!exit_bgr[static_cast<size_t>(i)].empty()) {
            save_bgr(p.c_str(), exit_bgr[static_cast<size_t>(i)]);
        }
    }
    std::printf(
        "Capture loop exit: mosaic_captured=%d yolo_runs=%d\n", P.frames_captured.load(), infer_count);
}

} // namespace

int main(int argc, char *argv[])
{
    setvbuf(stdout, nullptr, _IONBF, 0);
    setvbuf(stderr, nullptr, _IONBF, 0);

    if (argc > 1 && std::strcmp(argv[1], "--help") == 0) {
        print_usage(argv[0]);
        return 0;
    }

    if (argc < 2) {
        print_usage(argv[0]);
        return 1;
    }

    const char *kmodel_path = argv[1];
    float obj_thresh = 0.2F;
    float nms_thresh = 0.45F;
    int max_mosaics = 0;
    int frame_interval_ms = 0;
    int debug_mode = 0;
    int capture_period_ms = 0;

    if (argc >= 3) {
        obj_thresh = static_cast<float>(std::atof(argv[2]));
    }
    if (argc >= 4) {
        nms_thresh = static_cast<float>(std::atof(argv[3]));
    }
    if (argc >= 5 && is_integer_arg(argv[4])) {
        max_mosaics = std::atoi(argv[4]);
    }
    if (argc >= 6 && is_integer_arg(argv[5])) {
        frame_interval_ms = std::atoi(argv[5]);
    }
    if (argc >= 7 && is_integer_arg(argv[6])) {
        debug_mode = std::atoi(argv[6]);
    }
    if (argc >= 8 && is_integer_arg(argv[7])) {
        capture_period_ms = std::atoi(argv[7]);
        if (capture_period_ms < 0) {
            capture_period_ms = 0;
        }
    }

    int yolo_debug_internal = debug_mode;
    const char *const quiet_env = std::getenv("TRIPLE_CAM_MOSAIC_QUIET_YOLO");
    if (quiet_env != nullptr && quiet_env[0] != '\0' && std::atoi(quiet_env) != 0) {
        yolo_debug_internal = 0;
        std::printf("TRIPLE_CAM_MOSAIC_QUIET_YOLO set: mosaic pipeline dbg=%d, Yolo ScopedTiming disabled\n", debug_mode);
    }

    MosaicLayout layout{};
    layout.band_y = kMosaicBandY;
    layout.band_h = kMosaicBandH;
    layout.scale = kMosaicBandScale;
    layout.x_pad = kMosaicBandXPad;
    layout.y_pad_in_band = kMosaicBandYPadInBand;

    std::printf(
        "triple_cam_yolo_mosaic: kmodel=%s obj=%.3f nms=%.3f max_mosaics=%d interval_ms=%d debug=%d capture_period_ms=%d\n",
        kmodel_path, static_cast<double>(obj_thresh), static_cast<double>(nms_thresh), max_mosaics, frame_interval_ms,
        debug_mode, capture_period_ms);
    std::printf("  VICAP chn1 %dx%d; mosaic %dx%d (bands h=%d,%d,%d; aspect letterbox per band, RGB slots).\n",
        kIspChn1Width, kIspChn1Height, kMosaicCols, kMosaicRows, kMosaicBandH[0], kMosaicBandH[1], kMosaicBandH[2]);

    std::unique_ptr<YoloDetect> yolo;

    struct sigaction sa;
    std::memset(&sa, 0, sizeof(sa));
    sa.sa_handler = signal_handler;
    sigfillset(&sa.sa_mask);
    sigaction(SIGINT, &sa, nullptr);
    sigaction(SIGTERM, &sa, nullptr);

    k_s32 ret = vb_init();
    if (ret != K_SUCCESS) {
        return ret;
    }

    if (person_fifo_enabled_by_env()) {
        if (PersonFifoWriterInit() != 0) {
            std::printf("person datafifo disabled; continuing without Linux detection-count export.\n");
        }
    } else {
        std::printf("TRIPLE_CAM_MOSAIC_PERSON_FIFO=0: datafifo export disabled.\n");
    }

    const std::array<k_vicap_dev, kCamCount> devs{{VICAP_DEV_ID_0, VICAP_DEV_ID_1, VICAP_DEV_ID_2}};
    /** Top=OV5647@CSI0; middle+bottom=GC2093 profile (same as working cam2 — do not share OV5647 CSI0 on dev1). */
    const k_vicap_sensor_type sensors[kCamCount] = {
        OV_OV5647_MIPI_CSI0_1920X1080_30FPS_10BIT_LINEAR,
        OV_OV5647_MIPI_CSI1_1920X1080_30FPS_10BIT_LINEAR,
        GC2093_MIPI_CSI2_1920X1080_30FPS_10BIT_LINEAR,
    };

    bool inited[kCamCount] = {false, false, false};
    bool started[kCamCount] = {false, false, false};
    DetectionClassIds class_ids;

    for (int i = 0; i < kCamCount; ++i) {
        ret = vicap_init(devs[static_cast<size_t>(i)], sensors[i], kVicapMirrorAll);
        if (ret != K_SUCCESS) {
            g_running.store(false);
            goto cleanup;
        }
        inited[i] = true;
    }

    for (int i = 0; i < kCamCount; ++i) {
        ret = vicap_stream(devs[static_cast<size_t>(i)], true);
        if (ret != K_SUCCESS) {
            g_running.store(false);
            goto cleanup;
        }
        started[i] = true;
    }

    usleep(200000);

    yolo = std::make_unique<YoloDetect>(kmodel_path, obj_thresh, nms_thresh, yolo_debug_internal);

    {
        const dims_t in_shape = yolo->get_input_tensor(0).shape();
        if (in_shape.size() >= 4U) {
            const int in_h = static_cast<int>(in_shape[2]);
            const int in_w = static_cast<int>(in_shape[3]);
            std::printf("kmodel input[0] shape: NCHW [%d, %d, %d, %d]\n", static_cast<int>(in_shape[0]),
                static_cast<int>(in_shape[1]), in_h, in_w);
            if (in_h != kMosaicRows || in_w != kMosaicCols) {
                std::printf(
                    "WARNING: kmodel input %dx%d != mosaic %dx%d — YoloDetect::pre_process will letterbox/resize.\n",
                    in_w, in_h, kMosaicCols, kMosaicRows);
            }
        } else {
            std::printf("kmodel input[0]: unexpected rank %zu\n", in_shape.size());
        }
        const std::vector<int> &osh = yolo->output_shape_0();
        if (osh.size() >= 3) {
            std::printf("kmodel output[0] shape: [%d, %d, %d]\n", osh[0], osh[1], osh[2]);
        } else {
            std::printf("kmodel output[0]: unexpected rank %zu\n", osh.size());
        }
        std::printf("class name table size: %zu (inference uses kmodel output channels, not this count)\n",
            yolo->labels.size());
        std::printf("If every run shows exactly 3 detections: (1) three-class models often leave one box per class "
                    "after NMS — check decode=/cap= lines; (2) some exports fix max_det=3; (3) raise obj_thresh or "
                    "lower nms_thresh to see counts change.\n");
    }

    class_ids = resolve_detection_class_ids(yolo->labels);
    std::printf("detection class ids: pedestrian=%d motorcycle=%d vehicle=%d", class_ids.person, class_ids.car,
        class_ids.vehicle);
    if (labels_are_generic_cls012(yolo->labels)) {
        std::printf(" (3-class e2e: cls0=motorcycle cls1=pedestrian cls2=vehicle)");
    }
    std::printf("\n");
    for (size_t i = 0; i < yolo->labels.size() && i < 8U; ++i) {
        std::printf("  label[%zu]=%s\n", i, yolo->labels[i].c_str());
    }

    std::printf("triple_cam_yolo_mosaic running. Press Ctrl+C to stop.\n");
    run_mosaic_loop_pipelined(devs, yolo.get(), layout, max_mosaics, frame_interval_ms, capture_period_ms, debug_mode,
        class_ids);

cleanup:
    PersonFifoWriterDeinit();
    yolo.reset();
    for (int i = kCamCount - 1; i >= 0; --i) {
        if (started[i]) {
            vicap_stream(devs[static_cast<size_t>(i)], false);
        }
        if (inited[i]) {
            ret = kd_mpi_vicap_deinit(devs[static_cast<size_t>(i)]);
            if (ret != K_SUCCESS) {
                std::printf("vicap deinit failed dev=%d ret=%d\n", devs[static_cast<size_t>(i)], ret);
            }
        }
    }

    ret = kd_mpi_vb_exit();
    if (ret != K_SUCCESS) {
        std::printf("vb_exit failed ret=%d\n", ret);
        return ret;
    }

    return 0;
}
