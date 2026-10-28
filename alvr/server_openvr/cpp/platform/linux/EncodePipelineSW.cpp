#include "EncodePipelineSW.h"

#include "ALVR-common/packet_types.h"
#include "alvr_server/Logger.h"
#include "alvr_server/bindings.h"

#include <algorithm>
#include <cstdarg>
#include <cstdio>
#include <stdexcept>
#include <thread>

namespace {

void x264_log(void*, int level, const char* fmt, va_list args) {
    char buf[256];
    vsnprintf(buf, sizeof(buf), fmt, args);
    switch (level) {
    case X264_LOG_ERROR:
        Error("x264: %s", buf);
        break;
    case X264_LOG_WARNING:
        Warn("x264: %s", buf);
        break;
    case X264_LOG_INFO:
        Info("x264: %s", buf);
        break;
    default:
        break;
    }
}

u32 findMemoryType(alvr::VkContext& ctx, u32 typeBits, vk::MemoryPropertyFlags wanted) {
    auto props = ctx.physDev.getMemoryProperties();
    for (u32 i = 0; i < props.memoryTypeCount; ++i) {
        if ((typeBits & (1u << i)) && (props.memoryTypes[i].propertyFlags & wanted) == wanted) {
            return i;
        }
    }
    return UINT32_MAX;
}

// Full-range BT.709, the colour description the hardware encoders also use.
// Coefficients are scaled by 65536; each row of the chroma ones sums to zero.
inline uint8_t lumaOf(int r, int g, int b) {
    return uint8_t((13933 * r + 46871 * g + 4732 * b + 32768) >> 16);
}
inline uint8_t cbOf(int r, int g, int b) {
    return uint8_t(std::clamp(((-7510 * r - 25258 * g + 32768 * b + 32768) >> 16) + 128, 0, 255));
}
inline uint8_t crOf(int r, int g, int b) {
    return uint8_t(std::clamp(((32768 * r - 29766 * g - 3002 * b + 32768) >> 16) + 128, 0, 255));
}

} // namespace

alvr::EncodePipelineSW::EncodePipelineSW(
    VkContext& vk_ctx,
    vk::Image outputImage,
    vk::Format outputFormat,
    uint32_t width,
    uint32_t height
)
    : vk_ctx(vk_ctx)
    , outputImage(outputImage)
    , bgra(outputFormat == vk::Format::eB8G8R8A8Unorm || outputFormat == vk::Format::eB8G8R8A8Srgb)
    , width(width)
    , height(height) {
    const auto* settings = Settings_Instance();

    // The headset was told which codec to expect; x264 only makes H.264.
    if (settings->m_codec != ALVR_CODEC_H264) {
        throw std::runtime_error(
            "the software encoder only supports H264. Select H264 as the preferred codec in "
            "ALVR's video settings"
        );
    }
    if (width % 2 != 0 || height % 2 != 0) {
        throw std::runtime_error("the software encoder needs an even resolution");
    }

    try {
        // Host buffer the output image is copied into. Cached memory makes the
        // CPU reads during conversion much faster where the GPU offers it.
        vk::DeviceSize size = vk::DeviceSize(width) * height * 4;
        readbackBuffer = vk_ctx.dev.createBuffer({
            .size = size,
            .usage = vk::BufferUsageFlagBits::eTransferDst,
            .sharingMode = vk::SharingMode::eExclusive,
        });
        auto reqs = vk_ctx.dev.getBufferMemoryRequirements(readbackBuffer);
        u32 typeIndex = findMemoryType(
            vk_ctx,
            reqs.memoryTypeBits,
            vk::MemoryPropertyFlagBits::eHostVisible | vk::MemoryPropertyFlagBits::eHostCached
                | vk::MemoryPropertyFlagBits::eHostCoherent
        );
        if (typeIndex == UINT32_MAX) {
            typeIndex = findMemoryType(
                vk_ctx,
                reqs.memoryTypeBits,
                vk::MemoryPropertyFlagBits::eHostVisible | vk::MemoryPropertyFlagBits::eHostCached
            );
            readbackCoherent = false;
        }
        if (typeIndex == UINT32_MAX) {
            typeIndex = findMemoryType(
                vk_ctx,
                reqs.memoryTypeBits,
                vk::MemoryPropertyFlagBits::eHostVisible | vk::MemoryPropertyFlagBits::eHostCoherent
            );
            readbackCoherent = true;
        }
        if (typeIndex == UINT32_MAX) {
            throw std::runtime_error("no host-visible memory for the software encoder");
        }
        readbackMemory = vk_ctx.dev.allocateMemory({
            .allocationSize = reqs.size,
            .memoryTypeIndex = typeIndex,
        });
        vk_ctx.dev.bindBufferMemory(readbackBuffer, readbackMemory, 0);
        mapped
            = static_cast<const uint8_t*>(vk_ctx.dev.mapMemory(readbackMemory, 0, VK_WHOLE_SIZE));

        cmdPool = vk_ctx.dev.createCommandPool({
            .flags = vk::CommandPoolCreateFlagBits::eResetCommandBuffer,
            .queueFamilyIndex = vk_ctx.meta.queueFamily,
        });
        cmdBuf = vk_ctx.dev.allocateCommandBuffers({
            .commandPool = cmdPool,
            .level = vk::CommandBufferLevel::ePrimary,
            .commandBufferCount = 1,
        })[0];
        fence = vk_ctx.dev.createFence({});

        x264_param_default_preset(&param, "ultrafast", "zerolatency");
        param.pf_log = x264_log;
        param.i_log_level = X264_LOG_WARNING;
        param.b_aud = 0;
        param.b_cabac = settings->m_entropyCoding == ALVR_CABAC;
        param.b_sliced_threads = 1;
        param.i_threads = settings->m_swThreadCount; // 0 lets x264 choose
        param.i_width = width;
        param.i_height = height;
        param.i_csp = X264_CSP_I420;
        param.i_keyint_max = X264_KEYINT_MAX_INFINITE;
        param.b_repeat_headers = 1;
        param.b_annexb = 1;
        param.rc.i_rc_method = X264_RC_ABR;
        param.vui.b_fullrange = 1;
        param.vui.i_colorprim = 1; // BT.709
        param.vui.i_transfer = 13; // sRGB
        param.vui.i_colmatrix = 1; // BT.709

        auto params = GetDynamicEncoderParams();
        if (!params.updated || params.bitrate_bps <= 0) {
            params.bitrate_bps = 30'000'000;
        }
        params.updated = true;
        params.framerate = settings->m_refreshRate;
        SetParams(params);

        switch (settings->m_h264Profile) {
        case ALVR_H264_PROFILE_BASELINE:
            x264_param_apply_profile(&param, "baseline");
            break;
        case ALVR_H264_PROFILE_MAIN:
            x264_param_apply_profile(&param, "main");
            break;
        default:
            x264_param_apply_profile(&param, "high");
            break;
        }

        enc = x264_encoder_open(&param);
        if (!enc) {
            throw std::runtime_error("failed to open the x264 encoder");
        }

        if (x264_picture_alloc(&picture, X264_CSP_I420, width, height) < 0) {
            throw std::runtime_error("failed to allocate the x264 picture");
        }
        pictureAllocated = true;
        x264_picture_init(&pictureOut);

        convertThreads = std::clamp(std::thread::hardware_concurrency() / 2, 1u, 8u);
        Info(
            "Software encoder: %ux%u, x264 threads %d, colour conversion threads %u\n",
            width,
            height,
            param.i_threads,
            convertThreads
        );
    } catch (...) {
        // A throwing constructor skips the destructor.
        release();
        throw;
    }
}

alvr::EncodePipelineSW::~EncodePipelineSW() { release(); }

void alvr::EncodePipelineSW::release() {
    if (enc) {
        x264_encoder_close(enc);
        enc = nullptr;
    }
    if (pictureAllocated) {
        x264_picture_clean(&picture);
        pictureAllocated = false;
    }
    if (fence) {
        vk_ctx.dev.destroy(fence);
        fence = nullptr;
    }
    if (cmdPool) {
        vk_ctx.dev.destroy(cmdPool);
        cmdPool = nullptr;
    }
    if (readbackBuffer) {
        vk_ctx.dev.destroy(readbackBuffer);
        readbackBuffer = nullptr;
    }
    if (readbackMemory) {
        vk_ctx.dev.free(readbackMemory);
        readbackMemory = nullptr;
        mapped = nullptr;
    }
}

void alvr::EncodePipelineSW::readback() {
    cmdBuf.reset();
    cmdBuf.begin(vk::CommandBufferBeginInfo {
        .flags = vk::CommandBufferUsageFlagBits::eOneTimeSubmit,
    });

    // The renderer leaves its output in GENERAL after writing it from a
    // compute shader in an earlier submission.
    vk::ImageMemoryBarrier toCopy {
        .srcAccessMask = vk::AccessFlagBits::eShaderWrite,
        .dstAccessMask = vk::AccessFlagBits::eTransferRead,
        .oldLayout = vk::ImageLayout::eGeneral,
        .newLayout = vk::ImageLayout::eGeneral,
        .srcQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED,
        .dstQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED,
        .image = outputImage,
        .subresourceRange = { vk::ImageAspectFlagBits::eColor, 0, 1, 0, 1 },
    };
    cmdBuf.pipelineBarrier(
        vk::PipelineStageFlagBits::eComputeShader,
        vk::PipelineStageFlagBits::eTransfer,
        {},
        {},
        {},
        toCopy
    );

    vk::BufferImageCopy region {
        .bufferOffset = 0,
        .bufferRowLength = 0,
        .bufferImageHeight = 0,
        .imageSubresource = { vk::ImageAspectFlagBits::eColor, 0, 0, 1 },
        .imageOffset = { 0, 0, 0 },
        .imageExtent = { width, height, 1 },
    };
    cmdBuf.copyImageToBuffer(outputImage, vk::ImageLayout::eGeneral, readbackBuffer, region);

    vk::BufferMemoryBarrier toHost {
        .srcAccessMask = vk::AccessFlagBits::eTransferWrite,
        .dstAccessMask = vk::AccessFlagBits::eHostRead,
        .srcQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED,
        .dstQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED,
        .buffer = readbackBuffer,
        .offset = 0,
        .size = VK_WHOLE_SIZE,
    };
    cmdBuf.pipelineBarrier(
        vk::PipelineStageFlagBits::eTransfer, vk::PipelineStageFlagBits::eHost, {}, {}, toHost, {}
    );
    cmdBuf.end();

    vk::SubmitInfo submit {
        .commandBufferCount = 1,
        .pCommandBuffers = &cmdBuf,
    };
    vk_ctx.useQueue([&](auto& queue) { queue.submit(submit, fence); });
    auto res = vk_ctx.dev.waitForFences(fence, true, UINT64_MAX);
    vk_ctx.dev.resetFences(fence);
    if (res != vk::Result::eSuccess) {
        throw std::runtime_error("software encoder readback wait failed");
    }

    if (!readbackCoherent) {
        vk_ctx.dev.invalidateMappedMemoryRanges(vk::MappedMemoryRange {
            .memory = readbackMemory,
            .offset = 0,
            .size = VK_WHOLE_SIZE,
        });
    }
}

void alvr::EncodePipelineSW::convertToI420() {
    const int ri = bgra ? 2 : 0;
    const int bi = bgra ? 0 : 2;
    const uint32_t srcStride = width * 4;

    uint8_t* yPlane = picture.img.plane[0];
    uint8_t* uPlane = picture.img.plane[1];
    uint8_t* vPlane = picture.img.plane[2];
    const int yStride = picture.img.i_stride[0];
    const int uStride = picture.img.i_stride[1];
    const int vStride = picture.img.i_stride[2];

    // Each worker handles a band of row pairs, since chroma covers 2x2 pixels.
    auto work = [&](uint32_t pairBegin, uint32_t pairEnd) {
        for (uint32_t py = pairBegin; py < pairEnd; ++py) {
            const uint8_t* row0 = mapped + size_t(py * 2) * srcStride;
            const uint8_t* row1 = row0 + srcStride;
            uint8_t* y0 = yPlane + size_t(py * 2) * yStride;
            uint8_t* y1 = y0 + yStride;
            uint8_t* u = uPlane + size_t(py) * uStride;
            uint8_t* v = vPlane + size_t(py) * vStride;

            for (uint32_t x = 0; x < width; x += 2) {
                const uint8_t* p00 = row0 + x * 4;
                const uint8_t* p01 = p00 + 4;
                const uint8_t* p10 = row1 + x * 4;
                const uint8_t* p11 = p10 + 4;

                y0[x] = lumaOf(p00[ri], p00[1], p00[bi]);
                y0[x + 1] = lumaOf(p01[ri], p01[1], p01[bi]);
                y1[x] = lumaOf(p10[ri], p10[1], p10[bi]);
                y1[x + 1] = lumaOf(p11[ri], p11[1], p11[bi]);

                int r = (p00[ri] + p01[ri] + p10[ri] + p11[ri] + 2) >> 2;
                int g = (p00[1] + p01[1] + p10[1] + p11[1] + 2) >> 2;
                int b = (p00[bi] + p01[bi] + p10[bi] + p11[bi] + 2) >> 2;
                u[x / 2] = cbOf(r, g, b);
                v[x / 2] = crOf(r, g, b);
            }
        }
    };

    const uint32_t pairs = height / 2;
    const uint32_t threads = std::min<uint32_t>(convertThreads, pairs);
    if (threads <= 1) {
        work(0, pairs);
        return;
    }
    std::vector<std::thread> pool;
    pool.reserve(threads - 1);
    const uint32_t band = (pairs + threads - 1) / threads;
    for (uint32_t t = 1; t < threads; ++t) {
        uint32_t begin = std::min(pairs, t * band);
        uint32_t end = std::min(pairs, begin + band);
        pool.emplace_back(work, begin, end);
    }
    work(0, std::min(pairs, band));
    for (auto& th : pool) {
        th.join();
    }
}

void alvr::EncodePipelineSW::PushFrame(uint64_t targetTimestampNs, bool idr) {
    readback();
    convertToI420();

    picture.i_type = idr ? X264_TYPE_IDR : X264_TYPE_AUTO;
    picture.i_pts = int64_t(targetTimestampNs);
    pts = picture.i_pts;
    isIdr = idr;

    int nalCount = 0;
    nalSize = x264_encoder_encode(enc, &nal, &nalCount, &picture, &pictureOut);
    if (nalSize < 0) {
        throw std::runtime_error("x264_encoder_encode failed");
    }
    if (nalCount == 0) {
        nalSize = 0;
    }
}

bool alvr::EncodePipelineSW::GetEncoded(FramePacket& packet) {
    if (!nal || nalSize <= 0) {
        return false;
    }
    // x264 lays the NAL payloads of one frame out back to back.
    packet.data = nal[0].p_payload;
    packet.size = nalSize;
    packet.pts = pts;
    packet.isIDR = pictureOut.b_keyframe || isIdr;
    nalSize = 0;
    return true;
}

void alvr::EncodePipelineSW::SetParams(FfiDynamicEncoderParams params) {
    if (!params.updated) {
        return;
    }
    // x264 does not follow fast bitrate changes well; keep the frame rate fixed.
    param.i_fps_num = Settings_Instance()->m_refreshRate;
    param.i_fps_den = 1;
    param.rc.i_bitrate = int(params.bitrate_bps / 1'000 * 1.4); // ABR undershoots
    param.rc.i_vbv_max_bitrate = param.rc.i_bitrate;
    param.rc.i_vbv_buffer_size
        = std::max(1, param.rc.i_bitrate / std::max(1, int(param.i_fps_num)) * 11 / 10);
    param.rc.f_vbv_buffer_init = 0.75f;
    if (enc) {
        x264_encoder_reconfig(enc, &param);
    }
}

int alvr::EncodePipelineSW::GetCodec() { return ALVR_CODEC_H264; }
