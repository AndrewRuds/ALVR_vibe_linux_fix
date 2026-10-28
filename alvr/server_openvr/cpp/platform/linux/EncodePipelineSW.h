#pragma once

#include "EncodePipeline.h"
#include "VkContext.hpp"

#include <chrono>
#include <cstdint>
#include <vector>

extern "C" {
#include <x264.h>
}

namespace alvr {

// CPU encoding with x264, for GPUs without a supported hardware encoder (for
// example Apple Silicon on Asahi Linux) or when software encoding is forced.
// Each frame is copied from the renderer's output image into host memory,
// converted to I420 on several threads and encoded as H.264.
class EncodePipelineSW : public EncodePipeline {
public:
    EncodePipelineSW(
        VkContext& vk_ctx,
        vk::Image outputImage,
        vk::Format outputFormat,
        uint32_t width,
        uint32_t height
    );
    ~EncodePipelineSW();

    void PushFrame(uint64_t targetTimestampNs, bool idr) override;
    bool GetEncoded(FramePacket& packet) override;
    void SetParams(FfiDynamicEncoderParams params) override;
    int GetCodec() override;

private:
    void release();
    void readback();
    void convertToI420();
    void reportStats(double readbackMs, double convertMs, double encodeMs);

    struct Stats {
        uint64_t frames = 0;
        double readbackMs = 0, convertMs = 0, encodeMs = 0;
        uint64_t bytes = 0;
        std::chrono::steady_clock::time_point windowStart {};
        double windowStartCpu = 0.0;
    } stats;

    VkContext& vk_ctx;
    vk::Image outputImage;
    bool bgra;
    uint32_t width;
    uint32_t height;

    vk::Buffer readbackBuffer;
    vk::DeviceMemory readbackMemory;
    bool readbackCoherent = true;
    const uint8_t* mapped = nullptr;
    vk::CommandPool cmdPool;
    vk::CommandBuffer cmdBuf;
    vk::Fence fence;

    x264_t* enc = nullptr;
    x264_param_t param;
    x264_picture_t picture;
    bool pictureAllocated = false;
    x264_picture_t pictureOut;
    x264_nal_t* nal = nullptr;
    int nalSize = 0;
    int64_t pts = 0;
    bool isIdr = false;
    unsigned convertThreads = 1;
};

}
