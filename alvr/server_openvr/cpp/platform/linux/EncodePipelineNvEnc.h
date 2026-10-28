#pragma once

#include "EncodePipeline.h"
#include "VkContext.hpp"
#include "ffmpeg_helper.h"
#include <functional>
#include <memory>

#if __has_include(<ffnvcodec/dynlink_loader.h>)
#define ALVR_HAVE_CUDA_INTEROP
#include <ffnvcodec/dynlink_loader.h>
#endif

extern "C" struct AVBufferRef;
extern "C" struct AVCodecContext;
extern "C" struct AVFrame;

namespace alvr {

class EncodePipelineNvEnc : public EncodePipeline {
public:
    ~EncodePipelineNvEnc();
    EncodePipelineNvEnc(
        HWContext& vk_ctx, VkContext& v_ctx, VkFrame& input_frame, uint32_t width, uint32_t height
    );

    void PushFrame(uint64_t targetTimestampNs, bool idr) override;
    void SetParams(FfiDynamicEncoderParams params) override;

private:
    VkContext& v_ctx;
    // Declared before vk_frame: make_av_frame takes a reference to this, so
    // it has to outlive the frame built from it.
    VkFrameCtx frame_ctx;
    AVBufferRef* hw_ctx = nullptr;
    std::unique_ptr<AVFrame, std::function<void(AVFrame*)>> vk_frame;
    AVFrame* hw_frame = nullptr;

#ifdef ALVR_HAVE_CUDA_INTEROP
    void importOutputToCuda(VkFrame& input_frame);
    void releaseCuda();

    CudaFunctions* cu = nullptr;
    CUexternalMemory cuExtMem = nullptr;
    CUmipmappedArray cuMipArray = nullptr;
    CUarray cuArray = nullptr;
    size_t copyWidthBytes = 0;
    size_t copyHeight = 0;
#endif
};
}
