#include "EncodePipelineNvEnc.h"
#include "alvr_server/Logger.h"
#include "alvr_server/bindings.h"
#include "ffmpeg_helper.h"
#include <ALVR-common/packet_types.h>
#include <chrono>
#include <memory>

extern "C" {
#include <libavcodec/avcodec.h>
#include <libavutil/opt.h>
}

#ifdef ALVR_HAVE_CUDA_INTEROP
#include <unistd.h>
extern "C" {
#include <libavutil/hwcontext_cuda.h>
}
// Not in the nv-codec-headers copy of the CUDA API.
#ifndef CUDA_EXTERNAL_MEMORY_DEDICATED
#define CUDA_EXTERNAL_MEMORY_DEDICATED 0x1
#endif
#endif

namespace {

const char* encoder(ALVR_CODEC codec) {
    switch (codec) {
    case ALVR_CODEC_H264:
        return "h264_nvenc";
    case ALVR_CODEC_HEVC:
        return "hevc_nvenc";
    case ALVR_CODEC_AV1:
        return "av1_nvenc";
    }
    throw std::runtime_error("invalid codec " + std::to_string(codec));
}

// NvEnc takes an opaque-alpha format, so drop the alpha channel while keeping
// the channel order of the source. Picking the wrong one here is silent: the
// encode succeeds and ships swapped red and blue.
AVPixelFormat nvenc_surface_format(AVPixelFormat inputFormat) {
    switch (inputFormat) {
    case AV_PIX_FMT_BGRA:
        return AV_PIX_FMT_BGR0;
    case AV_PIX_FMT_RGBA:
        return AV_PIX_FMT_RGB0;
    case AV_PIX_FMT_BGR0:
    case AV_PIX_FMT_RGB0:
        return inputFormat;
    default:
        throw std::runtime_error(
            std::string("no NvEnc surface format for input pixel format ")
            + std::to_string((int)inputFormat)
        );
    }
}

void set_hwframe_ctx(AVCodecContext* ctx, AVBufferRef* hw_device_ctx, AVPixelFormat surfaceFormat) {
    AVBufferRef* hw_frames_ref;
    AVHWFramesContext* frames_ctx = NULL;
    int err = 0;

    if (!(hw_frames_ref = av_hwframe_ctx_alloc(hw_device_ctx))) {
        throw std::runtime_error("Failed to create CUDA frame context.");
    }
    frames_ctx = (AVHWFramesContext*)(hw_frames_ref->data);
    frames_ctx->format = AV_PIX_FMT_CUDA;
    frames_ctx->sw_format = surfaceFormat;
    frames_ctx->width = ctx->width;
    frames_ctx->height = ctx->height;
    if ((err = av_hwframe_ctx_init(hw_frames_ref)) < 0) {
        av_buffer_unref(&hw_frames_ref);
        throw alvr::AvException("Failed to initialize CUDA frame context:", err);
    }
    ctx->hw_frames_ctx = av_buffer_ref(hw_frames_ref);
    if (!ctx->hw_frames_ctx)
        err = AVERROR(ENOMEM);

    av_buffer_unref(&hw_frames_ref);
}

// Last target the bitrate manager delivered; see initEncoding.
int64_t last_known_bitrate_bps = 0;

} // namespace
alvr::EncodePipelineNvEnc::EncodePipelineNvEnc(
    HWContext& vk_ctx, VkContext& v_ctx, VkFrame& input_frame, uint32_t width, uint32_t height
)
    : v_ctx(v_ctx)
    , frame_ctx(vk_ctx.avCtx, input_frame.imageInfo()) {
    // A throw here skips the destructor, and a leaked CUDA context keeps a few
    // hundred MB of VRAM, so each failed retry would leave less for the next.
    try {
        // The surface format has to follow the output image's channel order. A
        // mismatch still encodes successfully, with red and blue swapped.
        auto input_frame_ctx = (AVHWFramesContext*)frame_ctx.ctx->data;
        auto surface_format = nvenc_surface_format(input_frame_ctx->sw_format);

        int err;
        vk_frame = input_frame.make_av_frame(frame_ctx);

        err = av_hwdevice_ctx_create_derived(&hw_ctx, AV_HWDEVICE_TYPE_CUDA, vk_ctx.avCtx, 0);
        if (err < 0) {
            throw alvr::AvException("Failed to create a CUDA device:", err);
        }

#ifdef ALVR_HAVE_CUDA_INTEROP
        importOutputToCuda(input_frame);
#endif

        const auto* settings = Settings_Instance();

        auto codec_id = ALVR_CODEC(settings->m_codec);
        const char* encoder_name = encoder(codec_id);
        const AVCodec* codec = avcodec_find_encoder_by_name(encoder_name);
        if (codec == nullptr) {
            throw std::runtime_error(std::string("Failed to find encoder ") + encoder_name);
        }

        encoder_ctx = avcodec_alloc_context3(codec);
        if (not encoder_ctx) {
            throw std::runtime_error("failed to allocate NvEnc encoder");
        }

        switch (codec_id) {
        case ALVR_CODEC_H264:
            switch (settings->m_entropyCoding) {
            case ALVR_CABAC:
                av_opt_set(encoder_ctx->priv_data, "coder", "ac", 0);
                break;
            case ALVR_CAVLC:
                av_opt_set(encoder_ctx->priv_data, "coder", "vlc", 0);
                break;
            }
            break;
        case ALVR_CODEC_HEVC:
            break;
        case ALVR_CODEC_AV1:
            break;
        }

        switch (settings->m_rateControlMode) {
        case ALVR_CBR:
            av_opt_set(encoder_ctx->priv_data, "rc", "cbr", 0);
            break;
        case ALVR_VBR:
            av_opt_set(encoder_ctx->priv_data, "rc", "vbr", 0);
            break;
        }

        if (codec_id == ALVR_CODEC_H264) {
            switch (settings->m_h264Profile) {
            case ALVR_H264_PROFILE_BASELINE:
                av_opt_set(encoder_ctx->priv_data, "profile", "baseline", 0);
                break;
            case ALVR_H264_PROFILE_MAIN:
                av_opt_set(encoder_ctx->priv_data, "profile", "main", 0);
                break;
            default:
            case ALVR_H264_PROFILE_HIGH:
                av_opt_set(encoder_ctx->priv_data, "profile", "high", 0);
                break;
            }
        }

        char preset[] = "p0";
        // replace 0 with preset number
        preset[1] += settings->m_nvencQualityPreset;
        av_opt_set(encoder_ctx->priv_data, "preset", preset, 0);

        if (settings->m_nvencAdaptiveQuantizationMode == 1) {
            av_opt_set_int(encoder_ctx->priv_data, "spatial_aq", 1, 0);
        } else if (settings->m_nvencAdaptiveQuantizationMode == 2) {
            av_opt_set_int(encoder_ctx->priv_data, "temporal_aq", 1, 0);
        }

        if (settings->m_nvencEnableWeightedPrediction) {
            av_opt_set_int(encoder_ctx->priv_data, "weighted_pred", 1, 0);
        }

        av_opt_set_int(encoder_ctx->priv_data, "tune", settings->m_nvencTuningPreset, 0);
        av_opt_set_int(encoder_ctx->priv_data, "zerolatency", 1, 0);
        // Delay isn't actually a delay instead its how many surfaces to encode at a time
        av_opt_set_int(encoder_ctx->priv_data, "delay", 1, 0);
        av_opt_set_int(encoder_ctx->priv_data, "forced-idr", 1, 0);
        // work around ffmpeg default not working for older NVIDIA cards
        av_opt_set_int(encoder_ctx->priv_data, "b_ref_mode", 0, 0);

        encoder_ctx->pix_fmt = AV_PIX_FMT_CUDA;
        encoder_ctx->width = width;
        encoder_ctx->height = height;
        encoder_ctx->time_base = { 1, (int)1e9 };
        encoder_ctx->framerate = AVRational { settings->m_refreshRate, 1 };
        encoder_ctx->sample_aspect_ratio = AVRational { 1, 1 };
        encoder_ctx->max_b_frames = 0;
        encoder_ctx->gop_size = INT16_MAX;
        encoder_ctx->color_range = AVCOL_RANGE_JPEG;
        encoder_ctx->color_primaries = AVCOL_PRI_BT709;
        encoder_ctx->color_trc = AVCOL_TRC_IEC61966_2_1;
        encoder_ctx->colorspace = AVCOL_SPC_BT709;
        // The manager reports a target only once per change, so a rebuild mid
        // session has to reuse the last one.
        auto params = GetDynamicEncoderParams();
        if (!params.updated || params.bitrate_bps <= 0) {
            params.bitrate_bps = last_known_bitrate_bps > 0 ? last_known_bitrate_bps : 30'000'000;
        }
        params.updated = true;
        params.framerate = settings->m_refreshRate;
        Info("NvEnc init bitrate: %.1f Mbps\n", params.bitrate_bps / 1e6);
        SetParams(params);

        set_hwframe_ctx(encoder_ctx, hw_ctx, surface_format);

        err = avcodec_open2(encoder_ctx, codec, NULL);
        if (err < 0) {
            throw alvr::AvException("Cannot open video encoder codec:", err);
        }

        hw_frame = av_frame_alloc();
    } catch (...) {
#ifdef ALVR_HAVE_CUDA_INTEROP
        releaseCuda();
#endif
        av_buffer_unref(&hw_ctx);
        av_frame_free(&hw_frame);
        throw;
    }
}

alvr::EncodePipelineNvEnc::~EncodePipelineNvEnc() {
#ifdef ALVR_HAVE_CUDA_INTEROP
    releaseCuda();
#endif
    av_buffer_unref(&hw_ctx);
    av_frame_free(&hw_frame);
}

#ifdef ALVR_HAVE_CUDA_INTEROP
// ffmpeg's own Vulkan to CUDA transfer is not used: it attaches the CUDA
// import, and a reference to the whole CUDA device and frame pool, to the
// AVVkFrame, and only frees that for frames ffmpeg allocated itself. Ours are
// not, so every encoder rebuild leaked hundreds of MB of VRAM until the next
// one failed with out of memory. It also describes an RGBA image to CUDA as a
// one channel array and drops the dedicated allocation flag. Importing here
// keeps the lifetime with this pipeline.
namespace {
void checkCu(CudaFunctions* cu, CUresult res, char const* what) {
    if (res == CUDA_SUCCESS) {
        return;
    }
    char const* name = nullptr;
    cu->cuGetErrorName(res, &name);
    throw std::runtime_error(std::string(what) + " failed: " + (name ? name : "unknown error"));
}
}

void alvr::EncodePipelineNvEnc::importOutputToCuda(VkFrame& input_frame) {
    if (cuda_load_functions(&cu, nullptr) < 0 || !cu->cuImportExternalMemory) {
        throw std::runtime_error("Failed to load the CUDA driver library (libcuda.so.1)");
    }
    auto* cudaDev = (AVCUDADeviceContext*)((AVHWDeviceContext*)hw_ctx->data)->hwctx;

    AVVkFrame* vkf = input_frame;
    auto info = input_frame.imageInfo();
    copyWidthBytes = size_t(info.extent.width) * 4;
    copyHeight = info.extent.height;

    vk::MemoryGetFdInfoKHR fdInfo {
        .memory = vkf->mem[0],
        .handleType = vk::ExternalMemoryHandleTypeFlagBits::eOpaqueFd,
    };
    int fd = v_ctx.dev.getMemoryFdKHR(fdInfo, v_ctx.dispatch);

    checkCu(cu, cu->cuCtxPushCurrent(cudaDev->cuda_ctx), "cuCtxPushCurrent");
    try {
        CUDA_EXTERNAL_MEMORY_HANDLE_DESC memDesc = {};
        memDesc.type = CU_EXTERNAL_MEMORY_HANDLE_TYPE_OPAQUE_FD;
        memDesc.handle.fd = fd;
        memDesc.size = vkf->size[0];
        // The renderer allocates the output image as a dedicated allocation.
        memDesc.flags = CUDA_EXTERNAL_MEMORY_DEDICATED;
        CUresult res = cu->cuImportExternalMemory(&cuExtMem, &memDesc);
        if (res != CUDA_SUCCESS) {
            // CUDA only takes the fd on success.
            close(fd);
            checkCu(cu, res, "cuImportExternalMemory");
        }

        CUDA_EXTERNAL_MEMORY_MIPMAPPED_ARRAY_DESC arrDesc = {};
        arrDesc.arrayDesc.Width = info.extent.width;
        arrDesc.arrayDesc.Height = info.extent.height;
        arrDesc.arrayDesc.Depth = 0;
        arrDesc.arrayDesc.Format = CU_AD_FORMAT_UNSIGNED_INT8;
        arrDesc.arrayDesc.NumChannels = 4;
        arrDesc.numLevels = 1;
        checkCu(
            cu,
            cu->cuExternalMemoryGetMappedMipmappedArray(&cuMipArray, cuExtMem, &arrDesc),
            "cuExternalMemoryGetMappedMipmappedArray"
        );
        checkCu(
            cu, cu->cuMipmappedArrayGetLevel(&cuArray, cuMipArray, 0), "cuMipmappedArrayGetLevel"
        );
    } catch (...) {
        CUcontext dummy;
        cu->cuCtxPopCurrent(&dummy);
        throw;
    }
    CUcontext dummy;
    cu->cuCtxPopCurrent(&dummy);
}

void alvr::EncodePipelineNvEnc::releaseCuda() {
    if (!cu) {
        return;
    }
    if (hw_ctx && (cuMipArray || cuExtMem)) {
        auto* cudaDev = (AVCUDADeviceContext*)((AVHWDeviceContext*)hw_ctx->data)->hwctx;
        if (cu->cuCtxPushCurrent(cudaDev->cuda_ctx) == CUDA_SUCCESS) {
            if (cuMipArray) {
                cu->cuMipmappedArrayDestroy(cuMipArray);
            }
            if (cuExtMem) {
                cu->cuDestroyExternalMemory(cuExtMem);
            }
            CUcontext dummy;
            cu->cuCtxPopCurrent(&dummy);
        }
    }
    cuMipArray = nullptr;
    cuArray = nullptr;
    cuExtMem = nullptr;
    cuda_free_functions(&cu);
}
#endif

void alvr::EncodePipelineNvEnc::SetParams(FfiDynamicEncoderParams params) {
    if (!params.updated) {
        return;
    }
    // Survives encoder rebuilds; see the note on the variable.
    last_known_bitrate_bps = params.bitrate_bps;
    encoder_ctx->bit_rate = params.bitrate_bps;
    encoder_ctx->framerate = AVRational { int(params.framerate * 1000), 1000 };
    encoder_ctx->rc_buffer_size = encoder_ctx->bit_rate / params.framerate;
    encoder_ctx->rc_max_rate = encoder_ctx->bit_rate;
    encoder_ctx->rc_initial_buffer_occupancy = encoder_ctx->rc_buffer_size;
}

void alvr::EncodePipelineNvEnc::PushFrame(uint64_t targetTimestampNs, bool idr) {
#ifdef ALVR_HAVE_CUDA_INTEROP
    {
        int err = av_hwframe_get_buffer(encoder_ctx->hw_frames_ctx, hw_frame, 0);
        if (err < 0) {
            throw alvr::AvException("Failed to allocate CUDA frame", err);
        }
        auto* cudaDev = (AVCUDADeviceContext*)((AVHWDeviceContext*)hw_ctx->data)->hwctx;

        // The renderer has already waited for its fence, so the image is
        // complete. Waiting for the copy keeps the next render from
        // overwriting it while CUDA still reads.
        CUDA_MEMCPY2D cpy = {};
        cpy.srcMemoryType = CU_MEMORYTYPE_ARRAY;
        cpy.srcArray = cuArray;
        cpy.dstMemoryType = CU_MEMORYTYPE_DEVICE;
        cpy.dstDevice = (CUdeviceptr)hw_frame->data[0];
        cpy.dstPitch = hw_frame->linesize[0];
        cpy.WidthInBytes = copyWidthBytes;
        cpy.Height = copyHeight;

        checkCu(cu, cu->cuCtxPushCurrent(cudaDev->cuda_ctx), "cuCtxPushCurrent");
        CUresult res = cu->cuMemcpy2DAsync(&cpy, cudaDev->stream);
        if (res == CUDA_SUCCESS) {
            res = cu->cuStreamSynchronize(cudaDev->stream);
        }
        CUcontext dummy;
        cu->cuCtxPopCurrent(&dummy);
        if (res != CUDA_SUCCESS) {
            av_frame_unref(hw_frame);
            checkCu(cu, res, "copy of the frame to CUDA");
        }

        hw_frame->pict_type = idr ? AV_PICTURE_TYPE_I : AV_PICTURE_TYPE_NONE;
        hw_frame->pts = targetTimestampNs;

        err = avcodec_send_frame(encoder_ctx, hw_frame);
        av_frame_unref(hw_frame);
        if (err < 0) {
            throw alvr::AvException("avcodec_send_frame failed:", err);
        }
        return;
    }
#endif
    AVVkFrame* vkf = reinterpret_cast<AVVkFrame*>(vk_frame->data[0]);
    vkf->sem_value[0]++;

    VkTimelineSemaphoreSubmitInfo timelineInfo = {};
    timelineInfo.sType = VK_STRUCTURE_TYPE_TIMELINE_SEMAPHORE_SUBMIT_INFO;
    timelineInfo.signalSemaphoreValueCount = 1;
    timelineInfo.pSignalSemaphoreValues = &vkf->sem_value[0];

    vk::PipelineStageFlags waitStage = vk::PipelineStageFlagBits::eBottomOfPipe;

    vk::SubmitInfo submitInfo = {};
    submitInfo.pNext = &timelineInfo;
    // submitInfo.waitSemaphoreCount = 1;
    // submitInfo.pWaitSemaphores = &r->GetOutput().semaphore;
    submitInfo.pWaitDstStageMask = &waitStage;
    submitInfo.signalSemaphoreCount = 1;
    submitInfo.pSignalSemaphores = reinterpret_cast<vk::Semaphore*>(&vkf->sem[0]);
    v_ctx.useQueue([&](auto& queue) { queue.submit(submitInfo); });

    int err = av_hwframe_get_buffer(encoder_ctx->hw_frames_ctx, hw_frame, 0);
    if (err < 0) {
        throw alvr::AvException("Failed to allocate CUDA frame", err);
    }
    err = av_hwframe_transfer_data(hw_frame, vk_frame.get(), 0);
    if (err < 0) {
        throw alvr::AvException("Failed to transfer Vulkan image to CUDA frame", err);
    }

    hw_frame->pict_type = idr ? AV_PICTURE_TYPE_I : AV_PICTURE_TYPE_NONE;
    hw_frame->pts = targetTimestampNs;

    if ((err = avcodec_send_frame(encoder_ctx, hw_frame)) < 0) {
        throw alvr::AvException("avcodec_send_frame failed:", err);
    }

    av_frame_unref(hw_frame);
}
