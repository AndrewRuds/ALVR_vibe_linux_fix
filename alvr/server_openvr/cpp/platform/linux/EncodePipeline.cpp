#include "EncodePipeline.h"

#include "EncodePipelineNvEnc.h"
#include "EncodePipelineSW.h"
#include "EncodePipelineVAAPI.h"
#include "alvr_server/Logger.h"
#include "alvr_server/bindings.h"
#include "ffmpeg_helper.h"

extern "C" {
#include <libavcodec/avcodec.h>
}

// TODO: Uninheritance this
void alvr::EncodePipeline::SetParams(FfiDynamicEncoderParams params) {
    if (params.updated) {
        encoder_ctx->bit_rate = params.bitrate_bps / params.framerate * 60.0;
        encoder_ctx->framerate = AVRational { 60, 1 };
        encoder_ctx->rc_buffer_size = encoder_ctx->bit_rate / 60.0 * 1.1;
        encoder_ctx->rc_max_rate = encoder_ctx->bit_rate;
        encoder_ctx->rc_initial_buffer_occupancy = encoder_ctx->rc_buffer_size / 4 * 3;
    }
}

std::unique_ptr<alvr::EncodePipeline> alvr::EncodePipeline::Create(
    alvr::VkContext& vk_ctx,
    std::string devicePath,
    VkFrame& input_frame,
    /* VkFrameCtx &vk_frame_ctx,  */ uint32_t width,
    uint32_t height
) {
    using alvr::Vendor;
    auto const vendor = vk_ctx.meta.vendor;
    if (Settings_Instance()->m_forceSwEncoding == false) {
        if (vendor == Vendor::Nvidia) {
            try {
                alvr::HWContext hwCtx(vk_ctx);
                auto nvenc = std::make_unique<alvr::EncodePipelineNvEnc>(
                    hwCtx, vk_ctx, input_frame, width, height
                );
                Info("Using NvEnc encoder");
                return nvenc;
            } catch (std::exception& e) {
                Error(
                    "Failed to create NvEnc encoder: %s\nPlease make sure you have installed CUDA "
                    "runtime.",
                    e.what()
                );
            }
        } else if (vendor == Vendor::Amd || vendor == Vendor::Intel) {
            try {
                alvr::HWContext hwCtx(vk_ctx);
                auto vaapi = std::make_unique<alvr::EncodePipelineVAAPI>(
                    hwCtx, devicePath, vendor, input_frame, width, height
                );
                Info("Using VAAPI encoder");
                return vaapi;
            } catch (std::exception& e) {
                Error(
                    "Failed to create VAAPI encoder: %s\nPlease make sure you have installed VAAPI "
                    "runtime.",
                    e.what()
                );
            }
        } else {
            Info("No hardware encoder is supported for this GPU, using the software encoder");
        }
    }

    // Software fallback. Slow and CPU heavy, but works on any GPU.
    try {
        auto sw = std::make_unique<alvr::EncodePipelineSW>(
            vk_ctx, input_frame.image(), vk::Format(input_frame.format()), width, height
        );
        Info("Using software (x264) encoder");
        return sw;
    } catch (std::exception& e) {
        Error("Failed to create software encoder: %s", e.what());
    }
    return nullptr;
}

alvr::EncodePipeline::~EncodePipeline() {
    av_packet_free(&encoder_packet);
    avcodec_free_context(&encoder_ctx);
}

bool alvr::EncodePipeline::GetEncoded(FramePacket& packet) {
    av_packet_free(&encoder_packet);
    encoder_packet = av_packet_alloc();
    int err = avcodec_receive_packet(encoder_ctx, encoder_packet);
    if (err != 0) {
        av_packet_free(&encoder_packet);
        if (err == AVERROR(EAGAIN)) {
            return false;
        }
        throw alvr::AvException("failed to encode", err);
    }
    packet.data = encoder_packet->data;
    packet.size = encoder_packet->size;
    packet.pts = encoder_packet->pts;
    packet.isIDR = (encoder_packet->flags & AV_PKT_FLAG_KEY) != 0;
    return true;
}

int alvr::EncodePipeline::GetCodec() { return Settings_Instance()->m_codec; }
