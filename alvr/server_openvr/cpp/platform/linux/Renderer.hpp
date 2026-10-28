#pragma once

#define VULKAN_HPP_NO_CONSTRUCTORS
#include <cstddef>
#include <vulkan/vulkan.h>
#include <vulkan/vulkan.hpp>
#include <vulkan/vulkan_core.h>
#include <vulkan/vulkan_enums.hpp>
#include <vulkan/vulkan_handles.hpp>
#include <vulkan/vulkan_structs.hpp>

#include "VkContext.hpp"
#include "utils.hpp"

#include "ffmpeg_helper.h"

namespace alvr::render {

constexpr u32 ImageCount = 6;
constexpr usize StagingImgCount = 2;

enum class HandleType {
    None,
    DmaBuf,
    OpaqueFd,
};

struct Image {
    vk::Image image = VK_NULL_HANDLE;
    vk::ImageLayout layout = vk::ImageLayout::eUndefined;
    vk::DeviceMemory memory = VK_NULL_HANDLE;
    vk::ImageView view = VK_NULL_HANDLE;

    void destroy(VkContext const& ctx) {
        ctx.dev.destroy(view);
        ctx.dev.destroy(image);
        ctx.dev.free(memory);
    }
};

struct Output {
    Image image;
    DrmImage drm;
    // VkSemaphore semaphore;
    VkImageCreateInfo imageCI;
    VkDeviceSize size;

    void destroy(VkContext const& ctx) {
        if (drm.fd >= 0) {
            close(drm.fd);
            drm.fd = -1;
        }
        image.destroy(ctx);
    }
};

struct PipelineCreateInfo {
    std::vector<u8> shaderData;
    std::vector<vk::SpecializationMapEntry> specs;
    std::vector<u8> specData;
};

struct RendererCreateInfo {
    vk::Format format;
    vk::Extent2D inputEyeExtent;
    vk::Extent2D outputExtent;
    std::array<int, ImageCount> inputImgFds;
};

// UNORM twin of an sRGB swap texture format; other formats map to themselves.
// Everything after the input copy, the output image included, uses this.
inline vk::Format unormFormatOf(vk::Format format) {
    if (format == vk::Format::eR8G8B8A8Srgb)
        return vk::Format::eR8G8B8A8Unorm;
    if (format == vk::Format::eB8G8R8A8Srgb)
        return vk::Format::eB8G8R8A8Unorm;
    return format;
}

// True when this device can import SteamVR's swap textures, i.e. it is the GPU
// SteamVR renders on. Leaves createInfo.inputImgFds untouched.
bool canImportInputImages(VkContext const& ctx, RendererCreateInfo const& createInfo);

// Push-constant payload for the rotational reprojection fold in the sampling
// shaders. Field offsets must match the GLSL block (0/64/80/96). enabled == 0
// reproduces the unwarped sampling exactly, so a zero-initialized struct is a
// correct passthrough.
struct WarpParams {
    float rotation[16] = {}; // column-major mat4, rotation in the upper-left 3x3
    float leftTans[4] = {}; // tanLeft, tanRight, tanUp, tanDown
    float rightTans[4] = {};
    u32 enabled = 0;
    u32 _padding[3] = {};
};
static_assert(sizeof(WarpParams) == 112);
static_assert(offsetof(WarpParams, rotation) == 0);
static_assert(offsetof(WarpParams, leftTans) == 64);
static_assert(offsetof(WarpParams, rightTans) == 80);
static_assert(offsetof(WarpParams, enabled) == 96);

// Per-eye foveation center shifts for ffr.comp, updated every frame.
struct FoveationCenterShifts {
    float left[2] = {};
    float right[2] = {};
};
static_assert(sizeof(FoveationCenterShifts) == 16);

// The push-constant range every pass shares. The warp sits at offset 0 so
// quad.comp can declare only that prefix; ffr.comp declares the whole block
// (centers at 112/120). Shaders that declare neither ignore it.
struct PushConstants {
    WarpParams warp;
    FoveationCenterShifts centerShifts;
};
static_assert(sizeof(PushConstants) == 128);
static_assert(offsetof(PushConstants, centerShifts) == 112);

namespace detail {

    class RenderPipeline {
        vk::ShaderModule shader;
        vk::PipelineLayout pipeLayout;
        vk::Pipeline pipe;

    public:
        RenderPipeline(
            VkContext const& ctx, vk::DescriptorSetLayout& layout, PipelineCreateInfo& pipelineCI
        );

        void render(
            VkContext const& ctx,
            vk::CommandBuffer cmdBuf,
            vk::ImageView in,
            vk::ImageView out,
            vk::Extent2D outSize,
            PushConstants const& pushConstants
        );

        void destroy(VkContext const& ctx);
    };

}

class Renderer {
    vk::Extent2D eyeExtent;
    vk::Extent2D outExtent;

    Image inputImages[ImageCount];
    Image stagingImgs[StagingImgCount];
    Output output;

    vk::QueryPool timestampPool;
    vk::CommandPool cmdPool;
    vk::CommandBuffer cmdBuf;
    vk::Sampler sampler;
    vk::DescriptorSetLayout descLayout;
    vk::Fence fence;

    std::vector<detail::RenderPipeline> pipes;

    // vk::Semaphore renderFinishedSem;

    void createInputs(VkContext const& vkCtx, RendererCreateInfo& createInfo);

public:
    Renderer(
        VkContext const& vkCtx,
        RendererCreateInfo& createInfo,
        std::vector<PipelineCreateInfo> pipeCIs
    );

    // Swaps in a new set of SteamVR swap textures, keeping the pipelines and
    // the output image, so the encoder reading that image can stay as is.
    // Only valid when the pass chain does not depend on the input size and
    // unormFormatOf(format) is unchanged. Takes ownership of the fds.
    void replaceInputs(VkContext const& ctx, RendererCreateInfo& createInfo);

    // TODO: Import output (somehow?)

    // NOTE: Use the output immediately afterwards, as this synchronizes to the end of gpu
    // operations
    // waitFds, when given, are sync_file fds imported as temporary semaphores
    // the submission waits on before the eye copies. Ownership transfers to
    // this call: a successful import hands the fd to Vulkan, a failed one is
    // closed here.
    void render(VkContext& vkCtx, u32 leftIdx, u32 rightIdx, int const waitFds[2] = nullptr);

    WarpParams warpParams {};

    FoveationCenterShifts foveationCenterShifts {};

    Output getOutput() { return output; }

    void destroy(VkContext const& ctx);
};

}
