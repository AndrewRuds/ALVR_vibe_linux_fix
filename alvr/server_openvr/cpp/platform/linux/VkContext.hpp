#pragma once

#include <algorithm>
#include <cassert>
#include <cstdint>
#include <cstring>
#include <fcntl.h>
#include <functional>
#include <iostream>
#include <optional>
#include <stdexcept>
#include <string_view>
#include <vector>

#include <sys/stat.h>
#include <sys/sysmacros.h>
#include <unistd.h>

#define VULKAN_HPP_NO_CONSTRUCTORS
#include <vulkan/vulkan.h>
#include <vulkan/vulkan.hpp>
#include <vulkan/vulkan_core.h>
#include <vulkan/vulkan_enums.hpp>
#include <vulkan/vulkan_handles.hpp>
#include <vulkan/vulkan_structs.hpp>

#include "utils.hpp"

#include "alvr_server/Logger.h"

extern "C" {
#include <libavutil/log.h>
}

namespace alvr {

// Other covers GPUs without a supported hardware encoder, such as Apple
// Silicon on Asahi Linux; they use the software encoder.
enum class Vendor { Amd, Intel, Nvidia, Other };

class VkContext {
public:
    vk::Instance instance;
    vk::PhysicalDevice physDev;
    vk::Device dev;

    vk::detail::DispatchLoaderDynamic dispatch;

    struct Meta {
        Vendor vendor = Vendor::Other;

        std::vector<char const*> instExtensions;
        std::vector<char const*> devExtensions;

        vk::PhysicalDeviceVulkan12Features feats12;
        vk::PhysicalDeviceFeatures2 feats;

        u32 queueFamily;
        u32 queueIndex;
    } meta;

private:
    vk::Queue queue;

    void sharedInit() {
        auto devProps = physDev.getProperties2();

        if (devProps.properties.vendorID == 0x1002)
            meta.vendor = Vendor::Amd;
        else if (devProps.properties.vendorID == 0x8086)
            meta.vendor = Vendor::Intel;
        else if (devProps.properties.vendorID == 0x10de)
            meta.vendor = Vendor::Nvidia;
        else
            meta.vendor = Vendor::Other;

        dispatch = { instance, vkGetInstanceProcAddr };
    }

public:
    // Indices into vkEnumeratePhysicalDevices, best candidate first.
    //
    // The encoder has to run on the GPU that SteamVR renders on, or the swap
    // textures cannot be imported and the headset gets no picture. Nothing
    // tells us which one that is, so the caller tries these in order and keeps
    // the first that can import the textures. On laptops the integrated GPU is
    // usually listed first while SteamVR runs on the discrete one, so discrete
    // GPUs go first. ALVR_VK_DEVICE_INDEX puts one device at the front.
    static std::vector<u32> candidateDeviceIndices() {
        vk::ApplicationInfo appInfo {
            .pApplicationName = "ALVR",
            .apiVersion = VK_API_VERSION_1_2,
        };
        auto instance = vk::createInstance({ .pApplicationInfo = &appInfo });
        auto physDevs = instance.enumeratePhysicalDevices();

        std::vector<u32> discrete, integrated, other;
        for (u32 i = 0; i < physDevs.size(); ++i) {
            auto props = physDevs[i].getProperties();
            Info("Vulkan device %u: %s\n", i, props.deviceName.data());
            switch (props.deviceType) {
            case vk::PhysicalDeviceType::eDiscreteGpu:
                discrete.push_back(i);
                break;
            case vk::PhysicalDeviceType::eIntegratedGpu:
                integrated.push_back(i);
                break;
            case vk::PhysicalDeviceType::eCpu:
                // llvmpipe and friends can neither import the textures nor encode.
                break;
            default:
                other.push_back(i);
                break;
            }
        }
        auto deviceCount = static_cast<u32>(physDevs.size());
        instance.destroy();

        std::vector<u32> order;
        if (auto env = getenv("ALVR_VK_DEVICE_INDEX")) {
            char* end = nullptr;
            auto idx = strtoul(env, &end, 10);
            if (end != env && *end == '\0' && idx < deviceCount) {
                order.push_back(static_cast<u32>(idx));
            } else {
                Warn("Ignoring invalid ALVR_VK_DEVICE_INDEX=%s\n", env);
            }
        }
        for (auto list : { &discrete, &integrated, &other }) {
            for (auto idx : *list) {
                if (std::find(order.begin(), order.end(), idx) == order.end()) {
                    order.push_back(idx);
                }
            }
        }
        return order;
    }

    explicit VkContext(u32 physDevIndex) {
        std::vector<std::string_view> wantedInstExts = {
            VK_KHR_GET_PHYSICAL_DEVICE_PROPERTIES_2_EXTENSION_NAME,
        };

        auto availInstExts = vk::enumerateInstanceExtensionProperties();
        std::vector<const char*> acquiredInstExts;
        for (auto wantedName : wantedInstExts) {
            auto it = std::find_if(availInstExts.begin(), availInstExts.end(), [&](auto& ext) {
                return wantedName == ext.extensionName;
            });
            if (it != availInstExts.end())
                acquiredInstExts.push_back(wantedName.data());
            else
                assert(false);
        }

        vk::ApplicationInfo appInfo {
            .pApplicationName = "ALVR",
            .apiVersion = VK_API_VERSION_1_2,
        };
        vk::InstanceCreateInfo instanceCI {
            .pApplicationInfo = &appInfo,
            .enabledExtensionCount = static_cast<uint32_t>(acquiredInstExts.size()),
            .ppEnabledExtensionNames = acquiredInstExts.data(),
        };
        instance = vk::createInstance(instanceCI);
        auto physDevs = instance.enumeratePhysicalDevices();
        if (physDevIndex >= physDevs.size()) {
            instance.destroy();
            throw std::runtime_error("Vulkan device index out of range");
        }
        physDev = physDevs[physDevIndex];

        auto queueFamilyProps = physDev.getQueueFamilyProperties();

        std::optional<u32> wantedQueueFamily;

        for (u32 i = 0; i < queueFamilyProps.size(); ++i) {
            auto& props = queueFamilyProps[i];
            bool isGraphics = static_cast<bool>(props.queueFlags & vk::QueueFlagBits::eGraphics);
            bool isCompute = static_cast<bool>(props.queueFlags & vk::QueueFlagBits::eCompute);

            if (isCompute && (!wantedQueueFamily.has_value() || !isGraphics)) {
                wantedQueueFamily = i;
            }
        }
        meta.queueFamily = wantedQueueFamily.value();
        meta.queueIndex = 0;

        std::vector<std::string_view> wantedExts = {
            VK_KHR_EXTERNAL_MEMORY_FD_EXTENSION_NAME,
            VK_KHR_EXTERNAL_SEMAPHORE_FD_EXTENSION_NAME,
            VK_EXT_EXTERNAL_MEMORY_DMA_BUF_EXTENSION_NAME,
            VK_EXT_IMAGE_DRM_FORMAT_MODIFIER_EXTENSION_NAME,
            VK_KHR_EXTERNAL_SEMAPHORE_FD_EXTENSION_NAME,
            VK_EXT_EXTERNAL_MEMORY_HOST_EXTENSION_NAME,
            VK_KHR_PUSH_DESCRIPTOR_EXTENSION_NAME,
            VK_KHR_SAMPLER_YCBCR_CONVERSION_EXTENSION_NAME,
            VK_EXT_PHYSICAL_DEVICE_DRM_EXTENSION_NAME,
            VK_EXT_CALIBRATED_TIMESTAMPS_EXTENSION_NAME,
            VK_KHR_COPY_COMMANDS_2_EXTENSION_NAME,
        };

        auto availExts = physDev.enumerateDeviceExtensionProperties();

        std::vector<char const*> acquiredExts;
        for (auto name : wantedExts) {
            auto it = std::find_if(availExts.begin(), availExts.end(), [&](auto& other) {
                return name == (const char*)other.extensionName;
            });
            if (it != availExts.end())
                acquiredExts.push_back(name.data());
        }

        f32 queuePrio = 1.f;
        vk::DeviceQueueCreateInfo queueCI {
            .queueFamilyIndex = meta.queueFamily,
            .queueCount = 1,
            .pQueuePriorities = &queuePrio,
        };

        meta.feats12 = vk::PhysicalDeviceVulkan12Features {
            .timelineSemaphore = 1,
        };

        meta.feats = vk::PhysicalDeviceFeatures2 {
            .pNext = &meta.feats12,
            .features = {
                .robustBufferAccess = true,
                .samplerAnisotropy = true,
            },
        };
        vk::DeviceCreateInfo devCI {
            .pNext = &meta.feats,
            .queueCreateInfoCount = 1,
            .pQueueCreateInfos = &queueCI,
            .enabledExtensionCount = static_cast<u32>(acquiredExts.size()),
            .ppEnabledExtensionNames = acquiredExts.data(),
        };
        dev = physDev.createDevice(devCI);

        meta.devExtensions = acquiredExts;

        queue = dev.getQueue(wantedQueueFamily.value(), meta.queueIndex);

        sharedInit();
    }

    void useQueue(std::function<void(vk::Queue&)> fn) { fn(queue); }

    void destroy() {
        dev.destroy();
        instance.destroy();
    }
};

}
