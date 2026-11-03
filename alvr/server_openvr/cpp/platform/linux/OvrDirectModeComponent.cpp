#define _GNU_SOURCE
#include <link.h>
#include <stdio.h>
#include <sys/socket.h>
#include <sys/un.h>

#include <cstdlib>
#include <iostream>
#include <unistd.h>

#include "OvrDirectModeComponent.h"

#include "alvr_server/Logger.h"
#include "alvr_server/include/openvr_math.h"
#include <cmath>
#include <cstring>
#include <vector>

#include <linux/dma-buf.h>
#include <poll.h>
#include <sys/ioctl.h>

// Multiples of the frame interval. A job older than this when the worker picks
// it up is from a stall, and the swap texture sets are only three deep, so the
// slot is close to its next writer.
constexpr double JobAgeLimitFrames = 1.5;

// Consecutive writer-fence poll timeouts before the GPU wait disarms for the
// session, so a permanently wedged writer degrades the stream once instead of
// taxing every frame.
constexpr uint32_t GpuWaitDisarmThreshold = 30;

// Cap on the writer-fence poll, in frame intervals. The render submission
// waits on these fences unboundedly while holding the encoder mutex, which
// Present's rebuild paths also take, so an unsignalled fence must be caught
// here or it wedges SteamVR's frame loop through the worker built to keep it
// free.
constexpr int64_t GpuWaitPollCapFrames = 2;

// Guard sleep before render, ms. The compositor's own scene writes carry only
// detached pre-signaled placeholder fences, so the GPU-side fence wait cannot
// see them; this bounds how fresh such a write can be when the read starts.
// Runs on the worker, so it costs pipeline depth, not compositor-thread
// budget.
constexpr int GuardSleepMs = 4;

// The negotiated refresh rate expressed as a frame period.
static std::chrono::nanoseconds frameInterval() {
    return std::chrono::nanoseconds((int64_t)(1e9 / Settings_Instance()->m_refreshRate));
}

// Export the sync file a reader must wait on (the writer's fences) from a
// dma-buf fd. Returns the sync_file fd, or -1 when there is nothing to wait
// on or the kernel lacks the ioctl. The caller owns the returned fd.
static int exportReadSyncFile(int dmabufFd) {
#if defined(DMA_BUF_IOCTL_EXPORT_SYNC_FILE)
    struct dma_buf_export_sync_file exportInfo = {};
    exportInfo.flags = DMA_BUF_SYNC_READ;
    exportInfo.fd = -1;
    if (ioctl(dmabufFd, DMA_BUF_IOCTL_EXPORT_SYNC_FILE, &exportInfo) != 0) {
        return -1;
    }
    return exportInfo.fd;
#else
    (void)dmabufFd;
    return -1;
#endif
}

// Close whichever of a job's fence fds are open. Every path that drops a
// job without presenting it owes this call.
static void closeWaitFds(int fds[2]) {
    for (int e = 0; e < 2; ++e) {
        if (fds[e] >= 0) {
            close(fds[e]);
            fds[e] = -1;
        }
    }
}

// Loose on purpose: it rejects zeroed and uninitialized poses, which miss by
// orders of magnitude, not accumulated rounding drift.
constexpr double UnitQuatNormSqTolerance = 0.1;

static bool IsUnitQuat(double w, double x, double y, double z) {
    double normSq = w * w + x * x + y * y + z * z;
    return normSq > 1.0 - UnitQuatNormSqTolerance && normSq < 1.0 + UnitQuatNormSqTolerance;
}

// Fill the warp push constants for a rotational reprojection from
// renderOrientation to latestOrientation. The shaders sample backwards, so the
// rotation they need is source-from-destination, R_render^T * R_latest; with
// the Hamilton product convention of openvr_math.h, conj(q_render) * q_latest
// composes the same way. Both quaternions are world-from-head, so the world
// frame cancels and the playspace transform never enters. Returns false, with
// params left disabled, on a degenerate FOV or a non-unit orientation.
static bool BuildWarpParams(
    const vr::HmdQuaternion_t& renderOrientation,
    const FfiQuat& latestOrientation,
    const FfiFov& leftFov,
    const FfiFov& rightFov,
    alvr::render::WarpParams& params
) {
    float leftTans[4]
        = { tanf(leftFov.left), tanf(leftFov.right), tanf(leftFov.up), tanf(leftFov.down) };
    float rightTans[4]
        = { tanf(rightFov.left), tanf(rightFov.right), tanf(rightFov.up), tanf(rightFov.down) };
    if (leftTans[1] - leftTans[0] == 0.f || rightTans[1] - rightTans[0] == 0.f
        || leftTans[2] - leftTans[3] == 0.f || rightTans[2] - rightTans[3] == 0.f) {
        return false;
    }
    if (!IsUnitQuat(
            renderOrientation.w, renderOrientation.x, renderOrientation.y, renderOrientation.z
        )
        || !IsUnitQuat(
            latestOrientation.w, latestOrientation.x, latestOrientation.y, latestOrientation.z
        )) {
        return false;
    }

    vr::HmdQuaternion_t qLatest
        = { latestOrientation.w, latestOrientation.x, latestOrientation.y, latestOrientation.z };
    vr::HmdQuaternion_t qDelta = vrmath::quaternionConjugate(renderOrientation) * qLatest;

    vr::HmdMatrix34_t rot;
    HmdMatrix_QuatToMat(qDelta.w, qDelta.x, qDelta.y, qDelta.z, &rot);

    // GLSL mat4 is column-major: rotation[col * 4 + row].
    for (int row = 0; row < 4; ++row) {
        for (int col = 0; col < 4; ++col) {
            params.rotation[col * 4 + row]
                = (row < 3 && col < 3) ? rot.m[row][col] : (float)(row == col);
        }
    }
    memcpy(params.leftTans, leftTans, sizeof(leftTans));
    memcpy(params.rightTans, rightTans, sizeof(rightTans));
    params.enabled = 1;
    return true;
}

void OvrDirectModeComponent::SetViewParams(const FfiViewParams params[2]) {
    std::unique_lock<std::mutex> lock(m_viewParamsMutex);
    m_viewParams[0] = params[0];
    m_viewParams[1] = params[1];
    m_viewParamsValid = true;
}

OvrDirectModeComponent::OvrDirectModeComponent(std::shared_ptr<PoseHistory> poseHistory)
    : m_poseHistory(poseHistory)
    , m_submitLayer(0) {
    m_encodeWorker = std::thread(&OvrDirectModeComponent::EncodeWorkerLoop, this);
    StartVsyncAnnouncer();
}

OvrDirectModeComponent::~OvrDirectModeComponent() {
    // Stops before the encode worker: the announcer's grid is what
    // PostPresent holds against, and a hold outliving the worker leaves a
    // present waiting on a frame nothing will encode.
    StopVsyncAnnouncer();
    {
        std::lock_guard<std::mutex> lock(m_jobMutex);
        m_workerExit = true;
        m_jobPending = false;
    }
    m_jobCv.notify_one();
    if (m_encodeWorker.joinable()) {
        m_encodeWorker.join();
    }
    // The worker is gone; whatever job was pending will never present.
    closeWaitFds(m_job.waitFds);
    for (int fd : m_syncFds) {
        if (fd >= 0) {
            close(fd);
        }
    }
}

void OvrDirectModeComponent::DrainPendingJob() {
    std::lock_guard<std::mutex> lock(m_jobMutex);
    if (m_jobPending) {
        closeWaitFds(m_job.waitFds);
    }
    m_jobPending = false;
}

void OvrDirectModeComponent::EncodeWorkerLoop() {
    while (true) {
        FrameJob job;
        {
            std::unique_lock<std::mutex> lock(m_jobMutex);
            m_jobCv.wait(lock, [this] { return m_jobPending || m_workerExit; });
            if (m_workerExit) {
                break;
            }
            job = m_job;
            m_jobPending = false;
        }

        auto const ageLimit = std::chrono::duration_cast<std::chrono::nanoseconds>(
            frameInterval() * JobAgeLimitFrames
        );
        if (std::chrono::steady_clock::now() - job.enqueueTime > ageLimit) {
            closeWaitFds(job.waitFds);
            continue;
        }

        // Bounded gate on the writer's fences, before the encoder lock. On
        // timeout close the fds and present unguarded instead of wedging.
        if (job.waitFds[0] >= 0 || job.waitFds[1] >= 0) {
            bool dropFds = m_gpuWaitPollDisarmed;
            if (!dropFds) {
                struct pollfd pfds[2] = {};
                nfds_t count = 0;
                for (int e = 0; e < 2; ++e) {
                    if (job.waitFds[e] >= 0) {
                        pfds[count].fd = job.waitFds[e];
                        pfds[count].events = POLLIN;
                        ++count;
                    }
                }
                auto const capNs = frameInterval() * GpuWaitPollCapFrames;
                auto const deadline = std::chrono::steady_clock::now() + capNs;
                bool allSignaled = false;
                while (!allSignaled) {
                    int remainingMs = (int)std::chrono::duration_cast<std::chrono::milliseconds>(
                                          deadline - std::chrono::steady_clock::now()
                    )
                                          .count();
                    if (remainingMs < 0) {
                        break;
                    }
                    if (poll(pfds, count, remainingMs) <= 0) {
                        break;
                    }
                    allSignaled = true;
                    for (nfds_t i = 0; i < count; ++i) {
                        if ((pfds[i].revents & POLLIN) == 0) {
                            allSignaled = false;
                        }
                    }
                }
                if (allSignaled) {
                    m_gpuWaitConsecTimeouts = 0;
                } else {
                    dropFds = true;
                    if (++m_gpuWaitConsecTimeouts >= GpuWaitDisarmThreshold) {
                        m_gpuWaitPollDisarmed = true;
                        Info(
                            "Encode worker: writer fences timed out %u frames in a row; "
                            "GPU wait disarmed for this session\n",
                            m_gpuWaitConsecTimeouts
                        );
                    }
                }
            }
            if (dropFds) {
                closeWaitFds(job.waitFds);
            }
        }

        if (GuardSleepMs > 0) {
            std::this_thread::sleep_for(std::chrono::milliseconds(GuardSleepMs));
        }

        try {
            std::lock_guard<std::mutex> encLock(m_encMutex);

            if (job.generation != m_encGeneration) {
                closeWaitFds(job.waitFds);
                continue;
            }

            // Rotationally reproject the frame from the pose it was rendered
            // with to the newest tracking pose, then stamp it with that pose's
            // timestamp: after the warp the newest pose is the image's pose,
            // and the client reprojects from whatever the stamp names. Moving
            // one without the other is what breaks world lock. Sampled here
            // rather than in Present so the target is as fresh as possible.
            uint64_t stampTimestampNs = job.targetTimestampNs;
            alvr::render::WarpParams warpParams {};
            if (enc.warpCapable()) {
                auto latest = m_poseHistory->GetLatestPose();

                FfiViewParams viewParams[2];
                bool viewParamsValid;
                {
                    std::unique_lock<std::mutex> vpLock(m_viewParamsMutex);
                    viewParams[0] = m_viewParams[0];
                    viewParams[1] = m_viewParams[1];
                    viewParamsValid = m_viewParamsValid;
                }

                if (latest && viewParamsValid
                    && BuildWarpParams(
                        job.renderOrientation,
                        latest->motion.pose.orientation,
                        viewParams[0].fov,
                        viewParams[1].fov,
                        warpParams
                    )) {
                    stampTimestampNs = latest->targetTimestampNs;
                }
            }

            std::vector<alvr::LayerRef> layerRefs;
            bool const composeLayers = PrepareLayers(job, layerRefs);
            enc.present(
                job.leftIdx,
                job.rightIdx,
                stampTimestampNs,
                warpParams,
                job.waitFds,
                composeLayers ? &layerRefs : nullptr
            );
        } catch (std::exception const& e) {
            // An exception escaping a thread's start function is
            // std::terminate, which takes vrserver with it. Same recovery as
            // the compositor-thread paths: log, tear the encoder down, go
            // idle until the next client connect.
            // The job's fds may be partially consumed by the throw point, so
            // they are deliberately not closed here.
            Error("Encode worker: %s. Tearing down encoder and going idle.\n", e.what());
            std::lock_guard<std::mutex> encLock(m_encMutex);
            // The try block's lock released when the exception left it, so a
            // rebuild may have swapped the encoder in between. A moved
            // generation means this failure belongs to an encoder that no
            // longer exists; leave the new one alone.
            if (job.generation != m_encGeneration) {
                continue;
            }
            m_encGeneration++;
            try {
                enc.shutdown();
            } catch (std::exception const& e2) {
                // Teardown of already-broken state must not become the
                // std::terminate this handler exists to prevent.
                Error("Encode worker: teardown also failed: %s\n", e2.what());
            }
            m_encoderState = EncoderState::Idle;
        }
    }
}

void OvrDirectModeComponent::CreateSwapTextureSet(
    uint32_t unPid,
    const SwapTextureSetDesc_t* pSwapTextureSetDesc,
    SwapTextureSet_t* pOutSwapTextureSet
) {
    Info(
        "CreateSwapTextureSet pid=%d Format=%d %dx%d SampleCount=%d\n",
        unPid,
        pSwapTextureSetDesc->nFormat,
        pSwapTextureSetDesc->nWidth,
        pSwapTextureSetDesc->nHeight,
        pSwapTextureSetDesc->nSampleCount
    );

    // Report an empty set unless every texture is created; the flags are
    // ours to fill in and were left uninitialized before.
    pOutSwapTextureSet->unTextureFlags = 0;
    for (auto& handle : pOutSwapTextureSet->rSharedTextureHandles) {
        handle = 0;
    }

    ProcessResource* processResource = new ProcessResource();
    processResource->textDesc = *pSwapTextureSetDesc;
    processResource->pid = unPid;
    m_swapchainIndices[processResource] = 0;

    {
        auto pid = getpid();
        Info("VrServer PID %d\n", pid);
    }

    uint32_t usageFlags = static_cast<uint32_t>(
        vk::ImageUsageFlagBits::eTransferSrc | vk::ImageUsageFlagBits::eSampled
        | vk::ImageUsageFlagBits::eInputAttachment
    );

    uint32_t creationFlags = static_cast<uint32_t>(vk::ImageCreateFlagBits::eMutableFormat);

    for (int i = 0; i < 3; i++) {
        vr::SharedTextureHandle_t myHandle = 0;
        bool success = vr::VRIPCResourceManager()->NewSharedVulkanImage(
            pSwapTextureSetDesc->nFormat,
            pSwapTextureSetDesc->nWidth,
            pSwapTextureSetDesc->nHeight,
            true,
            false,
            // No compute access: sRGB formats cannot be storage images, and
            // with it SteamVR's own scene arrives as linear values.
            false,
            1,
            1,
            creationFlags,
            usageFlags,
            &myHandle
        );

        if (!success) {
            Error("VRCIPCResourceManager: Failed to create shared texture\n");
            CleanupProcessResource(processResource);
            for (auto& handle : pOutSwapTextureSet->rSharedTextureHandles) {
                handle = 0;
            }
            return;
        }

        uint64_t ipcHandle = 0;
        vr::VRIPCResourceManager()->RefResource(myHandle, &ipcHandle);

        int fd = 0;
        auto ret = vr::VRIPCResourceManager()->ReceiveSharedFd(ipcHandle, &fd);
        if (ret == false) {
            Error("Failed to get fd for texture\n");
            vr::VRIPCResourceManager()->UnrefResource(myHandle);
            CleanupProcessResource(processResource);
            for (auto& handle : pOutSwapTextureSet->rSharedTextureHandles) {
                handle = 0;
            }
            return;
        }

        processResource->fds[i] = fd;
        processResource->sharedHandles[i] = myHandle;

        m_handleMap.insert(
            std::make_pair(processResource->sharedHandles[i], std::make_pair(processResource, i))
        );
        pOutSwapTextureSet->rSharedTextureHandles[i] = myHandle;
        Info("Created Texture %d %p\n", i, processResource->sharedHandles[i]);
    }
}

/** Used to textures created using CreateSwapTextureSet.  Only one of the set's handles needs to be
 * used to destroy the entire set. */
void OvrDirectModeComponent::DestroySwapTextureSet(vr::SharedTextureHandle_t sharedTextureHandle) {
    Info("DestroySwapTextureSet %p\n", sharedTextureHandle);

    m_presentMutex.lock();
    auto id = m_handleMap.find(sharedTextureHandle);
    if (id != m_handleMap.end()) {
        ProcessResource* p = id->second.first;
        CleanupProcessResource(p);
    } else {
        Debug("Requested to destroy not managing texture. handle:%p\n", sharedTextureHandle);
    }
    m_presentMutex.unlock();
}

/** Used to purge all texture sets for a given process. */
void OvrDirectModeComponent::DestroyAllSwapTextureSets(uint32_t unPid) {
    Info("DestroyAllSwapTextureSets pid=%d\n", unPid);

    m_presentMutex.lock();
    std::vector<ProcessResource*> resourcesToDestroy;
    for (auto it = m_handleMap.begin(); it != m_handleMap.end(); ++it) {
        if (it->second.first->pid == unPid && it->second.second == 0) {
            resourcesToDestroy.push_back(it->second.first);
        }
    }
    for (auto* p : resourcesToDestroy) {
        CleanupProcessResource(p);
    }
    m_presentMutex.unlock();
}

void OvrDirectModeComponent::CleanupProcessResource(ProcessResource* processResource) {
    {
        std::lock_guard<std::mutex> lock(m_evictMutex);
        for (int i = 0; i < 3; i++) {
            if (processResource->sharedHandles[i]) {
                m_pendingEvictions.push_back(processResource->sharedHandles[i]);
            }
        }
    }
    for (int i = 0; i < 3; i++) {
        if (processResource->sharedHandles[i]) {
            vr::VRIPCResourceManager()->UnrefResource(processResource->sharedHandles[i]);
            m_handleMap.erase(processResource->sharedHandles[i]);
            processResource->sharedHandles[i] = 0;
        }
        if (processResource->fds[i] >= 0) {
            close(processResource->fds[i]);
            processResource->fds[i] = -1;
        }
    }
    m_swapchainIndices.erase(processResource);
    delete processResource;
}

/** After Present returns, calls this to get the next index to use for rendering. */
void OvrDirectModeComponent::GetNextSwapTextureSetIndex(
    vr::SharedTextureHandle_t sharedTextureHandles[2], uint32_t (*pIndices)[2]
) {
    Debug("OvrDirectModeComponent::GetNextSwapTextureSetIndex");

    m_presentMutex.lock();
    for (int eye = 0; eye < 2; eye++) {
        auto it = m_handleMap.find(sharedTextureHandles[eye]);
        if (it == m_handleMap.end()) {
            // A set this component does not hold (already destroyed, e.g.
            // while SteamVR shuts down). Leaving the index unset hands the
            // compositor an uninitialized value to index its set with.
            if (pIndices) {
                (*pIndices)[eye] = 0;
            }
            continue;
        }
        auto& idx = m_swapchainIndices[it->second.first];
        idx = (idx + 1) % 3;
        if (pIndices) {
            (*pIndices)[eye] = idx;
        }
    }
    m_presentMutex.unlock();
}

/** Call once per layer to draw for this frame.  One shared texture handle per eye.  Textures must
 * be created using CreateSwapTextureSet and should be alternated per frame.  Call Present once all
 * layers have been submitted. */
void OvrDirectModeComponent::SubmitLayer(const SubmitLayerPerEye_t (&perEye)[2]) {
    m_presentMutex.lock();

    auto pPose = &perEye[0].mHmdPose; // TODO: are both poses the same? Name HMD suggests yes.

    if (m_submitLayer == 0) {
        // Detect FrameIndex of submitted frame by pPose.
        // This is important part to achieve smooth headtracking.
        // We search for history of TrackingInfo and find the TrackingInfo which have nearest matrix
        // value.

        auto pose = m_poseHistory->GetBestPoseMatch(*pPose);
        if (pose) {
            // found the frameIndex
            m_prevTargetTimestampNs = m_targetTimestampNs;
            m_targetTimestampNs = pose->targetTimestampNs;

            m_prevFramePoseRotation = m_framePoseRotation;
            m_framePoseRotation.x = pose->motion.pose.orientation.x;
            m_framePoseRotation.y = pose->motion.pose.orientation.y;
            m_framePoseRotation.z = pose->motion.pose.orientation.z;
            m_framePoseRotation.w = pose->motion.pose.orientation.w;
        } else {
            m_targetTimestampNs = 0;
            m_framePoseRotation = HmdQuaternion_Init(0.0, 0.0, 0.0, 0.0);
        }
    }
    if (m_submitLayer < MAX_LAYERS) {
        m_submitLayers[m_submitLayer][0] = perEye[0];
        m_submitLayers[m_submitLayer][1] = perEye[1];
        m_submitLayer++;
    } else {
        Warn("Too many layers submitted!\n");
    }

    m_presentMutex.unlock();
}

/** Submits queued layers for display. */
void OvrDirectModeComponent::Present(vr::SharedTextureHandle_t syncTexture) {
    int const layerCount = std::min(m_submitLayer, MAX_LAYERS);
    if (layerCount != m_loggedLayerCount) {
        Info("SteamVR is submitting %d layer(s) per frame\n", layerCount);
        m_loggedLayerCount = layerCount;
    }
    m_submitLayer = 0;

    switch (m_encoderState.load()) {
    case EncoderState::RebuildRequested: {
        Info("Rebuilding encoder from negotiated settings\n");
        DrainPendingJob();
        std::lock_guard<std::mutex> encLock(m_encMutex);
        m_encGeneration++;
        enc.shutdown();
        layer0Texts.fill(0);
        m_encoderState = EncoderState::Streaming;
        break;
    }
    case EncoderState::ShutdownRequested: {
        DrainPendingJob();
        std::lock_guard<std::mutex> encLock(m_encMutex);
        m_encGeneration++;
        enc.shutdown();
        layer0Texts.fill(0);
        m_encoderState = EncoderState::Idle;
        return;
    }
    // Before a client connects there are no negotiated settings, so building
    // an encoder here would use wrong defaults and fail noisily. Do nothing
    // until streaming starts.
    case EncoderState::Idle:
        return;
    case EncoderState::Streaming:
        break;
    }

    std::optional<u32> leftIdx;
    std::optional<u32> rightIdx;

    for (u32 i = 0; i < layer0Texts.size(); ++i) {
        if (layer0Texts[i] == m_submitLayers[0][0].hTexture)
            leftIdx = i;

        if (layer0Texts[i] == m_submitLayers[0][1].hTexture)
            rightIdx = i;
    }

    if (!leftIdx.has_value() || !rightIdx.has_value()) {
        auto leftIt = m_handleMap.find(m_submitLayers[0][0].hTexture);
        auto rightIt = m_handleMap.find(m_submitLayers[0][1].hTexture);

        if (leftIt == m_handleMap.end() || rightIt == m_handleMap.end()) {
            Error(
                "Textures not found in handle map %llu, %llu\n",
                m_submitLayers[0][0].hTexture,
                m_submitLayers[0][1].hTexture
            );
            return;
        }

        std::array<int, 6> fds;

        // Vulkan takes ownership of an fd when it imports it, so the copies kept
        // in the handle map are only good for a single import. Hand it a dup
        // every time or the next rebuild fails with InvalidExternalHandle.
        for (u32 i = 0; i < 3; ++i) {
            layer0Texts[i] = leftIt->second.first->sharedHandles[i];
            fds[i] = dup(leftIt->second.first->fds[i]);
        }
        for (u32 i = 0; i < 3; ++i) {
            layer0Texts[i + 3] = rightIt->second.first->sharedHandles[i];
            fds[i + 3] = dup(rightIt->second.first->fds[i]);
        }

        for (u32 i = 0; i < fds.size(); ++i) {
            if (fds[i] == -1) {
                Error("Could not duplicate texture fd %u, skipping encoder setup\n", i);
                for (u32 j = 0; j < fds.size(); ++j) {
                    if (fds[j] != -1)
                        close(fds[j]);
                }
                layer0Texts.fill(0);
                return;
            }
        }

        // Keep a second dup of each texture fd for exporting write fences at
        // Present; Vulkan consumes the import dups below.
        for (u32 i = 0; i < fds.size(); ++i) {
            if (m_syncFds[i] >= 0) {
                close(m_syncFds[i]);
            }
            m_syncFds[i] = dup(fds[i]);
        }

        auto const& settings = Settings_Instance();

        // Hopefully it's the same for both eyes in a layer
        auto& desc = leftIt->second.first->textDesc;

        alvr::render::RendererCreateInfo rendererCI {
            .format = (vk::Format)(VkFormat)desc.nFormat,
            .inputEyeExtent {
                .width = desc.nWidth,
                .height = desc.nHeight,
            },
            // The stream size negotiated with the headset, not the size
            // SteamVR renders at: the two differ when the emulated headset
            // resolution is changed, and the headset only decodes the first.
            .outputExtent {
                .width = settings->m_renderWidth,
                .height = settings->m_renderHeight,
            },
            .inputImgFds = fds,
        };

        // Renderer and encoder setup can throw (Vulkan import, VAAPI). An
        // exception escaping Present kills vrserver, so log and stay idle.
        DrainPendingJob();
        std::lock_guard<std::mutex> encLock(m_encMutex);
        m_encGeneration++;

        try {
            bool replaced = false;
            try {
                replaced = enc.tryReplaceInputs(rendererCI);
            } catch (std::exception const& e) {
                // The fds may be partly consumed; dup fresh ones for the full rebuild.
                Warn("Swapping in new textures failed (%s), rebuilding the encoder\n", e.what());
                for (u32 i = 0; i < 3; ++i) {
                    rendererCI.inputImgFds[i] = dup(leftIt->second.first->fds[i]);
                    rendererCI.inputImgFds[i + 3] = dup(rightIt->second.first->fds[i]);
                }
            }
            if (replaced) {
                Info("Swapped in new %ux%u textures, encoder kept\n", desc.nWidth, desc.nHeight);
            } else {
                enc.createImages(rendererCI);
                enc.initEncoding();
            }
        } catch (std::exception const& e) {
            Error(
                "Could not set up the encoder: %s. Staying idle until the next client connect.\n",
                e.what()
            );
            std::string what = e.what();
            if (what.find("OutOfDeviceMemory") != std::string::npos
                || what.find("Cannot allocate memory") != std::string::npos
                || what.find("CUDA device") != std::string::npos) {
                Error(
                    "The GPU is probably out of video memory. SteamVR renders at %ux%u per eye; "
                    "lower the emulated headset resolution, the transcoding resolution or the "
                    "SteamVR render resolution.\n",
                    desc.nWidth,
                    desc.nHeight
                );
            }
            enc.shutdown();
            layer0Texts.fill(0);
            m_encoderState = EncoderState::Idle;
        }

        // We'll get em next time
        return;
    }

    FrameJob job {};
    // Every submitted layer, so the worker can compose them like the Windows
    // renderer does: the first opaque, the rest alpha blended over it, each
    // sampled within its bounds and rotated into the first layer's view.
    {
        auto const& base = m_submitLayers[0][0].mHmdPose.m;
        job.layerCount = (uint32_t)layerCount;
        for (int i = 0; i < layerCount; ++i) {
            auto& layer = job.layers[i];
            layer.left = m_submitLayers[i][0].hTexture;
            layer.right = m_submitLayers[i][1].hTexture;
            layer.boundsL = m_submitLayers[i][0].bounds;
            layer.boundsR = m_submitLayers[i][1].bounds;

            // R_i^T * R_0 from the rotation parts of the world-from-head poses.
            auto const& pose = m_submitLayers[i][0].mHmdPose.m;
            for (int c = 0; c < 3; ++c) {
                for (int r = 0; r < 3; ++r) {
                    float v = 0.f;
                    for (int k = 0; k < 3; ++k) {
                        v += pose[k][r] * base[k][c];
                    }
                    layer.rotation[c * 3 + r] = v;
                    if (std::fabs(v - (r == c ? 1.f : 0.f)) > 1e-5f) {
                        layer.rotated = true;
                    }
                }
            }
        }
    }
    job.leftIdx = leftIdx.value();
    job.rightIdx = rightIdx.value();
    job.targetTimestampNs = m_targetTimestampNs;
    job.renderOrientation = m_framePoseRotation;
    job.generation = m_encGeneration;
    job.enqueueTime = std::chrono::steady_clock::now();

    // Export each eye's write fences so the render submission starts exactly
    // when the writer finishes. Both eyes or neither: a single armed eye
    // would order one read correctly and leave the other racing, which is
    // strictly harder to diagnose than the symmetric fallback.
    {
        u32 const eyeSlot[2] = { leftIdx.value(), rightIdx.value() };
        for (int e = 0; e < 2; ++e) {
            if (m_syncFds[eyeSlot[e]] >= 0) {
                job.waitFds[e] = exportReadSyncFile(m_syncFds[eyeSlot[e]]);
            }
        }
        if (job.waitFds[0] < 0 || job.waitFds[1] < 0) {
            closeWaitFds(job.waitFds);
        }
    }

    {
        std::lock_guard<std::mutex> lock(m_jobMutex);
        if (m_workerExit) {
            // Teardown already drained the mailbox; do not repopulate it.
            closeWaitFds(job.waitFds);
            return;
        }
        // Latest wins: a still-pending job is replaced, and its fences will
        // never be waited on, so close them here.
        if (m_jobPending) {
            closeWaitFds(m_job.waitFds);
        }
        m_job = job;
        m_jobPending = true;
    }
    m_jobCv.notify_one();
}

bool OvrDirectModeComponent::PrepareLayers(FrameJob const& job, std::vector<alvr::LayerRef>& out) {
    // Drop imports of destroyed sets first, so their memory goes back
    // before anything new is imported.
    std::vector<uint64_t> evictions;
    {
        std::lock_guard<std::mutex> lock(m_evictMutex);
        evictions.swap(m_pendingEvictions);
    }
    if (!evictions.empty()) {
        enc.evictLayerImages(evictions);
        for (auto key : evictions) {
            m_failedLayerImports.erase(key);
        }
    }

    auto fullBounds = [](vr::VRTextureBounds_t const& b) {
        return b.uMin == 0.f && b.vMin == 0.f && b.uMax == 1.f && b.vMax == 1.f;
    };
    if (job.layerCount == 0
        || (job.layerCount == 1 && fullBounds(job.layers[0].boundsL)
            && fullBounds(job.layers[0].boundsR))) {
        return false;
    }

    // Per-eye view tangents for rotating layers into the first one's view.
    float tans[2][4] = {};
    bool tansValid = false;
    {
        std::unique_lock<std::mutex> lock(m_viewParamsMutex);
        if (m_viewParamsValid) {
            for (int e = 0; e < 2; ++e) {
                auto const& fov = m_viewParams[e].fov;
                tans[e][0] = tanf(fov.left);
                tans[e][1] = tanf(fov.right);
                tans[e][2] = tanf(fov.up);
                tans[e][3] = tanf(fov.down);
            }
            tansValid = tans[0][1] != tans[0][0] && tans[1][1] != tans[1][0]
                && tans[0][2] != tans[0][3] && tans[1][2] != tans[1][3];
        }
    }

    auto ensureImported = [&](vr::SharedTextureHandle_t handle) {
        if (handle == 0 || enc.hasLayerImage(handle) || m_failedLayerImports.count(handle)) {
            return;
        }
        int fd = -1;
        SwapTextureSetDesc_t desc {};
        {
            std::lock_guard<std::mutex> lock(m_presentMutex);
            auto it = m_handleMap.find(handle);
            if (it != m_handleMap.end()) {
                fd = dup(it->second.first->fds[it->second.second]);
                desc = it->second.first->textDesc;
            }
        }
        if (fd < 0) {
            return;
        }
        try {
            enc.importLayerImage(
                handle, fd, vk::Format(desc.nFormat), vk::Extent2D { desc.nWidth, desc.nHeight }
            );
        } catch (std::exception const& e) {
            m_failedLayerImports.insert(handle);
            Error(
                "Could not import a layer texture (%s); that layer will not be shown\n", e.what()
            );
        }
    };

    auto toArray = [](vr::VRTextureBounds_t const& b, float out[4]) {
        out[0] = b.uMin;
        out[1] = b.vMin;
        out[2] = b.uMax;
        out[3] = b.vMax;
    };

    out.clear();
    for (uint32_t i = 0; i < job.layerCount; ++i) {
        auto const& layer = job.layers[i];
        alvr::LayerRef ref;
        if (i > 0) {
            ensureImported(layer.left);
            ensureImported(layer.right);
            ref.left = layer.left;
            ref.right = layer.right;
        }
        toArray(layer.boundsL, ref.params.boundsL);
        toArray(layer.boundsR, ref.params.boundsR);
        if (layer.rotated && tansValid) {
            for (int c = 0; c < 3; ++c) {
                float* col = c == 0 ? ref.params.rotCol0
                    : c == 1        ? ref.params.rotCol1
                                    : ref.params.rotCol2;
                col[0] = layer.rotation[c * 3 + 0];
                col[1] = layer.rotation[c * 3 + 1];
                col[2] = layer.rotation[c * 3 + 2];
                col[3] = 0.f;
            }
            memcpy(ref.params.tansL, tans[0], sizeof(tans[0]));
            memcpy(ref.params.tansR, tans[1], sizeof(tans[1]));
            ref.params.warp = 1;
        }
        out.push_back(ref);
    }
    return true;
}

void OvrDirectModeComponent::PostPresent(const Throttling_t* pThrottling) {
    // Prop_Hmd_SupportsAppThrottling_Bool is set, so SteamVR sends throttle
    // hints here and predicts poses assuming we honor them. Ignoring them
    // while it throttles an app predicts poses for a cadence we are not
    // keeping, which reprojects the world against head motion.
    uint32_t throttleFrames = 0;
    if (pThrottling != nullptr) {
        throttleFrames = pThrottling->nFramesToThrottle;
        if (pThrottling->nFramesToThrottle != m_lastThrottleFrames
            || pThrottling->nAdditionalFramesToPredict != m_lastPredictFrames) {
            m_lastThrottleFrames = pThrottling->nFramesToThrottle;
            m_lastPredictFrames = pThrottling->nAdditionalFramesToPredict;
            Info(
                "Throttling: framesToThrottle=%u additionalFramesToPredict=%u\n",
                pThrottling->nFramesToThrottle,
                pThrottling->nAdditionalFramesToPredict
            );
        }
    }

    PaceAfterPresent(throttleFrames);
}

void OvrDirectModeComponent::GetFrameTiming(vr::DriverDirectMode_FrameTiming* pFrameTiming) {
    if (pFrameTiming == nullptr
        || pFrameTiming->m_nSize < sizeof(vr::DriverDirectMode_FrameTiming)) {
        return;
    }

    // Skipped ticks are not reported as dropped frames: the compositor reads a
    // drop as GPU overload and enables interleaved reprojection, which halves
    // the rate, so the grid then skips a tick every frame and reports another
    // drop, throttled from then on. Consume the counter without reporting it
    // until there is a channel that does not throttle.
    (void)TakeSkippedVsyncs();
    pFrameTiming->m_nNumFramePresents = 1;
    pFrameTiming->m_nNumMisPresented = 0;
    pFrameTiming->m_nNumDroppedFrames = 0;
    // These bits overlap the throttle mask and must not be echoed back
    // unhandled, per the interface header.
    pFrameTiming->m_nReprojectionFlags = 0;
}
