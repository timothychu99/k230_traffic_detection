/**
 * Little-core Linux: read triple_cam_yolo_mosaic RTOS person-count FIFO and print
 * per-band counts (cam0 top / cam1 mid / cam2 bottom).
 *
 * Usage: triple_cam_mosaic_person_reader.elf <phy_addr_hex>
 *   phy_addr from RTOS boot log PERSON_FIFO_PHY_ADDR=0x...
 *   or env TRIPLE_CAM_PERSON_FIFO_PHY_ADDR.
 *
 * Start order: run triple_cam_yolo_mosaic.elf first, wait for PERSON_FIFO_PHY_ADDR,
 * then start this reader with that exact address and matching block size (1024).
 */

#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <signal.h>
#include <unistd.h>

extern "C" {
#include "k_datafifo.h"
#include "k_type.h"
}

#include "person_fifo_protocol.h"

static constexpr uint32_t kPersonFifoBlockSize = PERSON_FIFO_PACKET_SIZE;
static constexpr unsigned kSummaryWindowFrames = 5U;

struct WindowMaxCounts {
    uint32_t pedestrian[PERSON_FIFO_CAM_COUNT]{};
    uint32_t motorcycle[PERSON_FIFO_CAM_COUNT]{};
    uint32_t vehicle[PERSON_FIFO_CAM_COUNT]{};
};

static void UpdateWindowMax(WindowMaxCounts *max_out, const person_fifo_packet_t &pkt)
{
    for (unsigned cam = 0; cam < PERSON_FIFO_CAM_COUNT; ++cam) {
        if (pkt.person_count[cam] > max_out->pedestrian[cam]) {
            max_out->pedestrian[cam] = pkt.person_count[cam];
        }
        if (pkt.car_count[cam] > max_out->motorcycle[cam]) {
            max_out->motorcycle[cam] = pkt.car_count[cam];
        }
        if (pkt.vehicle_count[cam] > max_out->vehicle[cam]) {
            max_out->vehicle[cam] = pkt.vehicle_count[cam];
        }
    }
}

static void PrintWindowMax(const WindowMaxCounts &max_counts, unsigned frame_count, uint64_t first_seq,
    uint64_t last_seq)
{
    static const char *kCamNames[PERSON_FIFO_CAM_COUNT] = {"cam0(top)", "cam1(mid)", "cam2(bot)"};
    printf("max over last %u frames (seq %llu-%llu):\n", frame_count, static_cast<unsigned long long>(first_seq),
        static_cast<unsigned long long>(last_seq));
    for (unsigned cam = 0; cam < PERSON_FIFO_CAM_COUNT; ++cam) {
        printf("  %s: pedestrians=%u motorcycles=%u vehicles=%u\n", kCamNames[cam], max_counts.pedestrian[cam],
            max_counts.motorcycle[cam], max_counts.vehicle[cam]);
    }
    fflush(stdout);
}

static volatile sig_atomic_t g_quit = 0;

static void SigHandler(int sig)
{
    (void)sig;
    g_quit = 1;
}

static uint64_t ParsePhyAddr(int argc, char **argv)
{
    const char *src = nullptr;
    if (argc >= 2 && argv[1][0] != '\0') {
        src = argv[1];
    } else {
        src = getenv("TRIPLE_CAM_PERSON_FIFO_PHY_ADDR");
    }
    if (src == nullptr || src[0] == '\0') {
        return 0;
    }
    return static_cast<uint64_t>(strtoull(src, nullptr, 0));
}

static void PrintUsage(const char *prog)
{
    fprintf(stderr,
        "Usage: %s <phy_addr_hex>\n"
        "  1) Start triple_cam_yolo_mosaic.elf on big core\n"
        "  2) Copy PERSON_FIFO_PHY_ADDR=0x... from its boot log\n"
        "  3) Run: %s 0x<PHY_ADDR>\n"
        "  Or set env TRIPLE_CAM_PERSON_FIFO_PHY_ADDR.\n"
        "  Block size must be %u (protocol v%u).\n",
        prog, prog, kPersonFifoBlockSize, static_cast<unsigned>(PERSON_FIFO_VERSION));
}

int main(int argc, char **argv)
{
    const uint64_t phy_addr = ParsePhyAddr(argc, argv);
    if (phy_addr == 0U) {
        PrintUsage(argc > 0 ? argv[0] : "triple_cam_mosaic_person_reader.elf");
        return 1;
    }

    signal(SIGINT, SigHandler);
    signal(SIGTERM, SigHandler);

    k_datafifo_handle handle = static_cast<k_datafifo_handle>(K_DATAFIFO_INVALID_HANDLE);
    k_datafifo_params_s params = {32, kPersonFifoBlockSize, K_TRUE, DATAFIFO_READER};

    k_s32 ret = kd_datafifo_open_by_addr(&handle, &params, phy_addr);
    if (ret != K_SUCCESS) {
        fprintf(stderr, "kd_datafifo_open_by_addr failed: 0x%x (addr=0x%lx block=%u)\n", static_cast<unsigned>(ret),
            static_cast<unsigned long>(phy_addr), kPersonFifoBlockSize);
        fprintf(stderr, "Ensure triple_cam_yolo_mosaic.elf is running and printed PERSON_FIFO_PHY_ADDR.\n");
        fprintf(stderr, "Do not use pose_detection's 0x14435000 unless that is the printed mosaic address.\n");
        return 1;
    }
    if (handle == static_cast<k_datafifo_handle>(K_DATAFIFO_INVALID_HANDLE)) {
        fprintf(stderr, "kd_datafifo_open_by_addr returned invalid handle (addr=0x%lx block=%u)\n",
            static_cast<unsigned long>(phy_addr), kPersonFifoBlockSize);
        return 1;
    }

    fprintf(stderr, "person fifo reader: magic=0x%x ver=%u block=%u addr=0x%lx (summary every %u frames)\n",
        static_cast<unsigned>(PERSON_FIFO_MAGIC), static_cast<unsigned>(PERSON_FIFO_VERSION), kPersonFifoBlockSize,
        static_cast<unsigned long>(phy_addr), kSummaryWindowFrames);

    WindowMaxCounts window_max{};
    unsigned frames_in_window = 0;
    uint64_t window_first_seq = 0;
    uint64_t window_last_seq = 0;

    while (!g_quit) {
        k_u32 avail = 0;
        ret = kd_datafifo_cmd(handle, DATAFIFO_CMD_GET_AVAIL_READ_LEN, &avail);
        if (ret != K_SUCCESS) {
            fprintf(stderr, "GET_AVAIL_READ_LEN failed: 0x%x\n", static_cast<unsigned>(ret));
            break;
        }

        if (avail >= kPersonFifoBlockSize) {
            void *p_data = nullptr;
            ret = kd_datafifo_read(handle, &p_data);
            if (ret != K_SUCCESS) {
                fprintf(stderr, "kd_datafifo_read failed: 0x%x\n", static_cast<unsigned>(ret));
                break;
            }

            if (p_data != nullptr) {
                person_fifo_packet_t packet_copy;
                memcpy(&packet_copy, p_data, PERSON_FIFO_PACKET_SIZE);
                ret = kd_datafifo_cmd(handle, DATAFIFO_CMD_READ_DONE, p_data);
                if (ret != K_SUCCESS) {
                    fprintf(stderr, "READ_DONE failed: 0x%x\n", static_cast<unsigned>(ret));
                    break;
                }

                if (packet_copy.magic == PERSON_FIFO_MAGIC && packet_copy.version == PERSON_FIFO_VERSION) {
                    if (frames_in_window == 0U) {
                        window_first_seq = packet_copy.seq;
                    }
                    window_last_seq = packet_copy.seq;
                    UpdateWindowMax(&window_max, packet_copy);
                    ++frames_in_window;

                    if (frames_in_window >= kSummaryWindowFrames) {
                        PrintWindowMax(window_max, frames_in_window, window_first_seq, window_last_seq);
                        window_max = WindowMaxCounts{};
                        frames_in_window = 0;
                    }
                }
            }
        } else {
            usleep(5000);
        }
    }

    kd_datafifo_close(handle);
    return 0;
}
