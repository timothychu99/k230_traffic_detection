#include "person_fifo_writer.h"

#include <cstdio>
#include <cstring>
#include <pthread.h>

extern "C" {
#include "k_datafifo.h"
#include "k_type.h"
}

#include "person_fifo_protocol.h"

extern "C" {
static void person_fifo_release_stream(void *pStream)
{
    (void)pStream;
}
}

namespace {

k_datafifo_handle g_person_fifo = static_cast<k_datafifo_handle>(K_DATAFIFO_INVALID_HANDLE);
uint64_t g_person_seq = 0;
person_fifo_packet_t g_person_packet;
pthread_mutex_t g_person_fifo_mu = PTHREAD_MUTEX_INITIALIZER;
bool g_person_fifo_ready = false;

} // namespace

int PersonFifoWriterInit()
{
    pthread_mutex_lock(&g_person_fifo_mu);
    if (g_person_fifo_ready) {
        pthread_mutex_unlock(&g_person_fifo_mu);
        return 0;
    }

    k_datafifo_params_s writer_params{};
    writer_params.u32EntriesNum = 32;
    writer_params.u32CacheLineSize = PERSON_FIFO_PACKET_SIZE;
    writer_params.bDataReleaseByWriter = K_TRUE;
    writer_params.enOpenMode = DATAFIFO_WRITER;
    k_s32 ret = kd_datafifo_open(&g_person_fifo, &writer_params);
    if (ret != K_SUCCESS) {
        std::printf("person datafifo open failed: 0x%x\n", static_cast<unsigned>(ret));
        g_person_fifo = static_cast<k_datafifo_handle>(K_DATAFIFO_INVALID_HANDLE);
        pthread_mutex_unlock(&g_person_fifo_mu);
        return -1;
    }

    uint64_t phy_addr = 0;
    ret = kd_datafifo_cmd(g_person_fifo, DATAFIFO_CMD_GET_PHY_ADDR, &phy_addr);
    if (ret != K_SUCCESS) {
        std::printf("person datafifo GET_PHY_ADDR failed: 0x%x\n", static_cast<unsigned>(ret));
        kd_datafifo_close(g_person_fifo);
        g_person_fifo = static_cast<k_datafifo_handle>(K_DATAFIFO_INVALID_HANDLE);
        pthread_mutex_unlock(&g_person_fifo_mu);
        return -1;
    }

    ret = kd_datafifo_cmd(g_person_fifo, DATAFIFO_CMD_SET_DATA_RELEASE_CALLBACK,
        (void *)person_fifo_release_stream);
    if (ret != K_SUCCESS) {
        std::printf("person datafifo SET_DATA_RELEASE_CALLBACK failed: 0x%x\n", static_cast<unsigned>(ret));
        kd_datafifo_close(g_person_fifo);
        g_person_fifo = static_cast<k_datafifo_handle>(K_DATAFIFO_INVALID_HANDLE);
        pthread_mutex_unlock(&g_person_fifo_mu);
        return -1;
    }

    g_person_fifo_ready = true;
    std::printf("PERSON_FIFO_PHY_ADDR=0x%lx BLOCK=%u (start Linux reader AFTER this line)\n",
        static_cast<unsigned long>(phy_addr), static_cast<unsigned>(PERSON_FIFO_PACKET_SIZE));
    pthread_mutex_unlock(&g_person_fifo_mu);
    return 0;
}

bool PersonFifoWriterReady()
{
    return g_person_fifo_ready;
}

void PersonFifoWriterDeinit()
{
    pthread_mutex_lock(&g_person_fifo_mu);
    if (g_person_fifo != static_cast<k_datafifo_handle>(K_DATAFIFO_INVALID_HANDLE)) {
        if (g_person_seq > 0U) {
            (void)kd_datafifo_write(g_person_fifo, nullptr);
        }
        kd_datafifo_close(g_person_fifo);
        g_person_fifo = static_cast<k_datafifo_handle>(K_DATAFIFO_INVALID_HANDLE);
    }
    g_person_fifo_ready = false;
    pthread_mutex_unlock(&g_person_fifo_mu);
}

void PersonFifoWriterPublish(const uint32_t person_count[3], const uint32_t car_count[3],
    const uint32_t vehicle_count[3], const uint16_t band_y[3], const uint16_t band_h[3], uint64_t timestamp_us)
{
    if (!g_person_fifo_ready) {
        return;
    }

    pthread_mutex_lock(&g_person_fifo_mu);
    if (g_person_fifo == static_cast<k_datafifo_handle>(K_DATAFIFO_INVALID_HANDLE)) {
        pthread_mutex_unlock(&g_person_fifo_mu);
        return;
    }

    k_u32 avail_write_len = 0;
    k_s32 ret = kd_datafifo_cmd(g_person_fifo, DATAFIFO_CMD_GET_AVAIL_WRITE_LEN, &avail_write_len);
    if (ret != K_SUCCESS || avail_write_len < PERSON_FIFO_PACKET_SIZE) {
        pthread_mutex_unlock(&g_person_fifo_mu);
        return;
    }

    std::memset(&g_person_packet, 0, sizeof(g_person_packet));
    g_person_packet.magic = PERSON_FIFO_MAGIC;
    g_person_packet.version = PERSON_FIFO_VERSION;
    g_person_packet.cam_count = PERSON_FIFO_CAM_COUNT;
    g_person_packet.seq = ++g_person_seq;
    g_person_packet.timestamp_us = timestamp_us;
    for (unsigned i = 0; i < PERSON_FIFO_CAM_COUNT; ++i) {
        g_person_packet.person_count[i] = person_count[i];
        g_person_packet.car_count[i] = car_count[i];
        g_person_packet.vehicle_count[i] = vehicle_count[i];
        g_person_packet.band_y[i] = band_y[i];
        g_person_packet.band_h[i] = band_h[i];
    }

    ret = kd_datafifo_write(g_person_fifo, &g_person_packet);
    if (ret != K_SUCCESS) {
        pthread_mutex_unlock(&g_person_fifo_mu);
        return;
    }
    (void)kd_datafifo_cmd(g_person_fifo, DATAFIFO_CMD_WRITE_DONE, nullptr);
    pthread_mutex_unlock(&g_person_fifo_mu);
}
