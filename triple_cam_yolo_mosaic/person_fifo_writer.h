#pragma once

#include <cstdint>

/** RTOS writer: per-camera detection counts to datafifo for little-core reader. */
int PersonFifoWriterInit();
bool PersonFifoWriterReady();
void PersonFifoWriterDeinit();
void PersonFifoWriterPublish(const uint32_t person_count[3], const uint32_t car_count[3],
    const uint32_t vehicle_count[3], const uint16_t band_y[3], const uint16_t band_h[3], uint64_t timestamp_us);
