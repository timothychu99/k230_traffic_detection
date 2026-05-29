#ifndef TRIPLE_CAM_YOLO_MOSAIC_PERSON_FIFO_PROTOCOL_H_
#define TRIPLE_CAM_YOLO_MOSAIC_PERSON_FIFO_PROTOCOL_H_

#include <stdint.h>

#define PERSON_FIFO_MAGIC 0x54435031U /* "TCP1" */
#define PERSON_FIFO_VERSION 2U
#define PERSON_FIFO_CAM_COUNT 3U
/** Kendryte datafifo item size (sample_writer uses 1024; must be multiple of 32 bytes). */
#define PERSON_FIFO_BLOCK_SIZE 1024U
#define PERSON_FIFO_PAYLOAD_SIZE 72U

typedef struct
{
    uint32_t magic;
    uint16_t version;
    uint16_t cam_count;
    uint64_t seq;
    uint64_t timestamp_us;
    uint32_t person_count[PERSON_FIFO_CAM_COUNT];
    uint32_t car_count[PERSON_FIFO_CAM_COUNT];
    uint32_t vehicle_count[PERSON_FIFO_CAM_COUNT];
    uint16_t band_y[PERSON_FIFO_CAM_COUNT];
    uint16_t band_h[PERSON_FIFO_CAM_COUNT];
    uint8_t reserved[PERSON_FIFO_BLOCK_SIZE - PERSON_FIFO_PAYLOAD_SIZE];
} person_fifo_packet_t;

#define PERSON_FIFO_PACKET_SIZE PERSON_FIFO_BLOCK_SIZE

#if defined(__cplusplus)
static_assert(sizeof(person_fifo_packet_t) == PERSON_FIFO_BLOCK_SIZE,
    "person_fifo_packet_t must match PERSON_FIFO_BLOCK_SIZE");
static_assert(PERSON_FIFO_BLOCK_SIZE % 32U == 0U, "datafifo block must be a multiple of 32 bytes");
#else
_Static_assert(sizeof(person_fifo_packet_t) == PERSON_FIFO_BLOCK_SIZE,
    "person_fifo_packet_t must match PERSON_FIFO_BLOCK_SIZE");
_Static_assert(PERSON_FIFO_BLOCK_SIZE % 32U == 0U, "datafifo block must be a multiple of 32 bytes");
#endif

#endif /* TRIPLE_CAM_YOLO_MOSAIC_PERSON_FIFO_PROTOCOL_H_ */
