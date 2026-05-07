#ifndef __SD_LOG_H
#define __SD_LOG_H

#include <stdint.h>
#include <stddef.h>

#ifdef __cplusplus
extern "C" {
#endif

/* 34-byte packed log frame. Layout matches CLAUDE.md and the offline parser. */
typedef struct __attribute__((packed)) {
    uint8_t  sync;            /* 0xA5 */
    uint32_t timestamp_us;    /* TIM5 microseconds */
    uint8_t  imu_index;       /* 0..4 */
    int16_t  gyro[3];         /* raw LSB */
    int16_t  accel[3];        /* raw LSB */
    uint8_t  crc8;            /* CCITT poly 0x07, init 0x00, over bytes 0..17 */
    uint8_t  reserved[15];    /* zero-filled */
} apex_frame_t;
_Static_assert(sizeof(apex_frame_t) == 34, "apex_frame_t must be 34 bytes");

typedef enum {
    SD_LOG_OK = 0,
    SD_LOG_ERR_MOUNT,
    SD_LOG_ERR_OPEN,
    SD_LOG_ERR_WRITE,
    SD_LOG_ERR_FULL,
    SD_LOG_ERR_NOT_INITIALIZED,
} sd_log_status_t;

sd_log_status_t sd_log_init(void);
sd_log_status_t sd_log_write_frame(const apex_frame_t *frame);
sd_log_status_t sd_log_flush(void);
sd_log_status_t sd_log_close(void);

/* Read frames back from the open log file and printf them in human-readable
   form for live UART comparison against the IMU printf. Both flush pending
   data first, save/restore the end-of-file position, and take a frame count
   (not bytes). */
sd_log_status_t sd_log_dump_head(size_t num_frames);
sd_log_status_t sd_log_dump_tail(size_t num_frames);

uint8_t crc8_compute(const uint8_t *data, uint8_t len);

#ifdef __cplusplus
}
#endif

#endif /* __SD_LOG_H */
