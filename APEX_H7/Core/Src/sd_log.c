#include <stdint.h>
#include <stddef.h>
#include <stdio.h>
#include <string.h>

#include "sd_log.h"
#include "ff.h"
#include "fatfs.h"          /* USERPath registered by MX_FATFS_Init */

/* AXI SRAM placement is mandatory: DMA2 cannot reach DTCM, and the project's
   default BSS lands in DTCMRAM on this linker. See CLAUDE.md trap #1. */
#define AXI_SRAM __attribute__((section(".axi_sram")))

AXI_SRAM static FATFS   fs;
AXI_SRAM static FIL     file;
AXI_SRAM static uint8_t sector_buf[512];

typedef enum { ST_UNINITIALIZED = 0, ST_READY, ST_FAULT } state_t;
static state_t state = ST_UNINITIALIZED;
static size_t  buf_pos;     /* bytes currently buffered in sector_buf */

/* CRC-8/CCITT, polynomial 0x07, init 0x00, no reflect, no xorout. */
uint8_t crc8_compute(const uint8_t *data, uint8_t len) {
    uint8_t crc = 0x00;
    for (uint8_t i = 0; i < len; i++) {
        crc ^= data[i];
        for (uint8_t b = 0; b < 8; b++) {
            crc = (crc & 0x80) ? (uint8_t)((crc << 1) ^ 0x07) : (uint8_t)(crc << 1);
        }
    }
    return crc;
}

/* Write the full 512-byte sector_buf and reset buf_pos. Caller guarantees
   buf_pos == 512. */
static sd_log_status_t flush_full_sector(void) {
    UINT bw = 0;
    FRESULT fr = f_write(&file, sector_buf, 512, &bw);
    if (fr != FR_OK || bw != 512) {
        state = ST_FAULT;
        return SD_LOG_ERR_WRITE;
    }
    buf_pos = 0;
    return SD_LOG_OK;
}

/* Probe APEX0000.BIN .. APEX9999.BIN for the lowest unused name.
   Returns SD_LOG_OK on success and writes the filename into out (>=13 bytes).
   Returns SD_LOG_ERR_OPEN if all 10000 indices are taken or on FS error. */
static sd_log_status_t pick_filename(char *out, size_t out_len) {
    for (int i = 0; i <= 9999; i++) {
        snprintf(out, out_len, "APEX%04d.BIN", i);
        FILINFO fi;
        FRESULT fr = f_stat(out, &fi);
        if (fr == FR_NO_FILE) return SD_LOG_OK;
        if (fr != FR_OK)      return SD_LOG_ERR_OPEN;
    }
    return SD_LOG_ERR_OPEN;
}

sd_log_status_t sd_log_init(void) {
    /* From FAULT, attempt remount. From READY, no-op. */
    if (state == ST_READY) return SD_LOG_OK;

    FRESULT fr = f_mount(&fs, USERPath, 1);
    if (fr != FR_OK) {
        state = ST_FAULT;
        return SD_LOG_ERR_MOUNT;
    }

    char path[16];
    sd_log_status_t s = pick_filename(path, sizeof path);
    if (s != SD_LOG_OK) {
        state = ST_FAULT;
        return s;
    }

    fr = f_open(&file, path, FA_CREATE_ALWAYS | FA_WRITE);
    if (fr != FR_OK) {
        state = ST_FAULT;
        return SD_LOG_ERR_OPEN;
    }

    buf_pos = 0;
    state = ST_READY;
    return SD_LOG_OK;
}

sd_log_status_t sd_log_write_frame(const apex_frame_t *frame) {
    if (state != ST_READY)        return SD_LOG_ERR_NOT_INITIALIZED;
    if (frame == NULL)            return SD_LOG_ERR_WRITE;

    const uint8_t *src = (const uint8_t *)frame;
    size_t free_bytes = 512 - buf_pos;

    if (free_bytes >= sizeof(apex_frame_t)) {
        memcpy(&sector_buf[buf_pos], src, sizeof(apex_frame_t));
        buf_pos += sizeof(apex_frame_t);
        if (buf_pos == 512) return flush_full_sector();
        return SD_LOG_OK;
    }

    /* Frame straddles a 512-byte boundary: fill, flush, then copy remainder. */
    memcpy(&sector_buf[buf_pos], src, free_bytes);
    buf_pos = 512;
    sd_log_status_t r = flush_full_sector();
    if (r != SD_LOG_OK) return r;

    size_t rest = sizeof(apex_frame_t) - free_bytes;
    memcpy(&sector_buf[0], src + free_bytes, rest);
    buf_pos = rest;
    return SD_LOG_OK;
}

sd_log_status_t sd_log_flush(void) {
    if (state != ST_READY) return SD_LOG_ERR_NOT_INITIALIZED;

    if (buf_pos > 0) {
        UINT bw = 0;
        FRESULT fr = f_write(&file, sector_buf, (UINT)buf_pos, &bw);
        if (fr != FR_OK || bw != (UINT)buf_pos) {
            state = ST_FAULT;
            return SD_LOG_ERR_WRITE;
        }
        buf_pos = 0;
    }

    FRESULT fr = f_sync(&file);
    if (fr != FR_OK) {
        state = ST_FAULT;
        return SD_LOG_ERR_WRITE;
    }
    return SD_LOG_OK;
}

sd_log_status_t sd_log_close(void) {
    if (state != ST_READY) return SD_LOG_ERR_NOT_INITIALIZED;

    sd_log_status_t s = sd_log_flush();
    /* Close even if flush failed, to release the file handle. */
    FRESULT fr = f_close(&file);
    state = ST_UNINITIALIZED;
    if (s != SD_LOG_OK) return s;
    return (fr == FR_OK) ? SD_LOG_OK : SD_LOG_ERR_WRITE;
}

/* Shared dump body. Flushes, prints file size, optionally dumps the first
   frame's raw bytes (head only), then prints `count` frames starting at
   `start_frame_index` in decoded form. Restores the file pointer to EOF
   before returning so subsequent sd_log_write_frame calls keep appending. */
static sd_log_status_t dump_helper(size_t start_frame_index,
                                   size_t count,
                                   _Bool include_raw_dump) {
    if (state != ST_READY) return SD_LOG_ERR_NOT_INITIALIZED;

    sd_log_status_t s = sd_log_flush();
    if (s != SD_LOG_OK) return s;

    FSIZE_t end_pos = f_size(&file);
    printf("file size: %lu bytes\r\n", (unsigned long)end_pos);

    FRESULT fr;
    UINT br;

    if (include_raw_dump) {
        fr = f_lseek(&file, 0);
        if (fr != FR_OK) { state = ST_FAULT; return SD_LOG_ERR_WRITE; }

        uint8_t raw[sizeof(apex_frame_t)];
        br = 0;
        fr = f_read(&file, raw, sizeof(raw), &br);
        if (fr == FR_OK && br == sizeof(raw)) {
            printf("raw bytes (frame 0):");
            for (size_t i = 0; i < sizeof(raw); i++) {
                printf(" %02x", (unsigned int)raw[i]);
            }
            printf("\r\n");
        }
    }

    fr = f_lseek(&file, (FSIZE_t)(start_frame_index * sizeof(apex_frame_t)));
    if (fr != FR_OK) { state = ST_FAULT; return SD_LOG_ERR_WRITE; }

    for (size_t i = 0; i < count; i++) {
        apex_frame_t f;
        br = 0;
        fr = f_read(&file, &f, sizeof(f), &br);
        if (fr != FR_OK || br != sizeof(f)) break;

        printf("[disk frame %3lu] sync=0x%02X ts=%luus idx=%u  "
               "Gyro: %6d %6d %6d  Accel: %6d %6d %6d  crc=0x%02X\r\n",
               (unsigned long)(start_frame_index + i),
               (unsigned int)f.sync,
               (unsigned long)f.timestamp_us,
               (unsigned int)f.imu_index,
               f.gyro[0], f.gyro[1], f.gyro[2],
               f.accel[0], f.accel[1], f.accel[2],
               (unsigned int)f.crc8);
    }

    fr = f_lseek(&file, end_pos);
    if (fr != FR_OK) { state = ST_FAULT; return SD_LOG_ERR_WRITE; }

    return SD_LOG_OK;
}

sd_log_status_t sd_log_dump_head(size_t num_frames) {
    return dump_helper(0, num_frames, 1);
}

sd_log_status_t sd_log_dump_tail(size_t num_frames) {
    if (state != ST_READY) return SD_LOG_ERR_NOT_INITIALIZED;

    /* Flush before sizing so the tail reflects on-disk bytes including any
       frames still in sector_buf. dump_helper will flush again (idempotent). */
    sd_log_status_t s = sd_log_flush();
    if (s != SD_LOG_OK) return s;

    size_t total = (size_t)(f_size(&file) / sizeof(apex_frame_t));
    size_t start = (total <= num_frames) ? 0 : (total - num_frames);
    return dump_helper(start, num_frames, 0);
}

                                                                                                                                                                                    
_Bool sd_log_self_test(void) {
    if (sd_log_init() != SD_LOG_OK) { printf("init fail\r\n"); return 0; }                                                                                                        
                                                                                                                                                                                
    apex_frame_t f;
    memset(&f, 0, sizeof f);                                                                                                                                                      
    f.sync = 0xA5;                                                                                                                                                                

    for (uint32_t i = 0; i < 100; i++) {                                                                                                                                          
        f.timestamp_us = i;
        f.imu_index    = (uint8_t)(i % 5);                                                                                                                                        
        f.gyro[0]      = (int16_t)i;
        f.accel[2]     = (int16_t)-i;                                                                                                                                             
        f.crc8         = crc8_compute((const uint8_t *)&f, 18);                                                                                                                   
        if (sd_log_write_frame(&f) != SD_LOG_OK) { printf("write %lu fail\r\n", i); return 0; }                                                                                   
    }                                                                                                                                                                             
                
    if (sd_log_flush() != SD_LOG_OK) { printf("flush fail\r\n"); return 0; }                                                                                                      
    if (sd_log_close() != SD_LOG_OK) { printf("close fail\r\n"); return 0; }
    printf("self_test OK: 3400 bytes written\r\n");                                                                                                                               
    return 1;                                                                                                                                                                     
}   