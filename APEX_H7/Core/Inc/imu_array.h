#ifndef __IMU_ARRAY_H
#define __IMU_ARRAY_H

#include "lsm6dso.h"
#include "stm32h7xx.h"
#include "main.h"

#include <stdint.h>

// Initialize all IMUs within the system. Returns 0 if successful on all inits, 
// otherwise returns the id of the first failed IMU
int imu_init_all(SPI_HandleTypeDef *hspi);

// A wrapper to read all of the IMU data in the system
void read_all_imus(int16_t imu_data[NUM_IMUs][NUM_IMU_CHANNELS]);

// Check WHO_AM_I for all registers in the system
int check_all_who_am_i();



#endif /* __IMU_ARRAY_H */
