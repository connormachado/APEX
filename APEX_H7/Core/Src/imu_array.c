#include "imu_array.h"

// IMU Pin Definitions from CubeMX
static GPIO_TypeDef *cs_ports[NUM_IMUs] = {
    CS_IMU0_GPIO_Port,
    CS_IMU1_GPIO_Port,
    // CS_IMU2_GPIO_Port,
    // CS_IMU3_GPIO_Port,
    // CS_IMU4_GPIO_Port
};

static uint16_t cs_pins[NUM_IMUs] = {
    CS_IMU0_Pin,
    CS_IMU1_Pin,
    // CS_IMU2_Pin,
    // CS_IMU3_Pin,
    // CS_IMU4_Pin
};

// Initialize all IMUs in the system. Return 0 if successful on all inits, otherwise return the id of the first failed IMU
int imu_init_all(SPI_HandleTypeDef *hspi) {
    // Assign the IMUs their pins and save them within the global IMU array
    for (int i=0; i<NUM_IMUs; i++) {
        imus[i] = (imu_t) {
            .hspi = hspi,
            .cs_port = cs_ports[i],
            .cs_pin = cs_pins[i],
            .id = i,
        };

        if (!imu_init(&imus[i])) {
            return imus[i].id;
        }
    }
    return 0;
}


// A wrapper to read all of the IMU data in the system
// Saves returned data into a multi-dimensional array passed into function
void read_all_imus(int16_t imu_data[NUM_IMUs][NUM_IMU_CHANNELS]) {
    for (int i=0; i<NUM_IMUs; i++) {
        imu_read_all_data(&imus[i], imu_data[i]);
    }
}


// A wrapper to check all IMUs against the WHO_AM_I register
// Returns 0 if all checks are successful, otherwise returns id of first IMU failure
int check_all_who_am_i() {
    for (int i=0; i<NUM_IMUs; i++) {
        if (!imu_check_who_am_i(&imus[i])) {
            return imus[i].id;
        }
    }
    return 0;
}
