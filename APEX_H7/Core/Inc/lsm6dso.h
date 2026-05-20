#ifndef __LSM6DSO_H
#define __LSM6DSO_H

#include <stdint.h>
#include <stdbool.h>

#include "stm32h7xx_hal.h"

/*****************************************************************/
//// Constants ////
#define NUM_IMU_CHANNELS (6)  // Number of IMU channels to read from
#define NUM_IMUs (1)          // Number of IMUs in the system

/*****************************************************************/
//// imu_t ////
typedef struct {
    SPI_HandleTypeDef *hspi;    // SPI handle for communication
    GPIO_TypeDef *cs_port;      // GPIO port for CS pin
    uint16_t cs_pin;            // GPIO pin number for CS pin
    uint8_t  id;                // IMU index (0..4) for logging purposes
} imu_t;

extern imu_t imus[NUM_IMUs];

/*****************************************************************/
//// Function Prototypes ////

// Initializes the LSM6DSO IMU. Sets up the SPI communication and configures the accelerometer and gyroscope settings
// void imu_init(SPI_HandleTypeDef *hspi, GPIO_TypeDef *cs_port, uint16_t cs_pin)
bool imu_init(imu_t *imu);

// WHO_AM_I register check should return 0x6C
bool imu_check_who_am_i(imu_t *imu);

// Boolean check if new data is ready from sensors
bool imu_data_ready(imu_t *imu);

// Write a byte of data to a register at reg_addr
void imu_write_reg(imu_t *imu, uint8_t reg_addr, uint8_t data);

// Read a byte of data from a register at reg_addr
void imu_read_reg(imu_t *imu, uint8_t reg_addr, uint8_t *data);

// Read the raw data from all sensors
// BUFFER MUST BE INITIALIZED TO SIZE NUM_IMU_CHANNELS * int16_t
void imu_read_all_data(imu_t *imu, int16_t *return_data_buffer);


/*****************************************************************/
//// Register Addresses ////

// WHO_AM_I Register
#define WHO_AM_I_ADDR (0x0F)

// Configuration Registers
#define CTRL1_XL_ADDR (0x10)
#define CTRL2_G_ADDR (0x11)
#define CTRL3_C_ADDR (0x12)

// Data Flag Register
#define STATUS_REG_ADDR (0x1E)

// Data registers
#define OUTX_L_G_ADDR (0x22)
#define OUTX_H_G_ADDR (0x23)
#define OUTY_L_G_ADDR (0x24)
#define OUTY_H_G_ADDR (0x25)
#define OUTZ_L_G_ADDR (0x26)
#define OUTZ_H_G_ADDR (0x27)
#define OUTX_L_A_ADDR (0x28)
#define OUTX_H_A_ADDR (0x29)
#define OUTY_L_A_ADDR (0x2A)
#define OUTY_H_A_ADDR (0x2B)
#define OUTZ_L_A_ADDR (0x2C)
#define OUTZ_H_A_ADDR (0x2D)

// Interrupt Register
#define INT1_CTRL_ADDR (0x0D)


/*****************************************************************/
//// Bit Masks for configuration ////

// SPI read bit. OR with register address to read from register
#define READ_MASK (0x80)        // OR with register address to read from register
#define WRITE_MASK (0x7F)       // AND with register address to write to register

// Dummy Data bit to be explicit
#define DUMMY_DATA (0x00)

//// Accelerometer Configuration Bits
#define ODR_XL_104HZ (0x40)
#define FS_XL_4G (0x10)

//// Gyrospcope Configuration Bits
#define ODR_G_104HZ (0x40)
#define FS_G_1000DPS (0x04) // check again

//// Control Register 3 Configuration Bits
#define SW_RESET (0x01)       // Boot up device in deterministic state
#define IF_INC (0x04)         // auto-increment register address on multi-byte access
#define BDU (0x40)            // Output registers not updated until both H and L have been read

//// Status Register Flag Bits
#define XLDA (0x01)           // Accelerometer new data available
#define GDA (0x02)            // Gyroscope new data available


#endif /* __LSM6DSO_H */