#include "lsm6dso.h"
#include "spi.h"
#include "stm32h7xx_hal_def.h"

#include <stdio.h>

// Garbage Data back, will never be important but just 
// need to pass a valid pointer to spi_txfr_8
static uint8_t junk = 0x00;

// Our array of IMUs within the system
imu_t imus[NUM_IMUs];

void cs_low(imu_t *imu) {
    HAL_GPIO_WritePin(imu->cs_port, imu->cs_pin, GPIO_PIN_RESET);  // CS LOW = select
}

void cs_high(imu_t *imu) {
    HAL_GPIO_WritePin(imu->cs_port, imu->cs_pin, GPIO_PIN_SET);    // CS HIGH = deselect
}


bool imu_init(imu_t *imu) {
    // Initialize SPI and GPIO for CS pin here
    // Configure accelerometer and gyroscope settings

    
    // Drive CS high and HAL delay to ensure SPI mode activates
    // NOTE: we do need this, from codex (makes sense to me):
    //      "makes the driver depend on board-level startup sequencing in main.c. 
    //      Right now main provides the 50 ms boot delay and drives CS high first, 
    //      so this instance may work. But the driver is no longer self-contained: 
    //      if you instantiate another IMU or call imu_init() from a different path, 
    //      bring-up behavior changes. For low-level driver code, that hidden 
    //      dependency is risky."
    cs_high(imu);
    HAL_Delay(75);


    // Check WHO_AM_I register to verify we are connected
    cs_high(imu);   // Set CS high bc who_am_i check will set it low
    if (!imu_check_who_am_i(imu)) {
        printf("Failed to connect to LSM6DSO\r\n");
        return false;
    } else {
        printf("Successfully connected to LSM6DSO\r\n");
    }

    // Write to CTRL3_C
    //    - Perform a software reset (SW_RESET)
    //        - Check if reset was applied
    //    - Enable Block Data Update (BDU)
    //    - Enable auto-incrementing addresses (IF_INC)
    imu_write_reg(imu, CTRL3_C_ADDR, SW_RESET);
    HAL_Delay(100);   // Delay to allow reset to complete

    // Check if reset was successful by checking SW_RESET bit
    uint8_t reset_check = 0x00;
    imu_read_reg(imu, CTRL3_C_ADDR, &reset_check);
    if (reset_check & SW_RESET) {
        printf("Software reset failed for LSM6DSO\r\n");
        return false;
    } else {
        printf("Software reset successful for LSM6DSO\r\n");
    }

    // Apply the rest of the CTRL3_C config bits
    imu_write_reg(imu, CTRL3_C_ADDR, BDU | IF_INC);


    // Configure the accelerometer
    //    - ODR Selection of 104 Hz 
    //    - Full-scale selection of 4gs
    imu_write_reg(imu, CTRL1_XL_ADDR, ODR_XL_104HZ | FS_XL_4G);


    // Configure the gyroscope
    //    - ODR Selection of 104 Hz
    //    - Full-scale selection of 1000 dps
    imu_write_reg(imu, CTRL2_G_ADDR, ODR_G_104HZ | FS_G_1000DPS);


    // Return success (for now just print to coolterm)
    printf("LSM6DSO %d initialization complete\r\n", imu->cs_pin);
    return true;
}


bool imu_check_who_am_i(imu_t *imu) {
    // Read WHO_AM_I register and check if it returns 0x6C
    // Set CS to low
    cs_low(imu);

    // Build read bit + address package
    uint8_t metadata = READ_MASK | WHO_AM_I_ADDR;

    // Build the transmit data and declare the return address
    uint8_t return_data = 0x00;

    // Send the package and print the results (checking for return status)
    HAL_StatusTypeDef status = spi_txfr_8(imu->hspi, metadata, &junk);
    if (status != HAL_OK) {
        printf("Failed to read WHO_AM_I register\r\n");
        cs_high(imu);
        return false;
    }

    status = spi_txfr_8(imu->hspi, DUMMY_DATA, &return_data);
    if (status != HAL_OK) {
        printf("Failed to receive WHO_AM_I data\r\n");
        cs_high(imu);
        return false;
    }
    printf("imu_check_who_am_i (0x6C): 0x%02X\r\n", return_data);

    // Set CS to high
    cs_high(imu);

    if (return_data == 0x6C) {
        return true;
    } else {
        return false;
    }
}


void imu_write_reg(imu_t *imu, uint8_t reg_addr, uint8_t data) {
    // Write to a register at reg_addr with provided data
    // Set CS to low
    cs_low(imu);

    // Send the transmit data: write address first, then data value
    HAL_StatusTypeDef status = spi_txfr_8(imu->hspi, reg_addr & WRITE_MASK, &junk);
    if (status != HAL_OK) {
        printf("Failed to write to register 0x%02X\r\n", reg_addr);
    }

    status = spi_txfr_8(imu->hspi, data, &junk);
    if (status != HAL_OK) {
        printf("Failed to write data 0x%02X to register 0x%02X\r\n", data, reg_addr);
    }

    // Set CS to high
    cs_high(imu);
}


// Read a single byte from a register: used for STATUS_REGISTERS 
// to check data ready flags
void imu_read_reg(imu_t *imu, uint8_t reg_addr, uint8_t *data) {
    // Read a byte of data from a register at reg_addr and store in provided pointer
    // Set CS to low
    cs_low(imu);

    // Send the transmit data: read address first, then dummy data to clock out return value
    HAL_StatusTypeDef status = spi_txfr_8(imu->hspi, reg_addr | READ_MASK, &junk);
    if (status != HAL_OK) {
        printf("Failed to read register 0x%02X\r\n", reg_addr);
    }

    status = spi_txfr_8(imu->hspi, DUMMY_DATA, data);
    if (status != HAL_OK) {
        printf("Failed to receive data from register 0x%02X\r\n", reg_addr);
    }

    // Set CS to high
    cs_high(imu);
}


// Read the incrementing data registers for gyro and accel
void imu_read_all_data(imu_t *imu, int16_t *return_data_buffer) {
    // Read the 6 gyro and 6 accel data registers in one SPI transaction
    // Store the results in the provided pointers

    // Do I do this here or in the main loop?
    // Check if data samples are ready to be read from BOTH
    // the gyroscope and the accelerometer
    // Will eventually be an interrupt driven system, but for
    // hardware bringup this is what we got
    uint16_t timeout = 1000;   // timeout to prevent infinite loop in case of hardware failure
    while (!imu_data_ready(imu)) {
        HAL_Delay(1);
        timeout--;
        if (timeout == 0) {
            printf("Timeout while waiting for IMU data to be ready\r\n");
            return;
        }
    }

    // Set CS to low
    cs_low(imu);

    // Send the address we want to read from first
    HAL_StatusTypeDef status = spi_txfr_8(imu->hspi, OUTX_L_G_ADDR | READ_MASK, &junk);
    if (status != HAL_OK) {
        printf("Failed to initiate read from register 0x%02X\r\n", OUTX_L_G_ADDR);
        cs_high(imu);
        return;
    }

    // Send and recieve dummy data to recieve the 12 bytes of return data
    uint8_t low_data = 0x00;
    uint8_t high_data = 0x00;
    for (int i=0; i<NUM_IMU_CHANNELS; i++){
        // Capture the low bit
        status = spi_txfr_8(imu->hspi, DUMMY_DATA, &low_data);
        if (status != HAL_OK) {
            printf("Failed to receive low data from register %d\r\n", i);
            cs_high(imu);
            return;
        }

        // Capture the high bit
        status = spi_txfr_8(imu->hspi, DUMMY_DATA, &high_data);
        if (status != HAL_OK) {
            printf("Failed to receive high data from register %d\r\n", i);
            cs_high(imu);
            return;
        }

        // Combine the low and high bytes to form the 16 bit return data sample
        return_data_buffer[i] = ((int16_t)high_data << 8) | low_data;
    }

    // Set CS to high
    cs_high(imu);
}


// Check if data is ready from the accelerometer AND the gyroscope
bool imu_data_ready(imu_t *imu) {
    // Check the STATUS_REG register to see if new data is ready from both the gyro and accel
    uint8_t status_reg = 0x00;
    imu_read_reg(imu, STATUS_REG_ADDR, &status_reg);

    bool accel_ready = status_reg & 0x01;   // Check if bit 0 is high
    bool gyro_ready = status_reg & 0x02;    // Check if bit 1 is high

    return accel_ready && gyro_ready;       // Return true only if both are ready
}