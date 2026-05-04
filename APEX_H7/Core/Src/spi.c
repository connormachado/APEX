#include <stdint.h>
#include <stdio.h>

#include "spi.h"
#include "stm32h7xx_hal_spi.h"
#include "stm32h7xx_hal_def.h"


// Handle an 8-bit blocking SPI transfer, providing an 
// 8-bit data return into provided pointer
HAL_StatusTypeDef spi_txfr_8(SPI_HandleTypeDef *hspi, uint8_t tx_data, uint8_t *rx_data) {
    // Check for null pointers
    if (hspi == NULL || rx_data == NULL) {
        printf("Error: Null pointer provided to spi_txfr_8\r\n");
        return HAL_ERROR;
    }

    // Variable for return data
    uint8_t return_data = 0x00;

    // Use HAL to handle blocking tx and rx SPI transfer
    HAL_StatusTypeDef status = HAL_SPI_TransmitReceive(hspi, &tx_data, &return_data, 1, HAL_MAX_DELAY);
    
    // Check status of return and print error if failed. 
    // Return 0xFF in case of failure to indicate error in return data.
    if (status != HAL_OK) {
        printf("SPI transfer failed with status %d\r\n", status);
        *rx_data = return_data;
        return HAL_ERROR;
    }

    *rx_data = return_data;     // Store return data in provided pointer
    return HAL_OK;  
}