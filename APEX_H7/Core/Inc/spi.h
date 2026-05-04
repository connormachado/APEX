#ifndef __SPI_H
#define __SPI_H

#include <stdint.h>
#include "stm32h7xx_hal.h"

HAL_StatusTypeDef spi_txfr_8(SPI_HandleTypeDef *hspi, uint8_t tx_data, uint8_t *rx_data);


#endif /* __SPI_H */
