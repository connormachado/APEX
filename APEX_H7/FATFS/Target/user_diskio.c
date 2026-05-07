/* USER CODE BEGIN Header */
/**
 ******************************************************************************
  * @file    user_diskio.c
  * @brief   FatFs low-level disk I/O glue for SD card on SPI3 (APEX).
  *
  * Adapted from ChaN's reference SPI MMC/SDC driver (sdmm.c, elm-chan.org).
  * Targets STM32H753ZI, hspi3, CS line on PD2 (CS_SDCard0).
  ******************************************************************************
  */
 /* USER CODE END Header */

#ifdef USE_OBSOLETE_USER_CODE_SECTION_0
/* USER CODE BEGIN 0 */
/* USER CODE END 0 */
#endif

/* USER CODE BEGIN DECL */

/* Includes ------------------------------------------------------------------*/
#include <string.h>
#include "ff_gen_drv.h"
#include "main.h"
#include "user_diskio.h"

/* SPI handle from main.c (CubeMX-generated). */
extern SPI_HandleTypeDef hspi3;

/* ------------------------------------------------------------------------- */
/* MMC/SDC command set                                                       */
/* ------------------------------------------------------------------------- */
#define CMD0    (0)         /* GO_IDLE_STATE */
#define CMD1    (1)         /* SEND_OP_COND (MMC) */
#define ACMD41  (0x80 + 41) /* SEND_OP_COND (SDC) */
#define CMD8    (8)         /* SEND_IF_COND */
#define CMD9    (9)         /* SEND_CSD */
#define CMD10   (10)        /* SEND_CID */
#define CMD12   (12)        /* STOP_TRANSMISSION */
#define CMD16   (16)        /* SET_BLOCKLEN */
#define CMD17   (17)        /* READ_SINGLE_BLOCK */
#define CMD18   (18)        /* READ_MULTIPLE_BLOCK */
#define CMD23   (23)        /* SET_BLOCK_COUNT (MMC) */
#define ACMD23  (0x80 + 23) /* SET_WR_BLK_ERASE_COUNT (SDC) */
#define CMD24   (24)        /* WRITE_BLOCK */
#define CMD25   (25)        /* WRITE_MULTIPLE_BLOCK */
#define CMD55   (55)        /* APP_CMD */
#define CMD58   (58)        /* READ_OCR */

/* Card type flags (CardType) */
#define CT_MMC   0x01
#define CT_SD1   0x02
#define CT_SD2   0x04
#define CT_SDC   (CT_SD1 | CT_SD2)
#define CT_BLOCK 0x08

/* ------------------------------------------------------------------------- */
/* Module state                                                              */
/* ------------------------------------------------------------------------- */
static volatile DSTATUS Stat = STA_NOINIT;
static BYTE CardType;

/* ------------------------------------------------------------------------- */
/* Low-level SPI / CS / timing primitives                                    */
/* ------------------------------------------------------------------------- */

static void cs_low(void)  { HAL_GPIO_WritePin(CS_SDCard0_GPIO_Port, CS_SDCard0_Pin, GPIO_PIN_RESET); }
static void cs_high(void) { HAL_GPIO_WritePin(CS_SDCard0_GPIO_Port, CS_SDCard0_Pin, GPIO_PIN_SET); }

static BYTE xchg_spi(BYTE dat) {
    BYTE rx = 0xFF;
    HAL_SPI_TransmitReceive(&hspi3, &dat, &rx, 1, HAL_MAX_DELAY);
    return rx;
}

static void rcvr_spi_multi(BYTE *buff, UINT btr) {
    static const BYTE ones = 0xFF;
    /* Half-duplex bulk receive: clock out 0xFF, store the response. */
    for (UINT i = 0; i < btr; i++) {
        HAL_SPI_TransmitReceive(&hspi3, (uint8_t *)&ones, &buff[i], 1, HAL_MAX_DELAY);
    }
}

static void xmit_spi_multi(const BYTE *buff, UINT btx) {
    HAL_SPI_Transmit(&hspi3, (uint8_t *)buff, btx, HAL_MAX_DELAY);
}

/* SPI prescaler swap. The H7 SPI peripheral cannot change baud while enabled,
   so we DeInit/Init around the change. Mode 0, soft NSS, NSSP disable, etc.
   are preserved from CubeMX. */
static void sd_set_prescaler(uint32_t pres) {
    HAL_SPI_DeInit(&hspi3);
    hspi3.Init.BaudRatePrescaler = pres;
    HAL_SPI_Init(&hspi3);
}

void sd_spi_set_slow_clock(void) { sd_set_prescaler(SPI_BAUDRATEPRESCALER_256); }
void sd_spi_set_fast_clock(void) { sd_set_prescaler(SPI_BAUDRATEPRESCALER_16);  }

/* Wait until card is ready (DO drives 0xFF). Bound by HAL_GetTick() ms. */
static int wait_ready(UINT wait_ms) {
    uint32_t t0 = HAL_GetTick();
    BYTE d;
    do {
        d = xchg_spi(0xFF);
    } while (d != 0xFF && (HAL_GetTick() - t0) < wait_ms);
    return (d == 0xFF) ? 1 : 0;
}

static void deselect(void) {
    cs_high();
    xchg_spi(0xFF);     /* dummy clocks to release MISO */
}

static int select_card(void) {
    cs_low();
    xchg_spi(0xFF);     /* dummy clock to force DO state */
    if (wait_ready(500)) return 1;
    deselect();
    return 0;
}

/* Receive a data block of len btr (typ 512), framed by token 0xFE + 16-bit CRC. */
static int rcvr_datablock(BYTE *buff, UINT btr) {
    BYTE token;
    uint32_t t0 = HAL_GetTick();
    do {
        token = xchg_spi(0xFF);
    } while (token == 0xFF && (HAL_GetTick() - t0) < 200);
    if (token != 0xFE) return 0;
    rcvr_spi_multi(buff, btr);
    xchg_spi(0xFF);     /* discard CRC */
    xchg_spi(0xFF);
    return 1;
}

#if _USE_WRITE
/* Send a data block of 512 bytes with token. */
static int xmit_datablock(const BYTE *buff, BYTE token) {
    if (!wait_ready(500)) return 0;
    xchg_spi(token);
    if (token != 0xFD) {            /* token 0xFD = StopTran (no data) */
        xmit_spi_multi(buff, 512);
        xchg_spi(0xFF); xchg_spi(0xFF); /* dummy CRC */
        BYTE resp = xchg_spi(0xFF);
        if ((resp & 0x1F) != 0x05) return 0;  /* data response: accepted */
    }
    return 1;
}
#endif

/* Issue a command and return R1. ACMDs (cmd MSB set) prepend CMD55. */
static BYTE send_cmd(BYTE cmd, DWORD arg) {
    BYTE n, res;

    if (cmd & 0x80) {
        cmd &= 0x7F;
        res = send_cmd(CMD55, 0);
        if (res > 1) return res;
    }

    /* Toggle CS to start a fresh command frame. */
    if (cmd != CMD12) {
        deselect();
        if (!select_card()) return 0xFF;
    }

    BYTE buf[6];
    buf[0] = 0x40 | cmd;
    buf[1] = (BYTE)(arg >> 24);
    buf[2] = (BYTE)(arg >> 16);
    buf[3] = (BYTE)(arg >> 8);
    buf[4] = (BYTE)arg;
    n = 0x01;                       /* dummy CRC + stop */
    if (cmd == CMD0)  n = 0x95;
    if (cmd == CMD8)  n = 0x87;
    buf[5] = n;
    xmit_spi_multi(buf, 6);

    /* Receive response. */
    if (cmd == CMD12) xchg_spi(0xFF);   /* skip stuff byte */
    n = 10;
    do {
        res = xchg_spi(0xFF);
    } while ((res & 0x80) && --n);
    return res;
}

/* USER CODE END DECL */

/* Private function prototypes -----------------------------------------------*/
DSTATUS USER_initialize (BYTE pdrv);
DSTATUS USER_status (BYTE pdrv);
DRESULT USER_read (BYTE pdrv, BYTE *buff, DWORD sector, UINT count);
#if _USE_WRITE == 1
  DRESULT USER_write (BYTE pdrv, const BYTE *buff, DWORD sector, UINT count);
#endif /* _USE_WRITE == 1 */
#if _USE_IOCTL == 1
  DRESULT USER_ioctl (BYTE pdrv, BYTE cmd, void *buff);
#endif /* _USE_IOCTL == 1 */

Diskio_drvTypeDef  USER_Driver =
{
  USER_initialize,
  USER_status,
  USER_read,
#if  _USE_WRITE
  USER_write,
#endif  /* _USE_WRITE == 1 */
#if  _USE_IOCTL == 1
  USER_ioctl,
#endif /* _USE_IOCTL == 1 */
};

/* Private functions ---------------------------------------------------------*/

DSTATUS USER_initialize (
	BYTE pdrv
)
{
  /* USER CODE BEGIN INIT */
    BYTE n, cmd, ty, ocr[4];

    if (pdrv != 0) return STA_NOINIT;

    /* Slow clock, dummy clocks with CS high to enter SPI mode. */
    sd_spi_set_slow_clock();
    cs_high();
    for (n = 0; n < 10; n++) xchg_spi(0xFF);   /* >=74 dummy clocks */

    ty = 0;
    if (send_cmd(CMD0, 0) == 1) {              /* idle state */
        uint32_t t0 = HAL_GetTick();
        if (send_cmd(CMD8, 0x1AA) == 1) {       /* SDv2 */
            for (n = 0; n < 4; n++) ocr[n] = xchg_spi(0xFF);
            if (ocr[2] == 0x01 && ocr[3] == 0xAA) {     /* 2.7-3.6V range */
                while ((HAL_GetTick() - t0) < 1000 &&
                       send_cmd(ACMD41, 0x40000000)) { /* HCS=1 */ }
                if ((HAL_GetTick() - t0) < 1000 && send_cmd(CMD58, 0) == 0) {
                    for (n = 0; n < 4; n++) ocr[n] = xchg_spi(0xFF);
                    ty = (ocr[0] & 0x40) ? CT_SD2 | CT_BLOCK : CT_SD2;
                }
            }
        } else {
            if (send_cmd(ACMD41, 0) <= 1) { ty = CT_SD1; cmd = ACMD41; }
            else                          { ty = CT_MMC; cmd = CMD1;   }
            while ((HAL_GetTick() - t0) < 1000 && send_cmd(cmd, 0)) { }
            if ((HAL_GetTick() - t0) >= 1000 || send_cmd(CMD16, 512) != 0) {
                ty = 0;
            }
        }
    }
    CardType = ty;
    deselect();

    if (ty) {
        Stat &= ~STA_NOINIT;
        sd_spi_set_fast_clock();    /* bulk-transfer speed */
    } else {
        Stat = STA_NOINIT;
    }
    return Stat;
  /* USER CODE END INIT */
}

DSTATUS USER_status (
	BYTE pdrv
)
{
  /* USER CODE BEGIN STATUS */
    if (pdrv != 0) return STA_NOINIT;
    return Stat;
  /* USER CODE END STATUS */
}

DRESULT USER_read (
	BYTE pdrv,
	BYTE *buff,
	DWORD sector,
	UINT count
)
{
  /* USER CODE BEGIN READ */
    if (pdrv != 0 || !count) return RES_PARERR;
    if (Stat & STA_NOINIT)   return RES_NOTRDY;

    if (!(CardType & CT_BLOCK)) sector *= 512; /* byte addressing on non-SDHC */

    if (count == 1) {
        if ((send_cmd(CMD17, sector) == 0) && rcvr_datablock(buff, 512)) {
            count = 0;
        }
    } else {
        if (send_cmd(CMD18, sector) == 0) {
            do {
                if (!rcvr_datablock(buff, 512)) break;
                buff += 512;
            } while (--count);
            send_cmd(CMD12, 0);
        }
    }
    deselect();
    return count ? RES_ERROR : RES_OK;
  /* USER CODE END READ */
}

#if _USE_WRITE == 1
DRESULT USER_write (
	BYTE pdrv,
	const BYTE *buff,
	DWORD sector,
	UINT count
)
{
  /* USER CODE BEGIN WRITE */
    if (pdrv != 0 || !count) return RES_PARERR;
    if (Stat & STA_NOINIT)   return RES_NOTRDY;
    if (Stat & STA_PROTECT)  return RES_WRPRT;

    if (!(CardType & CT_BLOCK)) sector *= 512;

    if (count == 1) {
        if ((send_cmd(CMD24, sector) == 0) && xmit_datablock(buff, 0xFE)) {
            count = 0;
        }
    } else {
        if (CardType & CT_SDC) send_cmd(ACMD23, count);
        if (send_cmd(CMD25, sector) == 0) {
            do {
                if (!xmit_datablock(buff, 0xFC)) break;
                buff += 512;
            } while (--count);
            if (!xmit_datablock(0, 0xFD)) count = 1;        /* StopTran */
        }
    }
    deselect();
    return count ? RES_ERROR : RES_OK;
  /* USER CODE END WRITE */
}
#endif /* _USE_WRITE == 1 */

#if _USE_IOCTL == 1
DRESULT USER_ioctl (
	BYTE pdrv,
	BYTE cmd,
	void *buff
)
{
  /* USER CODE BEGIN IOCTL */
    if (pdrv != 0)         return RES_PARERR;
    if (Stat & STA_NOINIT) return RES_NOTRDY;

    DRESULT res = RES_ERROR;
    BYTE csd[16];
    DWORD csize;

    switch (cmd) {
    case CTRL_SYNC:
        if (select_card()) res = RES_OK;
        deselect();
        break;

    case GET_SECTOR_COUNT:
        if ((send_cmd(CMD9, 0) == 0) && rcvr_datablock(csd, 16)) {
            if ((csd[0] >> 6) == 1) {           /* SDv2 CSD */
                csize = ((DWORD)(csd[7] & 0x3F) << 16)
                      | ((DWORD)csd[8]  << 8)
                      |  (DWORD)csd[9];
                *(DWORD *)buff = (csize + 1) * 1024;
            } else {                            /* SDv1 / MMC CSD */
                BYTE n = (csd[5] & 0x0F) + ((csd[10] & 0x80) >> 7) + ((csd[9] & 0x03) << 1) + 2;
                csize = (csd[8] >> 6) | ((WORD)csd[7] << 2) | ((WORD)(csd[6] & 0x03) << 10);
                *(DWORD *)buff = (DWORD)(csize + 1) << (n - 9);
            }
            res = RES_OK;
        }
        deselect();
        break;

    case GET_BLOCK_SIZE:
        *(DWORD *)buff = 128;       /* erase-block size in 512B units, generic default */
        res = RES_OK;
        break;

    default:
        res = RES_PARERR;
        break;
    }
    return res;
  /* USER CODE END IOCTL */
}
#endif /* _USE_IOCTL == 1 */
