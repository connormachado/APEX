#!/bin/bash
set -e

echo "Configuring CMake..."
cmake --preset Debug

echo "Building project..."
cmake --build build/Debug

echo "Converting ELF to HEX..."
arm-none-eabi-objcopy -O ihex build/Debug/APEX_H7.elf build/Debug/APEX_H7.hex

echo "Checking ST-LINK..."
st-info --probe

echo "Flashing NUCLEO..."
st-flash --connect-under-reset --format ihex write build/Debug/APEX_H7.hex

echo "Done."
