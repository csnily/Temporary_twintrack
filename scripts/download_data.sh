#!/bin/bash
# Download datasets for TwinTrack
# Usage: bash scripts/download_data.sh
set -e
mkdir -p data
cd data

# Example: Download DCT dataset (replace with real link if available)
echo "Downloading DCT dataset..."
if [ ! -f DCT.zip ]; then
  wget -c https://example.com/DCT.zip -O DCT.zip
fi
if [ ! -d DCT ]; then
  unzip -q DCT.zip -d DCT
fi

# Example: Download AnimalTrack dataset
echo "Downloading AnimalTrack dataset..."
if [ ! -f AnimalTrack.zip ]; then
  wget -c https://example.com/AnimalTrack.zip -O AnimalTrack.zip
fi
if [ ! -d AnimalTrack ]; then
  unzip -q AnimalTrack.zip -d AnimalTrack
fi

# Example: Download BuckTales dataset
echo "Downloading BuckTales dataset..."
if [ ! -f BuckTales.zip ]; then
  wget -c https://example.com/BuckTales.zip -O BuckTales.zip
fi
if [ ! -d BuckTales ]; then
  unzip -q BuckTales.zip -d BuckTales
fi

# Example: Download HarvardCow dataset
echo "Downloading HarvardCow dataset..."
if [ ! -f HarvardCow.zip ]; then
  wget -c https://example.com/HarvardCow.zip -O HarvardCow.zip
fi
if [ ! -d HarvardCow ]; then
  unzip -q HarvardCow.zip -d HarvardCow
fi

echo "All datasets downloaded and extracted." 