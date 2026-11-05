#!/usr/bin/env bash
# Re-download the public raw datasets that are too large to ship (run from the project root).
set -e
mkdir -p data/raw/dl_roads data/raw/dl_counties data/raw/dl_counties2 data/raw/dl_cities data/raw/dl_places
curl -sSL -o data/raw/dl_roads/ne_10m_roads.geojson https://raw.githubusercontent.com/nvkelso/natural-earth-vector/master/geojson/ne_10m_roads.geojson
curl -sSL -o data/raw/dl_counties/county_centers.csv https://raw.githubusercontent.com/btskinner/spatial/master/data/county_centers.csv
curl -sSL -o data/raw/dl_counties2/counties.csv https://raw.githubusercontent.com/balsama/us_counties_data/main/data/counties.csv
curl -sSL -o data/raw/dl_cities/us-cities-top-1k.csv https://raw.githubusercontent.com/plotly/datasets/master/us-cities-top-1k.csv
curl -sSL -o data/raw/dl_places/us_cities.csv https://raw.githubusercontent.com/kelvins/US-Cities-Database/main/csv/us_cities.csv
echo "done. Supercharger (supercharge.info) and AFDC competitor extracts are already in data/raw/ (packed text files)."
