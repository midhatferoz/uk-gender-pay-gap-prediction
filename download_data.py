"""Download the official UK gender pay gap CSVs into data/raw/ as gpg_<year>.csv.

Source: https://gender-pay-gap.service.gov.uk/viewing/download  (Open Government Licence v3.0)
File naming: <year> is the START of the reporting year, e.g. 2024 = reporting year 2024-25.

If this script is blocked or the URL pattern changes, download the CSVs manually from
the page above and save them as data/raw/gpg_2017.csv ... gpg_2025.csv.
NOTE: written without network access, so it has not been test-run; check the files it saves.
"""
import urllib.request

import config

URL = "https://gender-pay-gap.service.gov.uk/viewing/download-data/{year}"


def main():
    config.RAW_DIR.mkdir(parents=True, exist_ok=True)
    for year in range(config.FIRST_YEAR, config.LAST_YEAR + 1):
        dest = config.RAW_DIR / f"gpg_{year}.csv"
        if dest.exists():
            print(f"[have] {dest.name}")
            continue
        req = urllib.request.Request(URL.format(year=year),
                                     headers={"User-Agent": "Mozilla/5.0 (course project)"})
        try:
            with urllib.request.urlopen(req, timeout=120) as r:
                dest.write_bytes(r.read())
            print(f"[ok]   {dest.name}  ({dest.stat().st_size / 1e6:.1f} MB)")
        except Exception as exc:                      # noqa: BLE001
            print(f"[FAIL] {year}: {exc}  -> download it manually from the website")


if __name__ == "__main__":
    main()
