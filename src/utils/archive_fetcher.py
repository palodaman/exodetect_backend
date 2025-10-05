"""
NASA Exoplanet Archive data fetcher
Fetches target information and light curves from NASA archives
"""

import requests
import pandas as pd
import numpy as np
from typing import Dict, Optional, Tuple, Any
from astroquery.mast import Observations
from astroquery.exoplanet_archive import NasaExoplanetArchive
import warnings

warnings.filterwarnings('ignore')


class ArchiveFetcher:
    """Fetch exoplanet data from NASA Exoplanet Archive and MAST"""

    def __init__(self):
        self.kepler_koi_table = None
        self.tess_toi_table = None

    def fetch_koi_data(self, koi_id: str) -> Dict[str, Any]:
        """
        Fetch KOI (Kepler Object of Interest) data

        Parameters:
        -----------
        koi_id : str
            KOI identifier (e.g., 'K00123.01' or 'KOI-123.01')

        Returns:
        --------
        dict : Archive data for the KOI
        """
        try:
            # Normalize KOI ID format
            koi_id = self._normalize_koi_id(koi_id)

            # Query NASA Exoplanet Archive
            table = NasaExoplanetArchive.query_criteria(
                table='cumulative',
                where=f"kepoi_name like '{koi_id}'"
            )

            if len(table) == 0:
                raise ValueError(f"KOI {koi_id} not found in archive")

            # Get first match
            row = table[0]

            # Extract relevant fields
            archive_data = {
                'kepoi_name': str(row.get('kepoi_name', koi_id)),
                'kepid': int(row.get('kepid', 0)) if row.get('kepid') else None,
                'koi_disposition': str(row.get('koi_disposition', 'UNKNOWN')),
                'koi_pdisposition': str(row.get('koi_pdisposition', 'UNKNOWN')),
                'koi_score': float(row.get('koi_score', 0.0)) if row.get('koi_score') else None,
                'koi_period': float(row.get('koi_period', 0.0)) if row.get('koi_period') else None,
                'koi_duration': float(row.get('koi_duration', 0.0)) if row.get('koi_duration') else None,
                'koi_depth': float(row.get('koi_depth', 0.0)) if row.get('koi_depth') else None,
                'koi_prad': float(row.get('koi_prad', 0.0)) if row.get('koi_prad') else None,
                'koi_teq': float(row.get('koi_teq', 0.0)) if row.get('koi_teq') else None,
                'koi_insol': float(row.get('koi_insol', 0.0)) if row.get('koi_insol') else None,
                'koi_steff': float(row.get('koi_steff', 0.0)) if row.get('koi_steff') else None,
                'koi_slogg': float(row.get('koi_slogg', 0.0)) if row.get('koi_slogg') else None,
                'koi_srad': float(row.get('koi_srad', 0.0)) if row.get('koi_srad') else None,
            }

            # Vetting flags
            archive_data['vetting_flags'] = {
                'ntl': bool(row.get('koi_flag_ntl', False)),  # Not transit-like
                'ss': bool(row.get('koi_flag_ss', False)),    # Stellar eclipse
                'co': bool(row.get('koi_flag_co', False)),    # Centroid offset
                'em': bool(row.get('koi_flag_em', False))     # Ephemeris match
            }

            return archive_data

        except Exception as e:
            raise ValueError(f"Error fetching KOI data: {str(e)}")

    def fetch_toi_data(self, toi_id: str) -> Dict[str, Any]:
        """
        Fetch TOI (TESS Object of Interest) data

        Parameters:
        -----------
        toi_id : str
            TOI identifier (e.g., 'TOI-123.01')

        Returns:
        --------
        dict : Archive data for the TOI
        """
        try:
            # Normalize TOI ID
            toi_id = self._normalize_toi_id(toi_id)

            # Query NASA Exoplanet Archive for TESS
            table = NasaExoplanetArchive.query_criteria(
                table='toi',
                where=f"toi like '{toi_id}'"
            )

            if len(table) == 0:
                raise ValueError(f"TOI {toi_id} not found in archive")

            row = table[0]

            archive_data = {
                'toi': str(row.get('toi', toi_id)),
                'tid': int(row.get('tid', 0)) if row.get('tid') else None,
                'toi_disposition': str(row.get('tfopwg_disp', 'UNKNOWN')),
                'toi_period': float(row.get('pl_orbper', 0.0)) if row.get('pl_orbper') else None,
                'toi_duration': float(row.get('pl_trandur', 0.0)) if row.get('pl_trandur') else None,
                'toi_depth': float(row.get('pl_trandep', 0.0)) if row.get('pl_trandep') else None,
                'toi_prad': float(row.get('pl_rade', 0.0)) if row.get('pl_rade') else None,
                'st_teff': float(row.get('st_teff', 0.0)) if row.get('st_teff') else None,
                'st_logg': float(row.get('st_logg', 0.0)) if row.get('st_logg') else None,
                'st_rad': float(row.get('st_rad', 0.0)) if row.get('st_rad') else None,
            }

            return archive_data

        except Exception as e:
            raise ValueError(f"Error fetching TOI data: {str(e)}")

    def fetch_light_curve(self, identifier: str, mission: str = 'Kepler') -> Tuple[np.ndarray, np.ndarray, Optional[np.ndarray]]:
        """
        Fetch light curve data from MAST

        Parameters:
        -----------
        identifier : str
            Target identifier (KIC, TIC, EPIC, KOI, TOI)
        mission : str
            Mission name ('Kepler', 'K2', 'TESS')

        Returns:
        --------
        tuple : (time, flux, flux_err) arrays
        """
        try:
            # Determine target ID
            target_id = self._parse_identifier(identifier, mission)

            # Query MAST for observations
            if mission.lower() == 'kepler':
                obs_table = Observations.query_criteria(
                    target_name=target_id,
                    obs_collection='Kepler'
                )
            elif mission.lower() == 'k2':
                obs_table = Observations.query_criteria(
                    target_name=target_id,
                    obs_collection='K2'
                )
            elif mission.lower() == 'tess':
                obs_table = Observations.query_criteria(
                    target_name=target_id,
                    obs_collection='TESS'
                )
            else:
                raise ValueError(f"Unsupported mission: {mission}")

            if len(obs_table) == 0:
                raise ValueError(f"No observations found for {identifier}")

            # Get data products
            data_products = Observations.get_product_list(obs_table[0])

            # Filter for light curve files
            lc_products = data_products[
                (data_products['productType'] == 'SCIENCE') &
                (data_products['description'].str.contains('Light curve', case=False, na=False))
            ]

            if len(lc_products) == 0:
                raise ValueError(f"No light curve data found for {identifier}")

            # Download first light curve file
            manifest = Observations.download_products(lc_products[0:1], download_dir='./temp_lc')

            # Read FITS file
            from astropy.io import fits

            fits_file = manifest['Local Path'][0]
            with fits.open(fits_file) as hdul:
                data = hdul[1].data
                time = data['TIME']
                flux = data['PDCSAP_FLUX'] if 'PDCSAP_FLUX' in data.columns.names else data['SAP_FLUX']
                flux_err = data['PDCSAP_FLUX_ERR'] if 'PDCSAP_FLUX_ERR' in data.columns.names else None

            # Clean NaN values
            mask = np.isfinite(time) & np.isfinite(flux)
            time = time[mask]
            flux = flux[mask]
            if flux_err is not None:
                flux_err = flux_err[mask]

            # Normalize flux
            median_flux = np.median(flux)
            flux = flux / median_flux
            if flux_err is not None:
                flux_err = flux_err / median_flux

            # Clean up temp files
            import os
            import shutil
            if os.path.exists('./temp_lc'):
                shutil.rmtree('./temp_lc')

            return time, flux, flux_err

        except Exception as e:
            raise ValueError(f"Error fetching light curve: {str(e)}")

    def _normalize_koi_id(self, koi_id: str) -> str:
        """Normalize KOI ID to archive format"""
        koi_id = koi_id.upper().strip()

        # Remove common prefixes
        koi_id = koi_id.replace('KOI-', '').replace('KOI ', '').replace('K', '')

        # Ensure proper format: K#####.##
        if '.' not in koi_id:
            koi_id = f"{koi_id}.01"

        parts = koi_id.split('.')
        number = parts[0].zfill(5)
        planet = parts[1].zfill(2)

        return f"K{number}.{planet}"

    def _normalize_toi_id(self, toi_id: str) -> str:
        """Normalize TOI ID to archive format"""
        toi_id = toi_id.upper().strip()
        toi_id = toi_id.replace('TOI-', '').replace('TOI ', '')

        if '.' not in toi_id:
            toi_id = f"{toi_id}.01"

        return toi_id

    def _parse_identifier(self, identifier: str, mission: str) -> str:
        """Parse identifier to get target ID for MAST query"""
        identifier = identifier.upper().strip()

        # Extract numeric ID
        if 'KIC' in identifier or mission.lower() == 'kepler':
            # Extract KIC number
            kic_id = identifier.replace('KIC', '').replace('-', '').replace(' ', '').strip()
            return f"KIC {kic_id}"

        elif 'EPIC' in identifier or mission.lower() == 'k2':
            # Extract EPIC number
            epic_id = identifier.replace('EPIC', '').replace('-', '').replace(' ', '').strip()
            return f"EPIC {epic_id}"

        elif 'TIC' in identifier or mission.lower() == 'tess':
            # Extract TIC number
            tic_id = identifier.replace('TIC', '').replace('-', '').replace(' ', '').strip()
            return f"TIC {tic_id}"

        elif 'KOI' in identifier:
            # Get KIC from KOI
            koi_data = self.fetch_koi_data(identifier)
            kepid = koi_data.get('kepid')
            if kepid:
                return f"KIC {kepid}"

        elif 'TOI' in identifier:
            # Get TIC from TOI
            toi_data = self.fetch_toi_data(identifier)
            tid = toi_data.get('tid')
            if tid:
                return f"TIC {tid}"

        return identifier


def test_archive_fetcher():
    """Test the archive fetcher"""
    fetcher = ArchiveFetcher()

    # Test KOI fetch
    print("Testing KOI fetch...")
    try:
        koi_data = fetcher.fetch_koi_data("KOI-123.01")
        print(f"✓ KOI data fetched: {koi_data['kepoi_name']}")
        print(f"  Disposition: {koi_data['koi_disposition']}")
        print(f"  Period: {koi_data['koi_period']} days")
    except Exception as e:
        print(f"✗ KOI fetch failed: {e}")

    # Test light curve fetch (commented out to avoid large download)
    # print("\nTesting light curve fetch...")
    # try:
    #     time, flux, flux_err = fetcher.fetch_light_curve("KIC 11446443", "Kepler")
    #     print(f"✓ Light curve fetched: {len(time)} data points")
    # except Exception as e:
    #     print(f"✗ Light curve fetch failed: {e}")


if __name__ == "__main__":
    test_archive_fetcher()
