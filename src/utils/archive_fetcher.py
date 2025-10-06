"""
NASA Exoplanet Archive data fetcher
Fetches target information and light curves from NASA archives
"""

import requests
import pandas as pd
import numpy as np
from typing import Dict, Optional, Tuple, Any
import warnings

warnings.filterwarnings('ignore')

# Try to import astroquery components
ASTROQUERY_AVAILABLE = False
MAST_AVAILABLE = False

try:
    from astroquery.mast import Observations
    MAST_AVAILABLE = True
except ImportError:
    print("Warning: astroquery.mast not available. Light curve fetching disabled.")

try:
    # Try new import path (astroquery >= 0.4.6)
    from astroquery.ipac.nexsci.nasa_exoplanet_archive import NasaExoplanetArchive
    ASTROQUERY_AVAILABLE = True
except ImportError:
    try:
        # Fallback to old import path
        from astroquery.nasa_exoplanet_archive import NasaExoplanetArchive
        ASTROQUERY_AVAILABLE = True
    except ImportError:
        print("Warning: astroquery NASA Exoplanet Archive not available. Using direct API fallback.")
        NasaExoplanetArchive = None


class ArchiveFetcher:
    """Fetch exoplanet data from NASA Exoplanet Archive and MAST"""

    def __init__(self):
        self.kepler_koi_table = None
        self.tess_toi_table = None

    def _fetch_koi_data_direct(self, koi_id: str) -> Dict[str, Any]:
        """Fetch KOI data using direct HTTP API"""
        try:
            # NASA TAP service URL
            url = "https://exoplanetarchive.ipac.caltech.edu/TAP/sync"

            # Normalize KOI ID
            koi_id = self._normalize_koi_id(koi_id)

            # Build query (single line to avoid whitespace issues)
            # Note: Using koi_fpflag_* (false positive flags) not koi_flag_*
            query = f"SELECT kepoi_name, kepid, koi_disposition, koi_pdisposition, koi_score, koi_period, koi_duration, koi_depth, koi_prad, koi_teq, koi_insol, koi_steff, koi_slogg, koi_srad, koi_fpflag_nt, koi_fpflag_ss, koi_fpflag_co, koi_fpflag_ec FROM cumulative WHERE kepoi_name = '{koi_id}'"

            params = {
                'query': query,
                'format': 'json'
            }

            response = requests.get(url, params=params, timeout=30)

            # Check for errors
            if response.status_code != 200:
                error_text = response.text[:200]  # First 200 chars of error
                raise ValueError(f"NASA API returned {response.status_code}: {error_text}")

            try:
                data = response.json()
            except Exception as e:
                raise ValueError(f"Failed to parse NASA response: {str(e)}")

            if not data or len(data) == 0:
                raise ValueError(f"KOI {koi_id} not found in archive. Query: {query}")

            row = data[0]

            # Helper to safely convert values
            def safe_float(val, default=None):
                if val is None or val == '':
                    return default
                try:
                    return float(val)
                except (ValueError, TypeError):
                    return default

            def safe_int(val, default=None):
                if val is None or val == '':
                    return default
                try:
                    return int(val)
                except (ValueError, TypeError):
                    return default

            # Extract relevant fields
            archive_data = {
                'kepoi_name': str(row.get('kepoi_name', koi_id)),
                'kepid': safe_int(row.get('kepid')),
                'koi_disposition': str(row.get('koi_disposition') or 'UNKNOWN'),
                'koi_pdisposition': str(row.get('koi_pdisposition') or 'UNKNOWN'),
                'koi_score': safe_float(row.get('koi_score')),
                'koi_period': safe_float(row.get('koi_period')),
                'koi_duration': safe_float(row.get('koi_duration')),
                'koi_depth': safe_float(row.get('koi_depth')),
                'koi_prad': safe_float(row.get('koi_prad')),
                'koi_teq': safe_float(row.get('koi_teq')),
                'koi_insol': safe_float(row.get('koi_insol')),
                'koi_steff': safe_float(row.get('koi_steff')),
                'koi_slogg': safe_float(row.get('koi_slogg')),
                'koi_srad': safe_float(row.get('koi_srad')),
            }

            # False positive flags (vetting flags)
            archive_data['vetting_flags'] = {
                'not_transit_like': bool(row.get('koi_fpflag_nt', 0)),
                'stellar_eclipse': bool(row.get('koi_fpflag_ss', 0)),
                'centroid_offset': bool(row.get('koi_fpflag_co', 0)),
                'ephemeris_match': bool(row.get('koi_fpflag_ec', 0))
            }

            return archive_data

        except Exception as e:
            raise ValueError(f"Error fetching KOI data: {str(e)}")

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
        # Try astroquery first, fallback to direct API
        if ASTROQUERY_AVAILABLE and NasaExoplanetArchive is not None:
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

                # Helper function to safely extract values from MaskedColumn
                def safe_extract(row, key, dtype=None, default=None):
                    try:
                        val = row[key]
                        # Check if masked
                        if hasattr(val, 'mask') and val.mask:
                            return default
                        # Convert to appropriate type
                        if dtype == 'int':
                            return int(val) if val is not None else default
                        elif dtype == 'float':
                            return float(val) if val is not None else default
                        elif dtype == 'str':
                            return str(val) if val is not None else default
                        elif dtype == 'bool':
                            # Safely convert to bool
                            if val is None:
                                return default
                            return bool(int(val)) if str(val).isdigit() else bool(val)
                        else:
                            return val if val is not None else default
                    except (KeyError, ValueError, TypeError):
                        return default

                # Extract relevant fields
                archive_data = {
                    'kepoi_name': safe_extract(row, 'kepoi_name', 'str', koi_id),
                    'kepid': safe_extract(row, 'kepid', 'int'),
                    'koi_disposition': safe_extract(row, 'koi_disposition', 'str', 'UNKNOWN'),
                    'koi_pdisposition': safe_extract(row, 'koi_pdisposition', 'str', 'UNKNOWN'),
                    'koi_score': safe_extract(row, 'koi_score', 'float'),
                    'koi_period': safe_extract(row, 'koi_period', 'float'),
                    'koi_duration': safe_extract(row, 'koi_duration', 'float'),
                    'koi_depth': safe_extract(row, 'koi_depth', 'float'),
                    'koi_prad': safe_extract(row, 'koi_prad', 'float'),
                    'koi_teq': safe_extract(row, 'koi_teq', 'float'),
                    'koi_insol': safe_extract(row, 'koi_insol', 'float'),
                    'koi_steff': safe_extract(row, 'koi_steff', 'float'),
                    'koi_slogg': safe_extract(row, 'koi_slogg', 'float'),
                    'koi_srad': safe_extract(row, 'koi_srad', 'float'),
                }

                # Vetting flags (try both flag names)
                archive_data['vetting_flags'] = {
                    'ntl': safe_extract(row, 'koi_fpflag_nt', 'bool', False) or safe_extract(row, 'koi_flag_ntl', 'bool', False),
                    'ss': safe_extract(row, 'koi_fpflag_ss', 'bool', False) or safe_extract(row, 'koi_flag_ss', 'bool', False),
                    'co': safe_extract(row, 'koi_fpflag_co', 'bool', False) or safe_extract(row, 'koi_flag_co', 'bool', False),
                    'em': safe_extract(row, 'koi_fpflag_ec', 'bool', False) or safe_extract(row, 'koi_flag_em', 'bool', False)
                }

                return archive_data

            except Exception as e:
                print(f"Astroquery failed, trying direct API: {e}")
                return self._fetch_koi_data_direct(koi_id)
        else:
            # Use direct API fallback
            return self._fetch_koi_data_direct(koi_id)

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
        if not MAST_AVAILABLE:
            raise ValueError("Light curve fetching requires astroquery.mast. Please install: pip install astroquery")

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
