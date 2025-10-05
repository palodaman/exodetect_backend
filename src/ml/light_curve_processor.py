import numpy as np
import pandas as pd
from scipy import signal, stats, optimize
from scipy.interpolate import interp1d
from astropy.stats import sigma_clip
from astropy.timeseries import BoxLeastSquares
import warnings
warnings.filterwarnings('ignore')


class LightCurveProcessor:
    """Process raw light curves to extract transit features"""

    def __init__(self):
        self.time = None
        self.flux = None
        self.flux_err = None
        self.cleaned_flux = None
        self.detrended_flux = None

    def load_light_curve(self, time, flux, flux_err=None):
        """Load light curve data"""
        self.time = np.array(time)
        self.flux = np.array(flux)
        self.flux_err = flux_err if flux_err is not None else np.ones_like(flux) * np.std(flux)

        # Remove NaN values
        mask = ~(np.isnan(self.time) | np.isnan(self.flux))
        self.time = self.time[mask]
        self.flux = self.flux[mask]
        self.flux_err = self.flux_err[mask]

        # Sort by time
        sort_idx = np.argsort(self.time)
        self.time = self.time[sort_idx]
        self.flux = self.flux[sort_idx]
        self.flux_err = self.flux_err[sort_idx]

        print(f"Loaded light curve with {len(self.time)} points")

    def clean_outliers(self, sigma=5):
        """Remove outliers using sigma clipping"""
        clipped = sigma_clip(self.flux, sigma=sigma, maxiters=5)
        mask = ~clipped.mask
        self.cleaned_flux = self.flux.copy()
        self.cleaned_flux[~mask] = np.interp(self.time[~mask], self.time[mask], self.flux[mask])
        return self.cleaned_flux

    def detrend(self, method='polynomial', window_length=None, poly_order=3):
        """Detrend the light curve"""
        if self.cleaned_flux is None:
            self.cleaned_flux = self.flux.copy()

        if method == 'polynomial':
            # Fit polynomial to the light curve
            coeffs = np.polyfit(self.time, self.cleaned_flux, poly_order)
            trend = np.polyval(coeffs, self.time)
            self.detrended_flux = self.cleaned_flux / trend

        elif method == 'median':
            # Median filter detrending
            if window_length is None:
                window_length = len(self.time) // 20

            # Ensure odd window length
            if window_length % 2 == 0:
                window_length += 1

            trend = signal.medfilt(self.cleaned_flux, kernel_size=window_length)
            self.detrended_flux = self.cleaned_flux / trend

        elif method == 'savgol':
            # Savitzky-Golay filter
            if window_length is None:
                window_length = len(self.time) // 10

            # Ensure odd window length
            if window_length % 2 == 0:
                window_length += 1

            trend = signal.savgol_filter(self.cleaned_flux, window_length, poly_order)
            self.detrended_flux = self.cleaned_flux / trend

        else:
            self.detrended_flux = self.cleaned_flux.copy()

        # Normalize
        self.detrended_flux = self.detrended_flux / np.median(self.detrended_flux)

        return self.detrended_flux

    def find_transits_bls(self, period_min=0.5, period_max=100, n_periods=5000):
        """Find transits using Box Least Squares (BLS) algorithm"""
        if self.detrended_flux is None:
            self.detrend()

        # Create BLS model
        model = BoxLeastSquares(self.time, self.detrended_flux, self.flux_err)

        # Period grid
        periods = np.linspace(period_min, period_max, n_periods)

        # Run BLS
        results = model.power(periods, 0.05)

        # Find best period
        best_idx = np.argmax(results.power)
        best_period = results.period[best_idx]
        best_t0 = results.transit_time[best_idx]
        best_duration = results.duration[best_idx]
        best_depth = results.depth[best_idx]
        best_snr = results.depth_snr[best_idx]

        # Calculate additional statistics
        stats = model.compute_stats(best_period, best_duration, best_t0)

        transit_params = {
            'period': best_period,
            'epoch': best_t0,
            'duration': best_duration * 24,  # Convert to hours
            'depth': best_depth * 1e6,  # Convert to ppm
            'snr': best_snr,
            'odd_even_mismatch': stats.get('odd_even_mismatch', 0),
            'harmonic_amplitude': stats.get('harmonic_amplitude', 0),
            'harmonic_delta': stats.get('harmonic_delta', 0),
            'transit_count': stats.get('transit_count', 0),
            'per_transit_count': stats.get('per_transit_count', []),
        }

        return transit_params

    def phase_fold(self, period, epoch=None):
        """Phase fold the light curve"""
        if epoch is None:
            epoch = self.time[0]

        phase = ((self.time - epoch) % period) / period
        phase[phase > 0.5] -= 1.0

        return phase

    def extract_transit_features(self, transit_params):
        """Extract detailed features from transit parameters"""
        features = {}

        # Basic transit parameters
        features['period'] = transit_params['period']
        features['duration'] = transit_params['duration']
        features['depth'] = transit_params['depth']
        features['snr'] = transit_params['snr']

        # Derived parameters
        features['period_to_duration'] = transit_params['period'] / (transit_params['duration'] / 24)

        # Impact parameter estimate (simplified)
        if transit_params['duration'] > 0 and transit_params['period'] > 0:
            duration_ratio = (transit_params['duration'] / 24) / transit_params['period']
            features['impact'] = np.sqrt(1 - duration_ratio ** 2) if duration_ratio < 1 else 0.5
        else:
            features['impact'] = 0.5

        # Transit shape parameters
        features['odd_even_mismatch'] = transit_params.get('odd_even_mismatch', 0)

        # Secondary eclipse detection (simplified - would need more sophisticated analysis)
        features['has_secondary'] = 0  # Placeholder

        # Vetting flags based on transit characteristics
        features['flag_not_transit'] = 1 if features['snr'] < 7 else 0
        features['flag_stellar_eclipse'] = 1 if features['odd_even_mismatch'] > 0.5 else 0
        features['flag_centroid_offset'] = 0  # Would need centroid data
        features['flag_ephemeris_match'] = 0  # Would need multiple transits

        # Log transformations
        features['log_period'] = np.log1p(features['period'])
        features['log_depth'] = np.log1p(features['depth'])
        features['log_snr'] = np.log1p(features['snr'])

        return features

    def process_light_curve_file(self, filepath, file_format='csv'):
        """Process a light curve from file"""
        if file_format == 'csv':
            df = pd.read_csv(filepath)
            time_col = 'time' if 'time' in df.columns else df.columns[0]
            flux_col = 'flux' if 'flux' in df.columns else df.columns[1]
            flux_err_col = 'flux_err' if 'flux_err' in df.columns else None

            time = df[time_col].values
            flux = df[flux_col].values
            flux_err = df[flux_err_col].values if flux_err_col and flux_err_col in df.columns else None

        elif file_format == 'fits':
            from astropy.io import fits
            with fits.open(filepath) as hdul:
                data = hdul[1].data
                time = data['TIME']
                flux = data['PDCSAP_FLUX'] if 'PDCSAP_FLUX' in data.columns.names else data['SAP_FLUX']
                flux_err = data['PDCSAP_FLUX_ERR'] if 'PDCSAP_FLUX_ERR' in data.columns.names else None

        else:
            raise ValueError(f"Unsupported file format: {file_format}")

        # Load and process
        self.load_light_curve(time, flux, flux_err)
        self.clean_outliers()
        self.detrend()

        # Find transits
        transit_params = self.find_transits_bls()

        # Extract features
        features = self.extract_transit_features(transit_params)

        return features, transit_params


class LightCurveBatchProcessor:
    """Process multiple light curves in batch"""

    def __init__(self):
        self.processor = LightCurveProcessor()

    def process_directory(self, directory_path, file_pattern='*.csv'):
        """Process all light curves in a directory"""
        import glob
        import os

        files = glob.glob(os.path.join(directory_path, file_pattern))
        results = []

        for filepath in files:
            try:
                print(f"Processing {filepath}...")
                features, transit_params = self.processor.process_light_curve_file(filepath)
                features['filename'] = os.path.basename(filepath)
                results.append(features)
            except Exception as e:
                print(f"Error processing {filepath}: {e}")
                continue

        return pd.DataFrame(results)


def simulate_light_curve(period=10, depth=0.01, duration=0.1, noise_level=0.001, n_points=10000):
    """Simulate a transit light curve for testing"""
    time = np.linspace(0, 100, n_points)
    flux = np.ones_like(time)

    # Add transits
    phase = (time % period) / period
    in_transit = np.abs(phase - 0.5) < (duration / 2 / period)
    flux[in_transit] -= depth

    # Add noise
    flux += np.random.normal(0, noise_level, n_points)

    return time, flux


if __name__ == "__main__":
    # Test with simulated data
    print("Testing with simulated light curve...")
    time, flux = simulate_light_curve(period=15, depth=0.005, duration=0.15)

    processor = LightCurveProcessor()
    processor.load_light_curve(time, flux)
    processor.clean_outliers()
    processor.detrend()

    transit_params = processor.find_transits_bls()
    features = processor.extract_transit_features(transit_params)

    print("\nDetected transit parameters:")
    for key, value in transit_params.items():
        if not isinstance(value, list):
            print(f"  {key}: {value:.4f}" if isinstance(value, float) else f"  {key}: {value}")

    print("\nExtracted features:")
    for key, value in features.items():
        if isinstance(value, float):
            print(f"  {key}: {value:.4f}")
        else:
            print(f"  {key}: {value}")