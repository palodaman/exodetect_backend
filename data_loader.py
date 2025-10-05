import pandas as pd
import numpy as np
import requests
from io import StringIO
import os

class ExoplanetDataLoader:
    def __init__(self):
        self.koi_columns = [
            'kepoi_name', 'koi_disposition', 'koi_pdisposition', 'koi_score',
            'koi_period', 'koi_period_err1', 'koi_period_err2',
            'koi_time0bk', 'koi_time0bk_err1', 'koi_time0bk_err2',
            'koi_impact', 'koi_impact_err1', 'koi_impact_err2',
            'koi_duration', 'koi_duration_err1', 'koi_duration_err2',
            'koi_depth', 'koi_depth_err1', 'koi_depth_err2',
            'koi_prad', 'koi_prad_err1', 'koi_prad_err2',
            'koi_teq', 'koi_insol', 'koi_insol_err1', 'koi_insol_err2',
            'koi_model_snr', 'koi_tce_plnt_num', 'koi_steff', 'koi_steff_err1',
            'koi_steff_err2', 'koi_slogg', 'koi_slogg_err1', 'koi_slogg_err2',
            'koi_srad', 'koi_srad_err1', 'koi_srad_err2', 'ra', 'dec',
            'koi_kepmag', 'koi_gmag', 'koi_rmag', 'koi_imag', 'koi_zmag',
            'koi_jmag', 'koi_hmag', 'koi_kmag', 'koi_tce_delivname',
            'koi_sparprov', 'koi_limbdark_mod', 'koi_ldm_coeff1',
            'koi_ldm_coeff2', 'koi_ldm_coeff3', 'koi_ldm_coeff4',
            'koi_parm_prov', 'koi_max_sngle_ev', 'koi_max_mult_ev',
            'koi_model_dof', 'koi_model_chisq', 'koi_datalink_dvr',
            'koi_datalink_dvs', 'kepid', 'kepler_name', 'koi_vet_stat',
            'koi_vet_date', 'koi_disp_prov', 'koi_comment', 'koi_fpflag_nt',
            'koi_fpflag_ss', 'koi_fpflag_co', 'koi_fpflag_ec', 'koi_dicco_mra',
            'koi_dicco_mra_err1', 'koi_dicco_mra_err2', 'koi_dicco_mdec',
            'koi_dicco_mdec_err1', 'koi_dicco_mdec_err2', 'koi_dicco_msky',
            'koi_dicco_msky_err1', 'koi_dicco_msky_err2', 'koi_dicco_fra',
            'koi_dicco_fra_err1', 'koi_dicco_fra_err2', 'koi_dicco_fdec',
            'koi_dicco_fdec_err1', 'koi_dicco_fdec_err2', 'koi_dicco_fsky',
            'koi_dicco_fsky_err1', 'koi_dicco_fsky_err2'
        ]

        self.feature_columns = [
            'koi_period', 'koi_impact', 'koi_duration', 'koi_depth',
            'koi_prad', 'koi_teq', 'koi_insol', 'koi_model_snr',
            'koi_steff', 'koi_slogg', 'koi_srad', 'koi_kepmag',
            'koi_fpflag_nt', 'koi_fpflag_ss', 'koi_fpflag_co', 'koi_fpflag_ec'
        ]

    def download_koi_data(self, save_path='data/koi_cumulative.csv'):
        """Download KOI cumulative data from NASA Exoplanet Archive"""
        base_url = "https://exoplanetarchive.ipac.caltech.edu/TAP/sync"

        columns_str = ','.join(self.koi_columns)
        query = f"SELECT {columns_str} FROM koi WHERE koi_pdisposition IS NOT NULL"

        params = {
            'query': query,
            'format': 'csv'
        }

        print(f"Downloading KOI data...")
        response = requests.get(base_url, params=params)

        if response.status_code == 200:
            os.makedirs(os.path.dirname(save_path), exist_ok=True)
            with open(save_path, 'w') as f:
                f.write(response.text)
            print(f"Data saved to {save_path}")
            return pd.read_csv(StringIO(response.text))
        else:
            print(f"Error downloading data: {response.status_code}")
            return None

    def load_local_koi_data(self, file_path='data/koi_cumulative.csv'):
        """Load KOI data from local file"""
        if os.path.exists(file_path):
            return pd.read_csv(file_path)
        else:
            print(f"File not found at {file_path}. Downloading...")
            return self.download_koi_data(file_path)

    def preprocess_data(self, df):
        """Preprocess the KOI data for training"""
        df = df.copy()

        df['label'] = df['koi_disposition'].map({
            'CONFIRMED': 1,
            'CANDIDATE': 1,
            'FALSE POSITIVE': 0
        })

        df = df.dropna(subset=['label'])

        for col in self.feature_columns:
            if col in df.columns:
                if col.startswith('koi_fpflag'):
                    df[col] = df[col].fillna(0)
                else:
                    df[col] = df[col].fillna(df[col].median())

        df['period_to_duration_ratio'] = df['koi_period'] / (df['koi_duration'] / 24)
        df['transit_depth_snr'] = df['koi_depth'] * df['koi_model_snr']
        df['stellar_flux'] = df['koi_insol'] * df['koi_srad'] ** 2

        df['log_period'] = np.log1p(df['koi_period'])
        df['log_depth'] = np.log1p(df['koi_depth'])
        df['log_snr'] = np.log1p(df['koi_model_snr'])

        feature_cols = self.feature_columns + [
            'period_to_duration_ratio', 'transit_depth_snr', 'stellar_flux',
            'log_period', 'log_depth', 'log_snr'
        ]

        feature_cols = [col for col in feature_cols if col in df.columns]

        return df, feature_cols

    def get_train_test_data(self, test_size=0.2, random_state=42):
        """Load data and prepare for training"""
        df = self.load_local_koi_data()
        if df is None:
            raise Exception("Failed to load data")

        df_processed, feature_cols = self.preprocess_data(df)

        from sklearn.model_selection import train_test_split

        X = df_processed[feature_cols]
        y = df_processed['label']

        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=test_size, random_state=random_state, stratify=y
        )

        print(f"Training set: {X_train.shape[0]} samples")
        print(f"Test set: {X_test.shape[0]} samples")
        print(f"Features: {len(feature_cols)}")
        print(f"Class distribution - Train: {y_train.value_counts().to_dict()}")
        print(f"Class distribution - Test: {y_test.value_counts().to_dict()}")

        return X_train, X_test, y_train, y_test, feature_cols