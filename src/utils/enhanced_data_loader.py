import pandas as pd
import numpy as np
import requests
from io import StringIO
import os

class EnhancedExoplanetDataLoader:
    def __init__(self):
        self.data_sources = {
            'koi': {
                'url': 'https://exoplanetarchive.ipac.caltech.edu/cgi-bin/nstedAPI/nph-nstedAPI',
                'table': 'cumulative',
                'prefix': 'koi'
            },
            'toi': {
                'url': 'https://exoplanetarchive.ipac.caltech.edu/cgi-bin/nstedAPI/nph-nstedAPI',
                'table': 'TOI',
                'prefix': 'toi'
            },
            'k2': {
                'url': 'https://exoplanetarchive.ipac.caltech.edu/cgi-bin/nstedAPI/nph-nstedAPI',
                'table': 'k2candidates',
                'prefix': 'k2'
            }
        }

        self.koi_features = [
            'kepoi_name', 'koi_disposition', 'koi_pdisposition', 'koi_score',
            'koi_period', 'koi_impact', 'koi_duration', 'koi_depth',
            'koi_prad', 'koi_teq', 'koi_insol', 'koi_model_snr',
            'koi_steff', 'koi_slogg', 'koi_srad', 'koi_kepmag',
            'koi_fpflag_nt', 'koi_fpflag_ss', 'koi_fpflag_co', 'koi_fpflag_ec'
        ]

        self.toi_features = [
            'toipfx', 'toi', 'tid', 'tfopwg_disp',
            'pl_orbper', 'pl_orbpererr1', 'pl_orbpererr2',
            'pl_trandur', 'pl_trandep', 'pl_rade',
            'pl_eqt', 'pl_insol', 'pl_trandep', 'st_teff',
            'st_logg', 'st_rad', 'st_tmag'
        ]

        self.k2_features = [
            'epic_name', 'k2_disp', 'pl_orbper', 'pl_trandur',
            'pl_trandep', 'pl_rade', 'pl_eqt', 'st_teff',
            'st_logg', 'st_rad'
        ]

    def download_dataset(self, dataset_name, save_path=None):
        """Download specific dataset from NASA Exoplanet Archive"""
        if dataset_name not in self.data_sources:
            raise ValueError(f"Unknown dataset: {dataset_name}")

        config = self.data_sources[dataset_name]

        if dataset_name == 'koi':
            columns = self.koi_features
        elif dataset_name == 'toi':
            columns = self.toi_features
        else:
            columns = self.k2_features

        params = {
            'table': config['table'],
            'format': 'csv',
            'select': ','.join([c for c in columns if c])
        }

        print(f"Downloading {dataset_name.upper()} data...")

        try:
            response = requests.get(config['url'], params=params, timeout=120)

            if response.status_code == 200:
                if save_path:
                    os.makedirs(os.path.dirname(save_path) if os.path.dirname(save_path) else '.', exist_ok=True)
                    with open(save_path, 'w', encoding='utf-8') as f:
                        f.write(response.text)
                    print(f"Data saved to {save_path}")

                df = pd.read_csv(StringIO(response.text), comment='#')
                print(f"Loaded {len(df)} {dataset_name.upper()} objects")
                return df
            else:
                print(f"Error downloading {dataset_name}: {response.status_code}")
                return None

        except Exception as e:
            print(f"Error downloading {dataset_name}: {e}")
            return None

    def standardize_features(self, df, dataset_name):
        """Standardize column names across different datasets"""
        standardized_df = df.copy()

        # Create mapping for standard feature names
        feature_mapping = {
            'toi': {
                'pl_orbper': 'period',
                'pl_trandur': 'duration',
                'pl_trandep': 'depth',
                'pl_rade': 'planet_radius',
                'pl_eqt': 'teq',
                'pl_insol': 'insol',
                'st_teff': 'star_teff',
                'st_logg': 'star_logg',
                'st_rad': 'star_radius',
                'st_tmag': 'magnitude',
                'tfopwg_disp': 'disposition'
            },
            'koi': {
                'koi_period': 'period',
                'koi_duration': 'duration',
                'koi_depth': 'depth',
                'koi_prad': 'planet_radius',
                'koi_teq': 'teq',
                'koi_insol': 'insol',
                'koi_model_snr': 'snr',
                'koi_steff': 'star_teff',
                'koi_slogg': 'star_logg',
                'koi_srad': 'star_radius',
                'koi_kepmag': 'magnitude',
                'koi_impact': 'impact',
                'koi_disposition': 'disposition',
                'koi_fpflag_nt': 'flag_not_transit',
                'koi_fpflag_ss': 'flag_stellar_eclipse',
                'koi_fpflag_co': 'flag_centroid_offset',
                'koi_fpflag_ec': 'flag_ephemeris_match'
            },
            'k2': {
                'pl_orbper': 'period',
                'pl_trandur': 'duration',
                'pl_trandep': 'depth',
                'pl_rade': 'planet_radius',
                'pl_eqt': 'teq',
                'st_teff': 'star_teff',
                'st_logg': 'star_logg',
                'st_rad': 'star_radius',
                'k2_disp': 'disposition'
            }
        }

        if dataset_name in feature_mapping:
            mapping = feature_mapping[dataset_name]
            for old_name, new_name in mapping.items():
                if old_name in standardized_df.columns:
                    standardized_df[new_name] = standardized_df[old_name]

        # Add source column
        standardized_df['source'] = dataset_name.upper()

        return standardized_df

    def create_labels(self, df):
        """Create binary labels from disposition columns"""
        df = df.copy()

        # Handle different disposition formats
        positive_labels = ['CONFIRMED', 'CANDIDATE', 'PC', 'CP', 'KP']
        negative_labels = ['FALSE POSITIVE', 'FP', 'FA', 'IS', 'V', 'O']

        def map_label(disp):
            if pd.isna(disp):
                return np.nan
            disp_upper = str(disp).upper()
            if any(pos in disp_upper for pos in positive_labels):
                return 1
            elif any(neg in disp_upper for neg in negative_labels):
                return 0
            else:
                return np.nan

        df['label'] = df['disposition'].apply(map_label)

        return df

    def combine_datasets(self, include_koi=True, include_toi=True, include_k2=False):
        """Combine multiple datasets into one training set"""
        datasets = []

        if include_koi:
            koi_df = self.download_dataset('koi', 'data/koi_cumulative.csv')
            if koi_df is not None:
                koi_std = self.standardize_features(koi_df, 'koi')
                datasets.append(koi_std)

        if include_toi:
            toi_df = self.download_dataset('toi', 'data/toi_catalog.csv')
            if toi_df is not None:
                toi_std = self.standardize_features(toi_df, 'toi')
                datasets.append(toi_std)

        if include_k2:
            k2_df = self.download_dataset('k2', 'data/k2_candidates.csv')
            if k2_df is not None:
                k2_std = self.standardize_features(k2_df, 'k2')
                datasets.append(k2_std)

        if not datasets:
            raise ValueError("No datasets could be loaded")

        # Combine all datasets
        combined_df = pd.concat(datasets, ignore_index=True)

        # Create labels
        combined_df = self.create_labels(combined_df)

        # Remove rows without labels
        combined_df = combined_df.dropna(subset=['label'])

        print(f"\nCombined dataset statistics:")
        print(f"Total samples: {len(combined_df)}")
        print(f"Sources: {combined_df['source'].value_counts().to_dict()}")
        print(f"Label distribution: {combined_df['label'].value_counts().to_dict()}")

        return combined_df

    def engineer_features(self, df):
        """Create engineered features"""
        df = df.copy()

        # Basic ratios
        if 'period' in df.columns and 'duration' in df.columns:
            df['period_to_duration'] = df['period'] / (df['duration'] / 24)

        if 'depth' in df.columns and 'snr' in df.columns:
            df['depth_snr_product'] = df['depth'] * df['snr']

        if 'insol' in df.columns and 'star_radius' in df.columns:
            df['stellar_flux'] = df['insol'] * df['star_radius'] ** 2

        # Log transformations
        for col in ['period', 'depth', 'snr', 'planet_radius']:
            if col in df.columns:
                df[f'log_{col}'] = np.log1p(df[col].fillna(0))

        # Temperature categories
        if 'teq' in df.columns:
            df['temp_category'] = pd.cut(df['teq'],
                                        bins=[0, 300, 500, 1000, 10000],
                                        labels=['cold', 'temperate', 'warm', 'hot'])
            df = pd.get_dummies(df, columns=['temp_category'], prefix='temp')

        # Planet size categories (Earth radii)
        if 'planet_radius' in df.columns:
            df['size_category'] = pd.cut(df['planet_radius'],
                                        bins=[0, 0.5, 1.25, 2, 4, 100],
                                        labels=['sub_earth', 'earth_like', 'super_earth', 'neptune', 'jupiter'])
            df = pd.get_dummies(df, columns=['size_category'], prefix='size')

        return df

    def get_training_data(self, test_size=0.2, random_state=42):
        """Get combined and processed training data"""
        # Combine datasets
        combined_df = self.combine_datasets(include_koi=True, include_toi=True, include_k2=False)

        # Engineer features
        combined_df = self.engineer_features(combined_df)

        # Select features for training
        feature_cols = [
            'period', 'duration', 'depth', 'planet_radius', 'teq', 'insol',
            'snr', 'star_teff', 'star_logg', 'star_radius', 'magnitude', 'impact',
            'flag_not_transit', 'flag_stellar_eclipse', 'flag_centroid_offset', 'flag_ephemeris_match',
            'period_to_duration', 'depth_snr_product', 'stellar_flux',
            'log_period', 'log_depth', 'log_snr', 'log_planet_radius'
        ]

        # Add any temp and size dummy columns
        temp_cols = [col for col in combined_df.columns if col.startswith('temp_')]
        size_cols = [col for col in combined_df.columns if col.startswith('size_')]
        feature_cols.extend(temp_cols)
        feature_cols.extend(size_cols)

        # Keep only available features
        available_features = [col for col in feature_cols if col in combined_df.columns]

        # Handle missing values
        for col in available_features:
            if col.startswith('flag_'):
                combined_df[col] = combined_df[col].fillna(0)
            else:
                combined_df[col] = combined_df[col].fillna(combined_df[col].median())

        X = combined_df[available_features]
        y = combined_df['label']

        from sklearn.model_selection import train_test_split
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=test_size, random_state=random_state, stratify=y
        )

        print(f"\nTraining set: {X_train.shape}")
        print(f"Test set: {X_test.shape}")
        print(f"Features used: {len(available_features)}")

        return X_train, X_test, y_train, y_test, available_features


if __name__ == "__main__":
    loader = EnhancedExoplanetDataLoader()
    X_train, X_test, y_train, y_test, features = loader.get_training_data()
    print("\nFeatures:", features)