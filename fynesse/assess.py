import pandas as pd
from typing import List, Tuple, Optional
import librosa
import numpy as np


class DataAssessment:
    def __init__(self, data_access):
        self.data_access = data_access
        self.default_features = [
            ("building", None),
            ("amenity", None),
            ("amenity", "school"),
            ("amenity", "hospital"),
            ("amenity", "restaurant"),
            ("amenity", "cafe"),
            ("shop", None),
            ("tourism", None),
            ("tourism", "hotel"),
            ("tourism", "museum"),
            ("leisure", None),
            ("leisure", "park"),
            ("historic", None),
            ("amenity", "place_of_worship"),
        ]

    def assess_poi_distribution(self, features: Optional[List[Tuple]] = None):
        if self.data_access.pois is None:
            self.data_access.access_pois()
        if features is None:
            features = self.default_features
            
        pois_df = pd.DataFrame(self.data_access.pois)
        pois_df['latitude'] = pois_df.apply(lambda row: row.geometry.centroid.y, axis=1)
        pois_df['longitude'] = pois_df.apply(lambda row: row.geometry.centroid.x, axis=1)
        
        poi_counts = {}
        for key, value in features:
            if key in pois_df.columns:
                if value:
                    poi_counts[f"{key}:{value}"] = (pois_df[key] == value).sum()
                else:
                    poi_counts[key] = pois_df[key].notnull().sum()
            else:
                poi_counts[f"{key}:{value}" if value else key] = 0
        
        results = pd.DataFrame(list(poi_counts.items()), columns=["POI Type", "Count"])
        print("POI Assessment Summary:")
        print(results)
        return results

    def assess_esc50_distribution(self):
        """Summarize class distribution from ESC-50 metadata."""
        if not hasattr(self.data_access, "esc50_meta"):
            raise ValueError("ESC-50 data not loaded in DataAccess")

        df = self.data_access.esc50_meta
        summary = df.groupby("category").size().reset_index(name="count")
        print("ESC-50 Category Distribution:")
        print(summary)
        return summary

    def assess_audio_duration(self, sample: int = 100):
        """Compute duration stats for a random sample of ESC-50 audio."""
        if not hasattr(self.data_access, "esc50_meta"):
            raise ValueError("ESC-50 data not loaded in DataAccess")

        df = self.data_access.esc50_meta.sample(min(sample, len(self.data_access.esc50_meta)))
        durations = []
        for path in df["file_path"]:  # requires DataAccess to add this
            y, sr = librosa.load(path, sr=None)
            durations.append(len(y) / sr)

        print(f"Average duration: {np.mean(durations):.2f}s, Std: {np.std(durations):.2f}s")
        return durations
