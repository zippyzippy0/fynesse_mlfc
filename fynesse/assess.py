import pandas as pd
import numpy as np
from scipy.stats import bernoulli, norm
import pymc as pm
import arviz as az
import geopandas as gpd
from shapely.geometry import Point

class DataAssessment:
    def __init__(self, data_access):
        self.data_access = data_access

    def assess_poi_distribution(self):
        if self.data_access.pois is None:
            self.data_access.access_pois()
        pois_df = pd.DataFrame(self.data_access.pois)
        counts = {
            "schools": len(self.data_access.schools) if self.data_access.schools is not None else 0,
            "hospitals": len(self.data_access.hospitals) if self.data_access.hospitals is not None else 0
        }
        results = pd.DataFrame(list(counts.items()), columns=["Facility Type", "Count"])
        print("Facility Assessment Summary:")
        print(results)
        return results

    def assess_bernoulli_access(self, households, facilities, max_distance=5000):
        results = []
        for h in households.geometry:
            dists = facilities.distance(h)
            accessible = int((dists.min() <= max_distance))
            results.append(accessible)
        prob = np.mean(results)
        print(f"Probability of access within {max_distance}m: {prob:.2f}")
        return bernoulli(prob), prob

    def assess_gaussian_distances(self, households, facilities):
        distances = []
        for h in households.geometry:
            dists = facilities.distance(h)
            distances.append(dists.min())
        mu, sigma = np.mean(distances), np.std(distances)
        print(f"Gaussian fit: mean={mu:.2f}m, std={sigma:.2f}m")
        return norm(mu, sigma), distances

    def assess_bayesian_regression(self, population_density, facility_counts):
        with pm.Model() as model:
            alpha = pm.Normal("alpha", mu=0, sigma=10)
            beta = pm.Normal("beta", mu=0, sigma=10)
            sigma = pm.HalfNormal("sigma", sigma=1)
            mu = alpha + beta * population_density
            y_obs = pm.Normal("y_obs", mu=mu, sigma=sigma, observed=facility_counts)
            trace = pm.sample(1000, tune=1000, target_accept=0.9, chains=2)
        summary = az.summary(trace)
        print("Bayesian Regression Summary:")
        print(summary)
        return model, trace, summary
