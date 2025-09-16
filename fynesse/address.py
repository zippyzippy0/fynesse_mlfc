import matplotlib.pyplot as plt
import osmnx as ox
import geopandas as gpd
import pandas as pd
from typing import List, Tuple, Optional

class DataSolution:
    def __init__(self, data_access):
        self.data_access = data_access

    def address_visualization(self, figsize: Tuple[int, int] = (6, 6)):
        if any(data is None for data in [
            self.data_access.area, 
            self.data_access.buildings,
            self.data_access.edges, 
            self.data_access.nodes, 
            self.data_access.schools, 
            self.data_access.hospitals
        ]):
            self.data_access.access_all_data()
        
        fig, ax = plt.subplots(figsize=figsize)
        if self.data_access.area is not None:
            self.data_access.area.plot(ax=ax, color="tan", alpha=0.2)
        if self.data_access.buildings is not None:
            self.data_access.buildings.plot(ax=ax, facecolor="gray", edgecolor="gray", alpha=0.5)
        if self.data_access.edges is not None:
            self.data_access.edges.plot(ax=ax, linewidth=0.5, edgecolor="black", alpha=0.3)
        if self.data_access.schools is not None:
            self.data_access.schools.plot(ax=ax, color="blue", markersize=10, label="Schools")
        if self.data_access.hospitals is not None:
            self.data_access.hospitals.plot(ax=ax, color="green", markersize=10, label="Hospitals")
        plt.title(f"{self.data_access.place_name} – Schools & Hospitals", fontsize=14)
        plt.xlabel("Longitude")
        plt.ylabel("Latitude")
        plt.legend()
        plt.show()
        return fig

    def address_service_density(self, population_df: pd.DataFrame, admin_col="County"):
        results = []
        for county, group in population_df.groupby(admin_col):
            pop = group["Population"].sum()
            schools = len(self.data_access.schools) if self.data_access.schools is not None else 0
            hospitals = len(self.data_access.hospitals) if self.data_access.hospitals is not None else 0
            density = {
                "County": county,
                "Population": pop,
                "Schools per 100k": (schools / pop * 1e5) if pop > 0 else 0,
                "Hospitals per 100k": (hospitals / pop * 1e5) if pop > 0 else 0
            }
            results.append(density)
        df = pd.DataFrame(results).sort_values(by=["Schools per 100k", "Hospitals per 100k"], ascending=True)
        print("Service Density Ranking:")
        print(df)
        return df

    def address_rural_vs_urban(self, population_df: pd.DataFrame, urban_col="Urban", admin_col="County"):
        results = []
        for key, group in population_df.groupby([admin_col, urban_col]):
            pop = group["Population"].sum()
            schools = len(self.data_access.schools) if self.data_access.schools is not None else 0
            hospitals = len(self.data_access.hospitals) if self.data_access.hospitals is not None else 0
            results.append({
                "County": key[0],
                "Urban/Rural": key[1],
                "Population": pop,
                "Schools per 100k": (schools / pop * 1e5) if pop > 0 else 0,
                "Hospitals per 100k": (hospitals / pop * 1e5) if pop > 0 else 0
            })
        df = pd.DataFrame(results)
        print("Urban vs Rural Service Density:")
        print(df)
        return df

    def address_priority_regions(self, service_density_df, threshold=1.0):
        underserved = service_density_df[
            (service_density_df["Schools per 100k"] < threshold) |
            (service_density_df["Hospitals per 100k"] < threshold)
        ]
        print("Priority Regions (underserved):")
        print(underserved)
        return underserved
