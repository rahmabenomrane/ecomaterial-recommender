import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics.pairwise import cosine_similarity
import sqlite3
import os
import warnings
warnings.filterwarnings('ignore')


class EcoMaterialRecommender:
    """Système principal de recommandation de matériaux éco-responsables"""

    def __init__(self):
        self.materials_df = None
        self.model = None
        self.scaler = StandardScaler()
        self.similarity_matrix = None
        self.db_path = "data/materials_database.db"

    def create_materials_database(self):
        """Créer la base de données avec des matériaux réels"""

        materials_data = {
            'material_name': [
                # Métaux
                'Acier recyclé', 'Acier vierge', 'Aluminium recyclé', 'Aluminium vierge',
                'Cuivre recyclé', 'Cuivre vierge', 'Titane recyclé', 'Zinc recyclé',

                # Construction
                'Béton ordinaire', 'Béton bas carbone', 'Béton chanvre', 'Béton recyclé',
                'Brique terre cuite', 'Brique terre crue', 'Pierre naturelle', 'Parpaing',

                # Bois et biosourcés
                'Bois FSC résineux', 'Bois tropical', 'Bambou Moso', 'Liège naturel',
                'Chanvre industriel', 'Lin technique', 'Paille compressée', 'Ouate cellulose',

                # Plastiques
                'PET recyclé', 'PET vierge', 'PE recyclé', 'PE vierge',
                'PP recyclé', 'PP vierge', 'PLA bioplastique', 'PHA bioplastique',

                # Composites
                'Fibre de verre', 'Fibre de carbone', 'Composite lin-résine', 'Composite chanvre',
                'Composite recyclé', 'Fibre basalte', 'Aramide recyclé', 'Bio-composite',

                # Isolants
                'Laine de roche', 'Polystyrène expansé', 'Polyuréthane', 'Laine de verre',
                'Liège expansé', 'Fibre de bois', 'Perlite', 'Vermiculite',

                # Innovants
                'Mycelium composite', 'Algues séchées', 'Papier stone', 'Géopolymère',
                'Aérogel biosourcé', 'Textile recyclé', 'Cuir végétal', 'Biociment'
            ],

            'category': [
                'Métaux', 'Métaux', 'Métaux', 'Métaux', 'Métaux', 'Métaux', 'Métaux', 'Métaux',
                'Construction', 'Construction', 'Construction', 'Construction', 'Construction', 'Construction', 'Construction', 'Construction',
                'Biosourcés', 'Bois', 'Biosourcés', 'Biosourcés', 'Biosourcés', 'Biosourcés', 'Biosourcés', 'Biosourcés',
                'Plastiques', 'Plastiques', 'Plastiques', 'Plastiques', 'Plastiques', 'Plastiques', 'Bioplastiques', 'Bioplastiques',
                'Composites', 'Composites', 'Biocomposites', 'Biocomposites', 'Composites', 'Composites', 'Composites', 'Biocomposites',
                'Isolants', 'Isolants', 'Isolants', 'Isolants', 'Isolants bio', 'Isolants bio', 'Isolants', 'Isolants',
                'Innovants', 'Innovants', 'Innovants', 'Innovants', 'Innovants', 'Textiles', 'Innovants', 'Innovants'
            ],

            # Empreinte carbone en kg CO2 eq/kg (données ICE Database + EPDs)
            'carbon_footprint_kg_co2_eq': [
                0.43, 2.29, 1.69, 11.46, 1.18, 3.77, 8.5, 2.84,
                0.154, 0.082, 0.039, 0.108, 0.242, 0.073, 0.087, 0.179,
                0.72, 1.86, 0.027, 0.159, 0.056, 0.075, 0.028, 0.263,
                1.24, 3.35, 1.05, 2.76, 1.37, 3.43, 1.85, 1.95,
                3.18, 26.7, 1.89, 1.45, 2.13, 2.65, 4.82, 1.25,
                1.16, 3.68, 4.24, 1.34, 0.143, 0.359, 0.087, 0.142,
                0.157, 0.048, 0.193, 0.187, 0.068, 1.83, 0.294, 0.089
            ],

            # Consommation énergétique en MJ/kg
            'energy_consumption_mj': [
                9.5, 29.7, 23.6, 154.4, 15.8, 68.9, 125.0, 38.5,
                0.78, 0.65, 0.52, 0.71, 4.2, 1.85, 0.28, 2.8,
                7.4, 11.3, 2.1, 5.8, 2.3, 3.1, 1.5, 4.8,
                28.5, 73.5, 26.8, 68.2, 35.4, 82.1, 28.7, 31.2,
                15.8, 198.5, 18.9, 16.2, 25.4, 28.9, 45.6, 18.5,
                12.4, 88.5, 101.2, 18.7, 4.8, 8.9, 3.2, 5.8,
                8.9, 2.1, 15.4, 12.8, 4.5, 45.2, 18.9, 6.7
            ],

            # Consommation d'eau en litres/kg
            'water_usage_liters': [
                4.1, 39.2, 9.8, 1534.6, 25.4, 158.7, 85.0, 48.9,
                1.2, 0.89, 0.65, 1.05, 8.5, 2.1, 0.35, 6.8,
                15.2, 89.5, 2850.4, 1.8, 12.5, 18.7, 125.4, 58.9,
                18.9, 48.5, 15.2, 38.9, 22.1, 54.8, 89.5, 158.7,
                35.8, 258.9, 45.2, 58.9, 68.2, 89.5, 125.4, 45.0,
                9.8, 158.7, 68.9, 18.5, 2.1, 35.8, 15.2, 25.4,
                258.9, 685.2, 125.4, 158.7, 125.4, 125.4, 289.5, 158.7
            ],

            # Score de recyclabilité (0-1)
            'recyclability_score': [
                0.95, 0.85, 0.98, 0.90, 0.95, 0.88, 0.85, 0.92,
                0.15, 0.25, 0.45, 0.65, 0.25, 0.85, 0.95, 0.35,
                0.80, 0.60, 0.90, 0.85, 0.95, 0.92, 0.88, 0.92,
                0.85, 0.48, 0.82, 0.52, 0.78, 0.48, 0.72, 0.68,
                0.25, 0.15, 0.88, 0.85, 0.65, 0.45, 0.55, 0.75,
                0.35, 0.15, 0.08, 0.25, 0.85, 0.88, 0.45, 0.55,
                0.88, 0.92, 0.78, 0.65, 0.85, 0.85, 0.88, 0.75
            ],

            # Durabilité en années
            'durability_years': [
                50, 50, 30, 30, 40, 40, 25, 35,
                100, 100, 80, 80, 100, 150, 200, 80,
                80, 60, 25, 40, 15, 15, 30, 25,
                25, 25, 20, 20, 15, 15, 8, 10,
                20, 30, 25, 25, 18, 25, 22, 20,
                50, 40, 30, 40, 40, 35, 80, 60,
                15, 10, 50, 80, 25, 10, 15, 50
            ],

            # Coût en €/kg
            'cost_eur_kg': [
                0.89, 0.78, 1.98, 1.89, 6.95, 6.35, 25.0, 3.45,
                0.12, 0.18, 0.28, 0.16, 0.38, 0.18, 0.95, 0.24,
                0.48, 0.68, 3.85, 4.58, 0.95, 1.35, 0.38, 0.75,
                1.28, 1.18, 1.25, 1.12, 1.45, 1.32, 5.85, 6.25,
                2.95, 48.5, 3.25, 3.65, 2.15, 2.85, 4.25, 3.5,
                0.98, 1.35, 9.85, 1.58, 4.58, 1.95, 2.45, 2.15,
                12.5, 18.5, 4.85, 8.95, 18.5, 4.85, 22.5, 3.85
            ],

            # Score de toxicité (0-1, 0=non toxique)
            'toxicity_score': [
                0.15, 0.25, 0.12, 0.22, 0.18, 0.28, 0.25, 0.22,
                0.08, 0.04, 0.01, 0.06, 0.12, 0.005, 0.005, 0.10,
                0.015, 0.065, 0.005, 0.005, 0.005, 0.005, 0.005, 0.015,
                0.35, 0.58, 0.32, 0.52, 0.32, 0.52, 0.08, 0.06,
                0.28, 0.42, 0.04, 0.05, 0.22, 0.32, 0.25, 0.08,
                0.12, 0.45, 0.68, 0.22, 0.005, 0.015, 0.04, 0.06,
                0.005, 0.005, 0.015, 0.04, 0.015, 0.28, 0.005, 0.015
            ]
        }

        self.materials_df = pd.DataFrame(materials_data)

        # Calculer le score d'impact environnemental global
        self.materials_df['environmental_impact_score'] = (
            self.materials_df['carbon_footprint_kg_co2_eq'] * 0.30 +
            (self.materials_df['energy_consumption_mj'] / 20) * 0.25 +
            (self.materials_df['water_usage_liters'] / 100) * 0.15 +
            (1 - self.materials_df['recyclability_score']) * 0.20 +
            self.materials_df['toxicity_score'] * 0.10
        )

        # Score de circularité
        self.materials_df['circularity_score'] = (
            self.materials_df['recyclability_score'] * 0.4 +
            (1 - self.materials_df['toxicity_score']) * 0.3 +
            (np.log(self.materials_df['durability_years'] + 1) / 6) * 0.3
        )

        return self.materials_df

    def save_to_database(self):
        """Sauvegarder dans une base SQLite"""
        os.makedirs(os.path.dirname(self.db_path), exist_ok=True)
        conn = sqlite3.connect(self.db_path)
        self.materials_df.to_sql(
            'materials', conn, if_exists='replace', index=False)
        conn.close()
        print(f"✅ Base de données sauvegardée : {self.db_path}")

    def load_from_database(self):
        """Charger depuis la base SQLite"""
        if os.path.exists(self.db_path):
            conn = sqlite3.connect(self.db_path)
            self.materials_df = pd.read_sql_query(
                "SELECT * FROM materials", conn)
            conn.close()
            print("✅ Base de données chargée")
            return True
        return False

    def train_model(self):
        """Entraîner le modèle de recommandation"""
        features = [
            'carbon_footprint_kg_co2_eq', 'energy_consumption_mj', 'water_usage_liters',
            'recyclability_score', 'durability_years', 'toxicity_score'
        ]

        X = self.materials_df[features]
        y = self.materials_df['environmental_impact_score']

        # Normalisation
        X_scaled = self.scaler.fit_transform(X)

        # Modèle Random Forest
        self.model = RandomForestRegressor(n_estimators=100, random_state=42)
        self.model.fit(X_scaled, y)

        # Matrice de similarité
        self.similarity_matrix = cosine_similarity(X_scaled)

        print(
            f"✅ Modèle entraîné - Score R²: {self.model.score(X_scaled, y):.3f}")

    def get_recommendations(self, material_name, n_recommendations=5, filters=None):
        """Obtenir des recommandations"""
        if material_name not in self.materials_df['material_name'].values:
            return pd.DataFrame(), None

        # Matériau d'entrée
        input_idx = self.materials_df[
            self.materials_df['material_name'] == material_name
        ].index[0]
        input_material = self.materials_df.loc[input_idx]

        # Filtrer les matériaux avec un meilleur impact
        candidates = self.materials_df[
            self.materials_df['environmental_impact_score'] < input_material['environmental_impact_score']
        ].copy()

        if candidates.empty:
            return pd.DataFrame(), input_material

        # Appliquer les filtres
        if filters:
            if filters.get('max_cost'):
                candidates = candidates[candidates['cost_eur_kg']
                                        <= filters['max_cost']]
            if filters.get('min_recyclability'):
                candidates = candidates[candidates['recyclability_score']
                                        >= filters['min_recyclability']]
            if filters.get('categories'):
                candidates = candidates[candidates['category'].isin(
                    filters['categories'])]

        if candidates.empty:
            return pd.DataFrame(), input_material

        # Calcul des scores de recommandation
        similarities = self.similarity_matrix[input_idx]
        candidates['similarity_score'] = [similarities[idx]
                                          for idx in candidates.index]

        # Score final
        candidates['recommendation_score'] = (
            candidates['similarity_score'] * 0.3 +
            candidates['circularity_score'] * 0.4 +
            (1 - candidates['environmental_impact_score'] /
             input_material['environmental_impact_score']) * 0.3
        )

        # Retourner les meilleures recommandations
        top_recommendations = candidates.nlargest(
            n_recommendations, 'recommendation_score')

        return top_recommendations, input_material
