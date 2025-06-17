import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
import plotly.express as px
from recommender import EcoMaterialRecommender


def create_streamlit_interface():
    """Interface Streamlit"""

    st.set_page_config(
        page_title="EcoMaterial Recommender",
        page_icon="🌱",
        layout="wide"
    )

    st.title("🌱 Système de Recommandation de Matériaux Éco-responsables")
    st.markdown("---")

    # Initialiser le système
    @st.cache_resource
    def initialize_system():
        recommender = EcoMaterialRecommender()

        # Charger ou créer la base de données
        if not recommender.load_from_database():
            with st.spinner("Création de la base de données..."):
                recommender.create_materials_database()
                recommender.save_to_database()

        # Entraîner le modèle
        with st.spinner("Entraînement du modèle..."):
            recommender.train_model()

        return recommender

    recommender = initialize_system()

    # Interface utilisateur
    col1, col2 = st.columns([1, 2])

    with col1:
        st.subheader("🔧 Configuration")

        # Sélection du matériau
        material_list = sorted(
            recommender.materials_df['material_name'].unique())
        selected_material = st.selectbox(
            "Matériau à remplacer:",
            material_list
        )

        # Filtres
        st.subheader("Filtres")
        max_cost = st.slider("Coût maximum (€/kg):", 0.0, 50.0, 25.0)
        min_recyclability = st.slider("Recyclabilité minimale:", 0.0, 1.0, 0.5)

        categories = st.multiselect(
            "Catégories:",
            options=sorted(recommender.materials_df['category'].unique())
        )

        n_recs = st.slider("Nombre de recommandations:", 1, 10, 5)

        filters = {
            'max_cost': max_cost,
            'min_recyclability': min_recyclability,
            'categories': categories if categories else None
        }

    with col2:
        st.subheader("📊 Résultats")

        # Obtenir les recommandations
        recommendations, input_material = recommender.get_recommendations(
            selected_material, n_recs, filters
        )

        if not recommendations.empty:
            # Afficher les recommandations
            st.write("**Alternatives recommandées:**")

            display_cols = [
                'material_name', 'category', 'environmental_impact_score',
                'carbon_footprint_kg_co2_eq', 'recyclability_score', 'cost_eur_kg'
            ]

            display_df = recommendations[display_cols].copy()
            display_df.columns = [
                'Matériau', 'Catégorie', 'Impact Env.',
                'Carbone (kg CO₂)', 'Recyclabilité', 'Coût (€/kg)'
            ]

            st.dataframe(display_df.round(3), use_container_width=True)

            # Graphique de comparaison
            fig = go.Figure()

            materials = [selected_material] + \
                recommendations['material_name'].tolist()
            impacts = [input_material['environmental_impact_score']] + \
                recommendations['environmental_impact_score'].tolist()
            colors = ['red'] + ['green'] * len(recommendations)

            fig.add_trace(go.Bar(
                x=materials,
                y=impacts,
                marker_color=colors,
                text=[f"{val:.3f}" for val in impacts],
                textposition='auto'
            ))

            fig.update_layout(
                title="Comparaison Impact Environnemental",
                xaxis_title="Matériaux",
                yaxis_title="Score d'Impact",
                xaxis_tickangle=-45
            )

            st.plotly_chart(fig, use_container_width=True)

            # Métriques d'amélioration
            best_alt = recommendations.iloc[0]

            carbon_reduction = (
                (input_material['carbon_footprint_kg_co2_eq'] -
                 best_alt['carbon_footprint_kg_co2_eq']) /
                input_material['carbon_footprint_kg_co2_eq'] * 100
            )

            impact_reduction = (
                (input_material['environmental_impact_score'] -
                 best_alt['environmental_impact_score']) /
                input_material['environmental_impact_score'] * 100
            )

            col3, col4, col5 = st.columns(3)
            with col3:
                st.metric("Réduction Carbone", f"{carbon_reduction:.1f}%")
            with col4:
                st.metric("Amélioration Impact", f"{impact_reduction:.1f}%")
            with col5:
                st.metric("Meilleure Alternative", best_alt['material_name'])

        else:
            st.warning("Aucune alternative trouvée avec ces critères.")

    # Informations sur les données
    st.markdown("---")
    st.subheader("ℹ️ Sources de données")

    col6, col7 = st.columns(2)
    with col6:
        st.info("""
        **Critères d'évaluation:**
        - Empreinte carbone (30%)
        - Consommation d'énergie (25%)
        - Usage d'eau (15%)
        - Recyclabilité (20%)
        - Toxicité (10%)
        """)

    with col7:
        st.success("""
        **Sources:**
        - ICE Database v3.0
        - EPD International
        - Material ConneXion
        - Données fabricants
        """)


if __name__ == "__main__":
    create_streamlit_interface()
