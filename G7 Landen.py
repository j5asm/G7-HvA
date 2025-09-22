import requests
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st
from typing import Dict, List, Optional, Tuple
import logging
from concurrent.futures import ThreadPoolExecutor, as_completed
import time

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Constants
API_KEY = "WXpLhqoFwtWNQK/4yBAnLQ==Dr4y3QC5e0OOcSpn"
BASE_URL_POPULATION = "https://api.api-ninjas.com/v1/population"
BASE_URL_GDP = "https://api.api-ninjas.com/v1/gdp"

# G7 Countries - The Group of Seven advanced economies
G7_COUNTRIES = {
    'Canada': 'Canada',
    'France': 'France', 
    'Germany': 'Germany',
    'Italy': 'Italy',
    'Japan': 'Japan',
    'United Kingdom': 'United Kingdom',
    'United States': 'United States'  # API might use 'United States' instead of 'United States of America'
}

# Configure pandas display options
pd.options.display.max_rows = 999
pd.set_option('display.width', 1000)
pd.set_option('display.max_columns', None)


def get_population_data(country_name: str) -> pd.DataFrame:
    """
    Original function - enhanced with better error handling
    """
    headers = {"X-Api-Key": API_KEY}
    
    try:
        # Population API call
        url = f"{BASE_URL_POPULATION}?country={country_name}"
        response = requests.get(url, headers=headers, timeout=10)
        
        hist_df = pd.DataFrame()
        
        if response.status_code == 200:
            data = response.json()
            
            # Process historical population data
            hist_df = pd.DataFrame(data.get('historical_population', []))
            if not hist_df.empty:
                hist_df = hist_df.set_index('year')
                hist_df.rename(columns={'population': 'historical_population'}, inplace=True)
        
            # Process forecast data
            forecast_df = pd.DataFrame(data.get('population_forecast', []))
            if not forecast_df.empty:
                forecast_df = forecast_df.set_index('year')
                forecast_df.rename(columns={'population': 'population_forecast'}, inplace=True)
                # Combine with historical data
                hist_df = hist_df.combine_first(forecast_df)
      
            # Add top-level fields
            top_level_fields = {k: v for k, v in data.items() 
                              if k not in ['historical_population', 'population_forecast']}
            for key, value in top_level_fields.items():
                hist_df[key] = value
            
            # Clean column names (remove dots)
            for col in list(hist_df.columns):
                if '.' in col: 
                    base_col = col.split('.')[0]
                    hist_df[base_col] = hist_df[base_col].combine_first(hist_df[col])
                    hist_df = hist_df.drop(columns=[col])
            
            # Clean and sort data
            key_cols = ['historical_population', 'median_age', 'fertility_rate', 'rank']
            available_key_cols = [col for col in key_cols if col in hist_df.columns]
            if available_key_cols:
                hist_df = hist_df.dropna(subset=available_key_cols, how='all')
            hist_df = hist_df.sort_index()
            
            print(f"✅ {country_name}: Data ophalen gelukt ({len(hist_df)} jaren)")
            
        else:
            print(f"❌ {country_name}: Error {response.status_code}, {response.text}")
            return pd.DataFrame()
        
        # GDP API call
        url_gdp = f"{BASE_URL_GDP}?country={country_name}"
        response_gdp = requests.get(url_gdp, headers=headers, timeout=10)
        
        if response_gdp.status_code == 200:
            gdp_data = response_gdp.json()
            gdp_df = pd.DataFrame(gdp_data)
            
            # Add GDP data to main dataframe if possible
            if not gdp_df.empty and 'year' in gdp_df.columns:
                gdp_df = gdp_df.set_index('year')
                hist_df = hist_df.combine_first(gdp_df)
                print(f"✅ {country_name}: GDP data toegevoegd")
            else:
                print(f"⚠️ {country_name}: GDP data beschikbaar maar geen jaartal info")
        else:
            print(f"⚠️ {country_name}: GDP Error {response_gdp.status_code}")
            
    except Exception as e:
        print(f"❌ {country_name}: Exception occurred: {e}")
        return pd.DataFrame()
    
    return hist_df


def fetch_all_g7_data() -> Dict[str, pd.DataFrame]:
    """
    Fetch data for all G7 countries efficiently
    """
    all_data = {}
    
    print("🌍 Bezig met ophalen G7 landen data...")
    print("=" * 50)
    
    # Sequential fetching to respect API limits
    for country_display, country_api in G7_COUNTRIES.items():
        print(f"Ophalen data voor {country_display}...")
        data = get_population_data(country_api)
        
        if not data.empty:
            all_data[country_display] = data
            print(f"📊 {country_display}: {len(data)} datapunten opgehaald")
        else:
            print(f"⚠️ {country_display}: Geen data beschikbaar")
        
        # Small delay to be respectful to the API
        time.sleep(0.5)
    
    print("=" * 50)
    print(f"✅ Klaar! Data voor {len(all_data)} van {len(G7_COUNTRIES)} G7 landen opgehaald")
    
    return all_data


def create_g7_comparison_chart(all_data: Dict[str, pd.DataFrame], 
                              metric: str = 'historical_population',
                              title_suffix: str = 'Historical Population') -> go.Figure:
    """
    Create comparison chart for G7 countries
    """
    fig = go.Figure()
    
    # Color palette for G7 countries
    colors = ['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728', '#9467bd', '#8c564b', '#e377c2']
    
    for i, (country, data) in enumerate(all_data.items()):
        if metric in data.columns and not data[metric].isna().all():
            fig.add_trace(go.Scatter(
                x=data.index,
                y=data[metric],
                mode='lines+markers',
                name=country,
                line=dict(width=2.5, color=colors[i % len(colors)]),
                marker=dict(size=4),
                hovertemplate=f'<b>{country}</b><br>' +
                             'Jaar: %{x}<br>' +
                             f'{metric.replace("_", " ").title()}: %{y:,.0f}<br>' +
                             '<extra></extra>'
            ))
    
    fig.update_layout(
        title=f'G7 Countries - {title_suffix}',
        xaxis_title='Jaar',
        yaxis_title=metric.replace('_', ' ').title(),
        hovermode='x unified',
        showlegend=True,
        height=600,
        template='plotly_white',
        font=dict(size=12),
        legend=dict(
            orientation="v",
            yanchor="top",
            y=1,
            xanchor="left",
            x=1.02
        )
    )
    
    return fig


def create_individual_charts(all_data: Dict[str, pd.DataFrame]) -> List[go.Figure]:
    """
    Create individual charts for each G7 country (like your original code)
    """
    figures = []
    
    for country, data in all_data.items():
        if 'historical_population' in data.columns:
            fig = px.line(
                data,
                x=data.index,
                y='historical_population',
                title=f'Historical Population of {country}',
                labels={'x': 'Year', 'historical_population': 'Population'}
            )
            fig.update_traces(mode='lines+markers')
            fig.update_layout(height=400, template='plotly_white')
            figures.append((country, fig))
    
    return figures


def main():
    """
    Streamlit app for G7 analysis
    """
    st.set_page_config(
        page_title="G7 Population & GDP Analyzer",
        page_icon="🏛️",
        layout="wide"
    )
    
    st.title("🏛️ G7 Countries - Population & GDP Analysis")
    st.markdown("""
    **The G7 (Group of Seven)** consists of the world's most advanced economies:
    Canada, France, Germany, Italy, Japan, the United Kingdom, and the United States.
    """)
    st.markdown("---")
    
    # Sidebar controls
    st.sidebar.header("🎛️ Analysis Options")
    
    # Load data button
    if st.sidebar.button("🔄 Load G7 Data", type="primary"):
        with st.spinner("Bezig met ophalen G7 data..."):
            st.session_state.g7_data = fetch_all_g7_data()
        st.sidebar.success(f"Data geladen voor {len(st.session_state.g7_data)} landen!")
    
    # Check if data is loaded
    if 'g7_data' not in st.session_state:
        st.info("👈 Klik op 'Load G7 Data' in de sidebar om te beginnen")
        st.stop()
    
    all_data = st.session_state.g7_data
    
    if not all_data:
        st.error("Geen data beschikbaar. Probeer opnieuw.")
        st.stop()
    
    # Get available metrics
    all_columns = set()
    for data in all_data.values():
        all_columns.update(data.select_dtypes(include=[np.number]).columns)
    
    available_metrics = sorted([col for col in all_columns if col not in ['rank']])
    
    # Sidebar filters
    st.sidebar.subheader("📊 Visualization Options")
    
    # Country selection
    selected_countries = st.sidebar.multiselect(
        "🌍 Select G7 Countries:",
        options=list(G7_COUNTRIES.keys()),
        default=list(G7_COUNTRIES.keys()),
        help="Choose which G7 countries to include in analysis"
    )
    
    # Metric selection
    selected_metric = st.sidebar.selectbox(
        "📈 Select Metric:",
        options=available_metrics,
        index=0 if 'historical_population' in available_metrics else 0,
        help="Choose the metric to analyze"
    )
    
    # Year range filter
    all_years = []
    for data in all_data.values():
        all_years.extend(data.index.tolist())
    
    if all_years:
        min_year = int(min(all_years))
        max_year = int(max(all_years))
        
        year_range = st.sidebar.slider(
            "📅 Year Range:",
            min_value=min_year,
            max_value=max_year,
            value=(min_year, max_year),
            help="Select the time period to analyze"
        )
    
    # Filter data based on selections
    filtered_data = {country: data for country, data in all_data.items() 
                    if country in selected_countries}
    
    # Apply year filter
    if all_years:
        for country in filtered_data:
            filtered_data[country] = filtered_data[country].loc[year_range[0]:year_range[1]]
    
    # Main content
    tabs = st.tabs(["📊 G7 Comparison", "📈 Individual Charts", "📋 Data Tables", "📉 Rankings"])
    
    with tabs[0]:
        st.subheader(f"G7 Comparison - {selected_metric.replace('_', ' ').title()}")
        
        if filtered_data:
            fig = create_g7_comparison_chart(
                filtered_data, 
                selected_metric,
                selected_metric.replace('_', ' ').title()
            )
            st.plotly_chart(fig, use_container_width=True)
            
            # Summary statistics
            st.subheader("📊 Summary Statistics")
            
            summary_data = []
            for country, data in filtered_data.items():
                if selected_metric in data.columns:
                    values = data[selected_metric].dropna()
                    if not values.empty:
                        summary_data.append({
                            'Country': country,
                            'Latest Value': values.iloc[-1] if len(values) > 0 else np.nan,
                            'Average': values.mean(),
                            'Min': values.min(),
                            'Max': values.max(),
                            'Data Points': len(values)
                        })
            
            if summary_data:
                summary_df = pd.DataFrame(summary_data)
                st.dataframe(summary_df, use_container_width=True)
        else:
            st.warning("Geen landen geselecteerd voor vergelijking.")
    
    with tabs[1]:
        st.subheader("Individual Country Charts")
        
        individual_charts = create_individual_charts(filtered_data)
        
        # Display charts in a 2-column layout
        cols = st.columns(2)
        for i, (country, fig) in enumerate(individual_charts):
            with cols[i % 2]:
                st.plotly_chart(fig, use_container_width=True)
    
    with tabs[2]:
        st.subheader("Raw Data Tables")
        
        for country, data in filtered_data.items():
            with st.expander(f"📊 {country} Data"):
                st.dataframe(data, use_container_width=True)
    
    with tabs[3]:
        st.subheader("G7 Rankings")
        
        if selected_metric in ['historical_population', 'gdp', 'median_age', 'fertility_rate']:
            ranking_data = []
            
            for country, data in filtered_data.items():
                if selected_metric in data.columns:
                    latest_value = data[selected_metric].dropna()
                    if not latest_value.empty:
                        ranking_data.append({
                            'Country': country,
                            'Value': latest_value.iloc[-1],
                            'Year': latest_value.index[-1]
                        })
            
            if ranking_data:
                ranking_df = pd.DataFrame(ranking_data)
                ranking_df = ranking_df.sort_values('Value', ascending=False)
                ranking_df['Rank'] = range(1, len(ranking_df) + 1)
                
                # Reorder columns
                ranking_df = ranking_df[['Rank', 'Country', 'Value', 'Year']]
                
                st.dataframe(ranking_df, use_container_width=True, hide_index=True)
                
                # Create ranking bar chart
                fig_ranking = px.bar(
                    ranking_df, 
                    x='Country', 
                    y='Value',
                    title=f'G7 Ranking by {selected_metric.replace("_", " ").title()}',
                    color='Value',
                    color_continuous_scale='viridis'
                )
                fig_ranking.update_layout(height=500, template='plotly_white')
                st.plotly_chart(fig_ranking, use_container_width=True)
    
    # Footer
    st.markdown("---")
    st.markdown("**🔍 Data Source:** API Ninjas - Population & GDP APIs")
    st.markdown("**📅 Last Updated:** Real-time data fetching")


# Console version (your original approach)
def run_console_version():
    """
    Run the original console version with G7 focus
    """
    print("🏛️ G7 POPULATION ANALYSIS")
    print("=" * 50)
    
    # Your original approach - individual charts
    print("\n📈 Creating individual charts for each G7 country...")
    for country in G7_COUNTRIES.values():
        hist_df = get_population_data(country)
        if not hist_df.empty and 'historical_population' in hist_df.columns:
            fig = px.line(
                hist_df,
                x=hist_df.index,
                y='historical_population',
                title=f'Historical Population of {country}',
                labels={'x': 'Year', 'historical_population': 'Population'}
            )
            fig.update_traces(mode='lines+markers')
            # fig.show()  # Uncomment this if running in Jupyter
            print(f"✅ Chart created for {country}")
        else:
            print(f"❌ No data for {country}")
    
    print("\n📊 Creating combined G7 comparison chart...")
    # Your second approach - combined chart
    fig = go.Figure()
    
    for country in list(G7_COUNTRIES.values())[:2]:  # Start with Canada and France like your example
        hist_df = get_population_data(country)
        if not hist_df.empty and 'historical_population' in hist_df.columns:
            fig.add_trace(go.Scatter(
                x=hist_df.index, 
                y=hist_df['historical_population'], 
                mode='lines+markers',
                name=country
            ))
    
    fig.update_layout(
        title='G7 Population Comparison',
        xaxis_title='Year',
        yaxis_title='Population'
    )
    
    # fig.show()  # Uncomment this if running in Jupyter
    print("✅ Combined chart created")


if __name__ == "__main__":
    # Choose which version to run
    import sys
    
    if len(sys.argv) > 1 and sys.argv[1] == "console":
        run_console_version()
    else:
        # Run Streamlit version
        main()