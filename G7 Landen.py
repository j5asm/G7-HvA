# G7 Landen.py - SUPER FAST VERSION
import streamlit as st
import requests
import pandas as pd
import plotly.graph_objects as go

# Config
API_KEY = "WXpLhqoFwtWNQK/4yBAnLQ==Dr4y3QC5e0OOcSpn"
G7_COUNTRIES = ['Canada', 'France', 'Germany', 'Italy', 'Japan', 'United Kingdom', 'United States']

@st.cache_data(ttl=7200)
def get_all_data():
    """Get all G7 data at once - CACHED"""
    all_data = {}
    
    for country in G7_COUNTRIES:
        try:
            # Population data
            pop_response = requests.get(f"https://api.api-ninjas.com/v1/population?country={country}", 
                                      headers={"X-Api-Key": API_KEY}, timeout=3)
            
            if pop_response.status_code == 200:
                pop_data = pop_response.json()
                
                # Build dataframe quickly
                rows = []
                
                # Historical population
                for item in pop_data.get('historical_population', []):
                    rows.append({
                        'year': item['year'],
                        'population': item['population'],
                        'type': 'historical'
                    })
                
                # Population forecast
                for item in pop_data.get('population_forecast', []):
                    rows.append({
                        'year': item['year'],
                        'population': item['population'],
                        'type': 'forecast'
                    })
                
                if rows:
                    df = pd.DataFrame(rows)
                    df = df.pivot_table(index='year', columns='type', values='population', aggfunc='first')
                    df.columns.name = None
                    
                    # Add metadata
                    for key in ['median_age', 'fertility_rate']:
                        if key in pop_data:
                            df[key] = pop_data[key]
                    
                    # Get migrants data
                    try:
                        migrants_response = requests.get(f"https://api.api-ninjas.com/v1/migrants?country={country}",
                                                       headers={"X-Api-Key": API_KEY}, timeout=3)
                        if migrants_response.status_code == 200:
                            migrants_data = migrants_response.json()
                            if migrants_data:
                                df['migrants'] = migrants_data[0].get('migrants', 0)
                                df['net_migration'] = migrants_data[0].get('net_migration', 0)
                    except:
                        pass
                    
                    # Interpolate for smooth lines
                    df = df.interpolate(method='linear')
                    all_data[country] = df
                    
        except:
            continue
    
    return all_data

def create_chart(data, metric, countries):
    """Ultra fast chart creation"""
    fig = go.Figure()
    colors = ['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728', '#9467bd', '#8c564b', '#e377c2']
    
    for i, country in enumerate(countries):
        if country in data and metric in data[country].columns:
            fig.add_trace(go.Scatter(
                x=data[country].index,
                y=data[country][metric],
                mode='lines',
                name=country,
                line=dict(width=2, color=colors[i % len(colors)]),
                connectgaps=True
            ))
    
    fig.update_layout(
        title=metric.replace('_', ' ').title(),
        height=400,
        showlegend=True,
        template='plotly_white'
    )
    return fig

def main():
    st.set_page_config(page_title="G7 Fast", layout="wide")
    st.title("G7 Analysis - Population & Migration")
    
    # Load data automatically
    if 'data_loaded' not in st.session_state:
        with st.spinner("Loading..."):
            st.session_state.g7_data = get_all_data()
            st.session_state.data_loaded = True
    
    data = st.session_state.g7_data
    
    if not data:
        st.error("No data")
        return
    
    # Quick controls
    col1, col2 = st.columns([1, 3])
    
    with col1:
        countries = st.multiselect("Countries:", G7_COUNTRIES, default=G7_COUNTRIES[:4])
        
        # Get available metrics
        all_metrics = set()
        for df in data.values():
            all_metrics.update(df.columns)
        
        metrics = ['historical', 'forecast', 'migrants', 'net_migration', 'median_age', 'fertility_rate']
        available = [m for m in metrics if m in all_metrics]
        
        metric = st.selectbox("Metric:", available)
    
    with col2:
        if countries and metric:
            fig = create_chart(data, metric, countries)
            st.plotly_chart(fig, use_container_width=True)
    
    # Quick stats
    if countries and metric:
        st.subheader("Latest Values")
        stats = []
        for country in countries:
            if country in data and metric in data[country].columns:
                latest = data[country][metric].dropna()
                if not latest.empty:
                    stats.append({
                        'Country': country,
                        'Value': f"{latest.iloc[-1]:,.0f}",
                        'Year': latest.index[-1]
                    })
        
        if stats:
            st.dataframe(pd.DataFrame(stats), hide_index=True)

if __name__ == "__main__":
    main()
