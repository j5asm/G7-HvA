# G7 Landen.py - Clean Version
import streamlit as st
import requests
import pandas as pd
import plotly.graph_objects as go
import plotly.express as px
import time

# API Configuration
API_KEY = "WXpLhqoFwtWNQK/4yBAnLQ==Dr4y3QC5e0OOcSpn"
BASE_URL_POPULATION = "https://api.api-ninjas.com/v1/population"
BASE_URL_GDP = "https://api.api-ninjas.com/v1/gdp"

# G7 Countries
G7_COUNTRIES = {
    'Canada': 'Canada',
    'France': 'France', 
    'Germany': 'Germany',
    'Italy': 'Italy',
    'Japan': 'Japan',
    'United Kingdom': 'United Kingdom',
    'United States': 'United States'
}

def get_population_data(country_name):
    """Fetch population data for a country"""
    headers = {"X-Api-Key": API_KEY}
    
    try:
        # Population API
        url = f"{BASE_URL_POPULATION}?country={country_name}"
        response = requests.get(url, headers=headers, timeout=10)
        
        hist_df = pd.DataFrame()
        
        if response.status_code == 200:
            data = response.json()
            
            # Process historical data
            historical_data = data.get('historical_population', [])
            if historical_data:
                hist_df = pd.DataFrame(historical_data)
                if not hist_df.empty:
                    hist_df = hist_df.set_index('year')
                    hist_df.rename(columns={'population': 'historical_population'}, inplace=True)
            
            # Process population forecast data
            forecast_data = data.get('population_forecast', [])
            if forecast_data:
                forecast_df = pd.DataFrame(forecast_data)
                forecast_df = forecast_df.set_index('year')
                forecast_df.rename(columns={'population': 'population_forecast'}, inplace=True)
                hist_df = hist_df.combine_first(forecast_df)
            
            # Add other top-level fields
            excluded_fields = ['historical_population', 'population_forecast']
            for key, value in data.items():
                if key not in excluded_fields:
                    hist_df[key] = value
            
            # Clean and sort data
            hist_df = hist_df.sort_index()
            
            print(f"Success {country_name}: {len(hist_df)} data points retrieved")
            
        else:
            print(f"Error {country_name}: API Error {response.status_code}")
            return pd.DataFrame()
            
        # GDP API call
        try:
            url_gdp = f"{BASE_URL_GDP}?country={country_name}"
            response_gdp = requests.get(url_gdp, headers=headers, timeout=10)
            
            if response_gdp.status_code == 200:
                gdp_data = response_gdp.json()
                gdp_df = pd.DataFrame(gdp_data)
                if not gdp_df.empty and 'year' in gdp_df.columns:
                    gdp_df = gdp_df.set_index('year')
                    hist_df = hist_df.combine_first(gdp_df)
                    print(f"Success {country_name}: GDP data added")
        except Exception as e:
            print(f"Warning {country_name}: GDP data error - {e}")
                
    except Exception as e:
        print(f"Error {country_name}: {e}")
        return pd.DataFrame()
    
    return hist_df

def create_g7_comparison_chart(all_data, metric='historical_population', title_suffix='Historical Population'):
    """Create comparison chart for G7 countries"""
    fig = go.Figure()
    
    # Color palette for G7 countries
    colors = ['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728', '#9467bd', '#8c564b', '#e377c2']
    
    for i, (country, data) in enumerate(all_data.items()):
        if metric in data.columns and not data[metric].isna().all():
            # Clean metric name for display
            metric_display = metric.replace('_', ' ').title()
            
            fig.add_trace(go.Scatter(
                x=data.index,
                y=data[metric],
                mode='lines+markers',
                name=country,
                line=dict(width=2.5, color=colors[i % len(colors)]),
                marker=dict(size=4),
                hovertemplate=f'<b>{country}</b><br>' +
                             'Year: %{x}<br>' +
                             f'{metric_display}: %{{y:,.0f}}<br>' +
                             '<extra></extra>'
            ))
    
    fig.update_layout(
        title=f'G7 Countries - {title_suffix}',
        xaxis_title='Year',
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

def main():
    """Streamlit Application"""
    st.set_page_config(
        page_title="G7 Population & GDP Analyzer",
        layout="wide"
    )
    
    st.title("G7 Countries Analysis")
    st.markdown("**The Group of Seven:** Canada, France, Germany, Italy, Japan, United Kingdom, United States")
    st.markdown("---")
    
    # Sidebar
    st.sidebar.header("Controls")
    
    # Load data button
    if st.sidebar.button("Load G7 Data", type="primary"):
        with st.spinner("Loading G7 data..."):
            g7_data = {}
            progress = st.progress(0)
            status = st.empty()
            
            for i, (display_name, api_name) in enumerate(G7_COUNTRIES.items()):
                status.text(f"Loading {display_name}...")
                progress.progress((i + 1) / len(G7_COUNTRIES))
                
                data = get_population_data(api_name)
                if not data.empty:
                    g7_data[display_name] = data
                    
                time.sleep(0.5)  # Be nice to the API
            
            st.session_state.g7_data = g7_data
            progress.empty()
            status.empty()
        
        st.sidebar.success(f"Data loaded for {len(st.session_state.g7_data)} countries!")
    
    # Check if data exists
    if 'g7_data' not in st.session_state:
        st.info("Click 'Load G7 Data' in the sidebar to start")
        st.markdown("""
        ### About G7
        The Group of Seven (G7) is an informal bloc of industrialized democracies:
        - **Canada**
        - **France** 
        - **Germany**
        - **Italy**
        - **Japan**
        - **United Kingdom**
        - **United States**
        
        Click the button above to start analyzing population and GDP data.
        """)
        return
    
    g7_data = st.session_state.g7_data
    
    if not g7_data:
        st.error("No data available. Please try loading again.")
        return
    
    # Country selection
    st.sidebar.subheader("Country Selection")
    selected_countries = st.sidebar.multiselect(
        "Select Countries:",
        options=list(G7_COUNTRIES.keys()),
        default=list(G7_COUNTRIES.keys()),
        help="Choose which G7 countries to include in analysis"
    )
    
    # Get available metrics from all data
    all_metrics = set()
    for data in g7_data.values():
        numeric_cols = data.select_dtypes(include=['number']).columns
        all_metrics.update(numeric_cols)
    
    # Remove unwanted columns
    available_metrics = sorted([col for col in all_metrics if col not in ['rank']])
    
    if not available_metrics:
        st.error("No numeric columns found in the data.")
        return
    
    # Metric selection
    st.sidebar.subheader("Metric Selection")
    selected_metric = st.sidebar.selectbox(
        "Select Metric:",
        options=available_metrics,
        index=0 if 'historical_population' not in available_metrics else available_metrics.index('historical_population'),
        help="Choose the metric to analyze"
    )
    
    # Year range filter
    st.sidebar.subheader("Time Period")
    all_years = []
    for data in g7_data.values():
        all_years.extend(data.index.tolist())
    
    if all_years:
        min_year = int(min(all_years))
        max_year = int(max(all_years))
        
        year_range = st.sidebar.slider(
            "Year Range:",
            min_value=min_year,
            max_value=max(max_year, 2025),
            value=(min_year, max(max_year, 2025)),
            help="Select the time period to analyze"
        )
    else:
        year_range = (2000, 2025)
    
    # Filter data based on selections
    filtered_data = {}
    for country, data in g7_data.items():
        if country in selected_countries:
            # Filter by year range
            filtered_data[country] = data.loc[year_range[0]:year_range[1]]
    
    if not filtered_data:
        st.warning("No countries selected for analysis.")
        return
    
    # Main content - Tabs
    tab1, tab2, tab3, tab4 = st.tabs(["G7 Comparison", "Individual Charts", "Data Tables", "Rankings"])
    
    with tab1:
        st.subheader(f"G7 Comparison: {selected_metric.replace('_', ' ').title()}")
        
        try:
            fig = create_g7_comparison_chart(
                filtered_data, 
                selected_metric,
                selected_metric.replace('_', ' ').title()
            )
            st.plotly_chart(fig, use_container_width=True)
            
            # Summary statistics
            st.subheader("Summary Statistics")
            
            summary_data = []
            for country, data in filtered_data.items():
                if selected_metric in data.columns:
                    values = data[selected_metric].dropna()
                    if not values.empty:
                        summary_data.append({
                            'Country': country,
                            'Latest Value': f"{values.iloc[-1]:,.0f}" if len(values) > 0 else "N/A",
                            'Average': f"{values.mean():,.0f}" if len(values) > 0 else "N/A",
                            'Min': f"{values.min():,.0f}" if len(values) > 0 else "N/A",
                            'Max': f"{values.max():,.0f}" if len(values) > 0 else "N/A",
                            'Data Points': len(values)
                        })
            
            if summary_data:
                summary_df = pd.DataFrame(summary_data)
                st.dataframe(summary_df, use_container_width=True)
            else:
                st.info("No data available for the selected metric and countries.")
                
        except Exception as e:
            st.error(f"Error creating comparison chart: {str(e)}")
    
    with tab2:
        st.subheader("Individual Country Charts")
        
        # Display charts in a 2-column layout
        cols = st.columns(2)
        chart_count = 0
        
        for country, data in filtered_data.items():
            if selected_metric in data.columns and not data[selected_metric].isna().all():
                with cols[chart_count % 2]:
                    try:
                        fig = px.line(
                            data,
                            x=data.index,
                            y=selected_metric,
                            title=f'{country} - {selected_metric.replace("_", " ").title()}',
                            labels={'x': 'Year', selected_metric: selected_metric.replace("_", " ").title()}
                        )
                        fig.update_traces(mode='lines+markers')
                        fig.update_layout(height=400, template='plotly_white')
                        st.plotly_chart(fig, use_container_width=True)
                        chart_count += 1
                    except Exception as e:
                        st.error(f"Error creating chart for {country}: {str(e)}")
    
    with tab3:
        st.subheader("Raw Data Tables")
        
        for country, data in filtered_data.items():
            with st.expander(f"{country} Data"):
                if not data.empty:
                    st.dataframe(data, use_container_width=True)
                else:
                    st.info(f"No data available for {country}")
    
    with tab4:
        st.subheader("G7 Rankings")
        
        if selected_metric in ['historical_population', 'gdp', 'median_age', 'fertility_rate', 'population_forecast']:
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
                
                # Format values
                ranking_df['Value'] = ranking_df['Value'].apply(lambda x: f"{x:,.0f}")
                
                st.dataframe(ranking_df, use_container_width=True, hide_index=True)
                
                # Create ranking bar chart
                try:
                    ranking_df['Value_Numeric'] = ranking_df['Value'].str.replace(',', '').astype(float)
                    
                    fig_ranking = px.bar(
                        ranking_df, 
                        x='Country', 
                        y='Value_Numeric',
                        title=f'G7 Ranking by {selected_metric.replace("_", " ").title()}',
                        color='Value_Numeric',
                        color_continuous_scale='viridis',
                        labels={'Value_Numeric': selected_metric.replace("_", " ").title()}
                    )
                    fig_ranking.update_layout(height=500, template='plotly_white')
                    st.plotly_chart(fig_ranking, use_container_width=True)
                except Exception as e:
                    st.error(f"Error creating ranking chart: {str(e)}")
            else:
                st.info("No ranking data available for the selected metric.")
        else:
            st.info("Rankings are available for: historical_population, gdp, median_age, fertility_rate, population_forecast")
    
    # Footer
    st.markdown("---")
    st.markdown("**Data Source:** API Ninjas - Population & GDP APIs")
    st.markdown("**Data Status:** Real-time API fetching")

if __name__ == "__main__":
    main()
