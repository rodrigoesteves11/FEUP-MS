"""
interactive_dashboard.py

Cria um dashboard HTML interativo onde podes navegar pelos resultados,
filtrar por política, ver seeds individuais, zoom, hover, etc.

Requer: plotly
Instalar: pip install plotly
"""
import pandas as pd
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import plotly.express as px


def load_results(csv_path="../results_new_model/kpi_results.csv"):
    """Load the KPI results from run.py"""
    df = pd.read_csv(csv_path)
    print(f"Loaded {len(df)} simulation results")
    return df


def create_interactive_dashboard(df, output_file="dashboard/interactive_dashboard.html"):
    """Create interactive HTML dashboard with Plotly"""
    print("\nCreating interactive dashboard...")
    

    fig = make_subplots(
        rows=4, cols=3,
        subplot_titles=(
            'Volatility by Policy', 'Mispricing by Policy', 'Gini by Policy',
            'Volume by Policy', 'Turnover by Policy', 'Max Drawdown by Policy',
            'Crashes vs Volatility', 'Gini vs Mispricing', 'Volume vs Volatility',
            'Policy Comparison - Key Metrics', 'Seed-by-Seed Volatility', 'Seed-by-Seed Gini'
        ),
        specs=[
            [{'type': 'box'}, {'type': 'box'}, {'type': 'box'}],
            [{'type': 'box'}, {'type': 'box'}, {'type': 'box'}],
            [{'type': 'scatter'}, {'type': 'scatter'}, {'type': 'scatter'}],
            [{'type': 'bar'}, {'type': 'scatter'}, {'type': 'scatter'}]
        ],
        vertical_spacing=0.08,
        horizontal_spacing=0.1
    )
    
    fig.update_layout(legend=dict(
        orientation="h",
        yanchor="bottom",
        y=1.02,
        xanchor="right",
        x=1,
        itemclick="toggle",
        itemdoubleclick="toggleothers"
    ))
    
    policies = ['none', 'moderate', 'excessive']
    colors = {'none': '#2ecc71', 'moderate': '#f39c12', 'excessive': '#e74c3c'}
    

    for policy in policies:
        df_policy = df[df['policy'] == policy]
        fig.add_trace(
            go.Box(y=df_policy['vol_mean'], name=policy.upper(), 
                   marker_color=colors[policy], boxmean='sd',
                   legendgroup=policy),
            row=1, col=1
        )
        fig.add_trace(
            go.Box(y=df_policy['mean_abs_mispricing'], name=policy.upper(),
                   marker_color=colors[policy], boxmean='sd', showlegend=False,
                   legendgroup=policy),
            row=1, col=2
        )
        fig.add_trace(
            go.Box(y=df_policy['gini_mean'], name=policy.upper(),
                   marker_color=colors[policy], boxmean='sd', showlegend=False,
                   legendgroup=policy),
            row=1, col=3
        )
    

    for policy in policies:
        df_policy = df[df['policy'] == policy]
        fig.add_trace(
            go.Box(y=df_policy['volume_mean'], name=policy.upper(),
                   marker_color=colors[policy], boxmean='sd', showlegend=False,
                   legendgroup=policy),
            row=2, col=1
        )
        fig.add_trace(
            go.Box(y=df_policy['turnover_mean'], name=policy.upper(),
                   marker_color=colors[policy], boxmean='sd', showlegend=False,
                   legendgroup=policy),
            row=2, col=2
        )
        fig.add_trace(
            go.Box(y=df_policy['max_drawdown'], name=policy.upper(),
                   marker_color=colors[policy], boxmean='sd', showlegend=False,
                   legendgroup=policy),
            row=2, col=3
        )
    

    for policy in policies:
        df_policy = df[df['policy'] == policy]
        

        fig.add_trace(
            go.Scatter(
                x=df_policy['vol_mean'], 
                y=df_policy['n_crashes_ret'],
                mode='markers',
                name=policy.upper(),
                marker=dict(size=8, color=colors[policy], opacity=0.7),
                text=[f"Seed: {s}" for s in df_policy['seed']],
                hovertemplate='<b>%{text}</b><br>Volatility: %{x:.4f}<br>Crashes: %{y}<extra></extra>',
                showlegend=False,
                legendgroup=policy
            ),
            row=3, col=1
        )
        

        fig.add_trace(
            go.Scatter(
                x=df_policy['mean_abs_mispricing'],
                y=df_policy['gini_mean'],
                mode='markers',
                name=policy.upper(),
                marker=dict(size=8, color=colors[policy], opacity=0.7),
                text=[f"Seed: {s}" for s in df_policy['seed']],
                hovertemplate='<b>%{text}</b><br>Mispricing: %{x:.2f}<br>Gini: %{y:.3f}<extra></extra>',
                showlegend=False,
                legendgroup=policy
            ),
            row=3, col=2
        )
        

        fig.add_trace(
            go.Scatter(
                x=df_policy['vol_mean'],
                y=df_policy['volume_mean'],
                mode='markers',
                name=policy.upper(),
                marker=dict(size=8, color=colors[policy], opacity=0.7),
                text=[f"Seed: {s}" for s in df_policy['seed']],
                hovertemplate='<b>%{text}</b><br>Volatility: %{x:.4f}<br>Volume: %{y:.0f}<extra></extra>',
                showlegend=False,
                legendgroup=policy
            ),
            row=3, col=3
        )
    

    metrics = ['vol_mean', 'mean_abs_mispricing', 'gini_mean', 'volume_mean']
    metric_names = ['Volatility', 'Mispricing', 'Gini', 'Volume']
    
    for metric, name in zip(metrics, metric_names):
        means = [df[df['policy'] == p][metric].mean() for p in policies]
        fig.add_trace(
            go.Bar(
                x=policies,
                y=means,
                name=name,
                marker_color=[colors[p] for p in policies],
                text=[f'{v:.4f}' if v < 100 else f'{v:.0f}' for v in means],
                textposition='outside',
                hovertemplate='<b>%{x}</b><br>' + name + ': %{y:.4f}<extra></extra>'
            ),
            row=4, col=1
        )
    

    for policy in policies:
        df_policy = df[df['policy'] == policy].sort_values('seed')
        fig.add_trace(
            go.Scatter(
                x=df_policy['seed'],
                y=df_policy['vol_mean'],
                mode='lines+markers',
                name=policy.upper(),
                line=dict(color=colors[policy]),
                marker=dict(size=6),
                hovertemplate='<b>Seed %{x}</b><br>Volatility: %{y:.4f}<extra></extra>',
                showlegend=False,
                legendgroup=policy
            ),
            row=4, col=2
        )
        
        fig.add_trace(
            go.Scatter(
                x=df_policy['seed'],
                y=df_policy['gini_mean'],
                mode='lines+markers',
                name=policy.upper(),
                line=dict(color=colors[policy]),
                marker=dict(size=6),
                hovertemplate='<b>Seed %{x}</b><br>Gini: %{y:.3f}<extra></extra>',
                showlegend=False,
                legendgroup=policy
            ),
            row=4, col=3
        )
    

    fig.update_layout(
        title_text="<b>Interactive Dashboard - Market ABM Results (90 Simulations)</b><br>" +
                   "<i>INSTRUÇÕES: Clica UMA VEZ na legenda para esconder/mostrar | Clica DUAS VEZES para isolar | Arrasta para zoom</i>",
        title_font_size=18,
        height=1600,
        showlegend=True,
        hovermode='closest',
        template='plotly_white',
        legend=dict(
            title=dict(text="<b>POLÍTICAS</b><br>(Clica para filtrar)", font=dict(size=12)),
            orientation="v",
            yanchor="top",
            y=0.98,
            xanchor="left",
            x=1.01,
            bgcolor="rgba(255,255,255,0.8)",
            bordercolor="Black",
            borderwidth=2
        )
    )
    

    fig.update_xaxes(title_text="Policy", row=1, col=1)
    fig.update_xaxes(title_text="Policy", row=1, col=2)
    fig.update_xaxes(title_text="Policy", row=1, col=3)
    fig.update_xaxes(title_text="Policy", row=2, col=1)
    fig.update_xaxes(title_text="Policy", row=2, col=2)
    fig.update_xaxes(title_text="Policy", row=2, col=3)
    fig.update_xaxes(title_text="Volatility", row=3, col=1)
    fig.update_xaxes(title_text="Mispricing", row=3, col=2)
    fig.update_xaxes(title_text="Volatility", row=3, col=3)
    fig.update_xaxes(title_text="Seed", row=4, col=2)
    fig.update_xaxes(title_text="Seed", row=4, col=3)
    
    fig.update_yaxes(title_text="Volatility", row=1, col=1)
    fig.update_yaxes(title_text="Mispricing", row=1, col=2)
    fig.update_yaxes(title_text="Gini", row=1, col=3)
    fig.update_yaxes(title_text="Volume", row=2, col=1)
    fig.update_yaxes(title_text="Turnover", row=2, col=2)
    fig.update_yaxes(title_text="Max Drawdown", row=2, col=3)
    fig.update_yaxes(title_text="# Crashes", row=3, col=1)
    fig.update_yaxes(title_text="Gini", row=3, col=2)
    fig.update_yaxes(title_text="Volume", row=3, col=3)
    fig.update_yaxes(title_text="Volatility", row=4, col=2)
    fig.update_yaxes(title_text="Gini", row=4, col=3)
    

    config = {
        'displayModeBar': True,
        'displaylogo': False,
        'modeBarButtonsToAdd': ['toggleSpikelines'],
        'toImageButtonOptions': {
            'format': 'png',
            'filename': 'dashboard_export',
            'height': 1600,
            'width': 1400,
            'scale': 2
        }
    }
    fig.write_html(output_file, config=config)
    print(f"\n✅ Interactive dashboard saved: {output_file}")
    print(f"\n🌐 Como usar no browser:")
    print(f"   ✓ Clica UMA VEZ na legenda → esconde/mostra essa política")
    print(f"   ✓ Clica DUAS VEZES na legenda → isola só essa política")
    print(f"   ✓ Hover sobre pontos → vê detalhes (seed, valores)")
    print(f"   ✓ Arrasta → zoom numa área")
    print(f"   ✓ Double-click no gráfico → reset zoom")
    print(f"   ✓ Botão 📷 no canto → exporta imagem PNG")
    

def create_individual_seed_viewer(df, output_file="dashboard/seed_explorer.html"):
    """Create an interactive viewer to explore individual seeds"""
    print("\nCreating seed explorer...")
    
    fig = make_subplots(
        rows=2, cols=2,
        subplot_titles=(
            'Select Seeds by Policy and Metric',
            'KPI Comparison for Selected Seeds',
            'All Seeds - Volatility vs Mispricing',
            'All Seeds - Gini vs Volume'
        ),
        specs=[
            [{'type': 'scatter'}, {'type': 'bar'}],
            [{'type': 'scatter'}, {'type': 'scatter'}]
        ]
    )
    
    policies = ['none', 'moderate', 'excessive']
    colors = {'none': '#2ecc71', 'moderate': '#f39c12', 'excessive': '#e74c3c'}
    
    for policy in policies:
        df_policy = df[df['policy'] == policy]
        fig.add_trace(
            go.Scatter(
                x=df_policy['seed'],
                y=df_policy['vol_mean'],
                mode='markers',
                name=policy.upper(),
                marker=dict(size=10, color=colors[policy]),
                text=[f"Policy: {policy}<br>Seed: {s}<br>Vol: {v:.4f}<br>Misp: {m:.2f}<br>Gini: {g:.3f}"
                      for s, v, m, g in zip(df_policy['seed'], df_policy['vol_mean'], 
                                           df_policy['mean_abs_mispricing'], df_policy['gini_mean'])],
                hovertemplate='%{text}<extra></extra>',
                legendgroup=policy
            ),
            row=1, col=1
        )
    
    for policy in policies:
        df_policy = df[df['policy'] == policy]
        fig.add_trace(
            go.Scatter(
                x=df_policy['vol_mean'],
                y=df_policy['mean_abs_mispricing'],
                mode='markers',
                name=policy.upper(),
                marker=dict(size=8, color=colors[policy], opacity=0.7),
                text=[f"Seed: {s}" for s in df_policy['seed']],
                hovertemplate='<b>%{text}</b><br>Volatility: %{x:.4f}<br>Mispricing: %{y:.2f}<extra></extra>',
                showlegend=False,
                legendgroup=policy
            ),
            row=2, col=1
        )
    
    for policy in policies:
        df_policy = df[df['policy'] == policy]
        fig.add_trace(
            go.Scatter(
                x=df_policy['gini_mean'],
                y=df_policy['volume_mean'],
                mode='markers',
                name=policy.upper(),
                marker=dict(size=8, color=colors[policy], opacity=0.7),
                text=[f"Seed: {s}" for s in df_policy['seed']],
                hovertemplate='<b>%{text}</b><br>Gini: %{x:.3f}<br>Volume: %{y:.0f}<extra></extra>',
                showlegend=False,
                legendgroup=policy
            ),
            row=2, col=2
        )
    
    fig.update_layout(
        title_text="<b>Seed Explorer - Interactive Seed Analysis</b><br>" +
                   "<i>Clica na legenda para filtrar políticas | Hover para ver seeds | Use para escolher seeds para viz_simple.py</i>",
        title_font_size=18,
        height=900,
        showlegend=True,
        hovermode='closest',
        template='plotly_white',
        legend=dict(
            title=dict(text="<b>POLÍTICAS</b><br>(Clica para filtrar)", font=dict(size=12)),
            orientation="v",
            yanchor="top",
            y=0.98,
            xanchor="left",
            x=1.01,
            bgcolor="rgba(255,255,255,0.8)",
            bordercolor="Black",
            borderwidth=1
        )
    )
    
    config = {
        'displayModeBar': True,
        'displaylogo': False,
        'toImageButtonOptions': {
            'format': 'png',
            'filename': 'seed_explorer_export',
            'height': 900,
            'width': 1400,
            'scale': 2
        }
    }
    fig.update_xaxes(title_text="Seed", row=1, col=1)
    fig.update_yaxes(title_text="Volatility", row=1, col=1)
    fig.update_xaxes(title_text="Volatility", row=2, col=1)
    fig.update_yaxes(title_text="Mispricing", row=2, col=1)
    fig.update_xaxes(title_text="Gini", row=2, col=2)
    fig.update_yaxes(title_text="Volume", row=2, col=2)
    
    config = {
        'displayModeBar': True,
        'displaylogo': False,
        'toImageButtonOptions': {
            'format': 'png',
            'filename': 'seed_explorer_export',
            'height': 900,
            'width': 1400,
            'scale': 2
        }
    }
    fig.write_html(output_file, config=config)
    print(f" Seed explorer saved: {output_file}")
    print(f"   Use para identificar seeds interessantes e depois correr:")
    print(f"   python3 viz_simple.py <policy> <steps> <seed>")


if __name__ == "__main__":
    print("="*70)
    print("INTERACTIVE DASHBOARD GENERATOR")
    print("="*70)
    
    try:
        import plotly
        print(f"✓ Plotly version: {plotly.__version__}")
    except ImportError:
        print(" ERROR: Plotly not installed!")
        print("   Install with: pip install plotly")
        print("   Or: python3 -m pip install plotly")
        exit(1)
    
    df = load_results()
    
    create_interactive_dashboard(df)
    create_individual_seed_viewer(df)
    
    print("\n" + "="*70)
    print("DASHBOARD CREATION COMPLETE!")
    print("="*70)
    print("\n Created 2 interactive HTML files:")
    print("  1. interactive_dashboard.html - Main dashboard with all visualizations")
    print("  2. seed_explorer.html         - Explore individual seeds interactively")
    print("\n How to use:")
    print("  - Double-click the HTML files to open in your browser")
    print("  - Hover over any point to see details")
    print("  - Click legend items to show/hide policies")
    print("  - Drag to zoom, double-click to reset zoom")
    print("  - In seed_explorer, find interesting seeds to run viz_simple.py")
    print("="*70)
