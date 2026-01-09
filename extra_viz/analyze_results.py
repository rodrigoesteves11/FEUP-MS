"""
Analyze aggregated results from run.py (30 seeds x 3 policies = 90 simulations)
Creates comprehensive statistical analysis and visualizations
"""
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import stats
import numpy as np

def load_results(csv_path="../results_new_model/kpi_results.csv"):
    """Load the KPI results from run.py"""
    df = pd.read_csv(csv_path)
    print(f"Loaded {len(df)} simulation results")
    print(f"Policies: {df['policy'].unique()}")
    print(f"Seeds per policy: {df.groupby('policy').size()}")
    return df

def create_comprehensive_analysis(df, output_dir="analysis_results"):
    """Create comprehensive analysis with multiple visualizations"""
    
    # 1. BOXPLOTS - Main comparison across policies
    print("\n[1/5] Creating boxplots...")
    fig, axes = plt.subplots(3, 3, figsize=(15, 12))
    fig.suptitle('KPI Comparison Across Policies (30 seeds each)', fontsize=16, fontweight='bold')
    
    kpis = [
        ('vol_mean', 'Volatility (Mean)'),
        ('vol_max', 'Volatility (Max)'),
        ('mean_abs_mispricing', 'Mean Abs Mispricing'),
        ('mean_abs_rel_mispricing', 'Mean Abs Rel Mispricing'),
        ('volume_mean', 'Mean Volume'),
        ('turnover_mean', 'Mean Turnover'),
        ('gini_mean', 'Mean Gini'),
        ('gini_final', 'Final Gini'),
        ('max_drawdown', 'Max Drawdown')
    ]
    
    policies_order = ['none', 'moderate', 'excessive']
    colors = {'none': '#2ecc71', 'moderate': '#f39c12', 'excessive': '#e74c3c'}
    
    for idx, (kpi_col, kpi_name) in enumerate(kpis):
        ax = axes[idx // 3, idx % 3]
        
        # Create boxplot
        bp = ax.boxplot(
            [df[df['policy'] == p][kpi_col].values for p in policies_order],
            labels=[p.upper() for p in policies_order],
            patch_artist=True,
            widths=0.6
        )
        
        # Color boxes
        for patch, policy in zip(bp['boxes'], policies_order):
            patch.set_facecolor(colors[policy])
            patch.set_alpha(0.7)
        
        ax.set_title(kpi_name, fontsize=11, fontweight='bold')
        ax.grid(axis='y', alpha=0.3)
        ax.tick_params(axis='x', rotation=45)
    
    plt.tight_layout()
    plt.savefig(f"{output_dir}/analysis_boxplots.png", dpi=150, bbox_inches='tight')
    plt.close()
    print(f"   Saved: {output_dir}/analysis_boxplots.png")
    
    # 2. DISTRIBUTIONS - Histograms for key KPIs
    print("[2/5] Creating distributions...")
    fig, axes = plt.subplots(2, 3, figsize=(15, 8))
    fig.suptitle('KPI Distributions by Policy', fontsize=16, fontweight='bold')
    
    key_kpis = [
        ('vol_mean', 'Volatility'),
        ('mean_abs_mispricing', 'Mean Abs Mispricing'),
        ('volume_mean', 'Mean Volume'),
        ('gini_mean', 'Mean Gini'),
        ('max_drawdown', 'Max Drawdown'),
        ('n_crashes_ret', 'Number of Crashes')
    ]
    
    for idx, (kpi_col, kpi_name) in enumerate(key_kpis):
        ax = axes[idx // 3, idx % 3]
        
        for policy in policies_order:
            data = df[df['policy'] == policy][kpi_col].values
            ax.hist(data, bins=15, alpha=0.6, label=policy.upper(), color=colors[policy], edgecolor='black')
        
        ax.set_title(kpi_name, fontsize=11, fontweight='bold')
        ax.set_xlabel('Value')
        ax.set_ylabel('Frequency')
        ax.legend()
        ax.grid(alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(f"{output_dir}/analysis_distributions.png", dpi=150, bbox_inches='tight')
    plt.close()
    print(f"   Saved: {output_dir}/analysis_distributions.png")
    
    # 3. STATISTICS TABLE
    print("[3/5] Creating statistics table...")
    fig, ax = plt.subplots(figsize=(14, 10))
    ax.axis('tight')
    ax.axis('off')
    
    stats_data = []
    for kpi_col, kpi_name in kpis:
        row = [kpi_name]
        for policy in policies_order:
            data = df[df['policy'] == policy][kpi_col].values
            mean_val = np.mean(data)
            std_val = np.std(data)
            row.append(f"{mean_val:.4f} ± {std_val:.4f}")
        stats_data.append(row)
    
    table = ax.table(
        cellText=stats_data,
        colLabels=['KPI', 'NONE', 'MODERATE', 'EXCESSIVE'],
        cellLoc='center',
        loc='center',
        colWidths=[0.4, 0.2, 0.2, 0.2]
    )
    table.auto_set_font_size(False)
    table.set_fontsize(9)
    table.scale(1, 2)
    
    # Header styling
    for i in range(4):
        table[(0, i)].set_facecolor('#34495e')
        table[(0, i)].set_text_props(weight='bold', color='white')
    
    plt.title('Statistical Summary: Mean ± Std (30 seeds each)', 
              fontsize=14, fontweight='bold', pad=20)
    plt.savefig(f"{output_dir}/analysis_statistics.png", dpi=150, bbox_inches='tight')
    plt.close()
    print(f"   Saved: {output_dir}/analysis_statistics.png")
    
    # 4. SIGNIFICANCE TESTS
    print("[4/5] Running statistical tests...")
    fig, ax = plt.subplots(figsize=(12, 8))
    ax.axis('tight')
    ax.axis('off')
    
    test_results = []
    comparisons = [('none', 'moderate'), ('none', 'excessive'), ('moderate', 'excessive')]
    
    for kpi_col, kpi_name in kpis[:6]:  # Test first 6 KPIs
        row = [kpi_name]
        for policy1, policy2 in comparisons:
            data1 = df[df['policy'] == policy1][kpi_col].values
            data2 = df[df['policy'] == policy2][kpi_col].values
            
            # Mann-Whitney U test (non-parametric)
            statistic, pvalue = stats.mannwhitneyu(data1, data2, alternative='two-sided')
            
            if pvalue < 0.001:
                sig = "***"
            elif pvalue < 0.01:
                sig = "**"
            elif pvalue < 0.05:
                sig = "*"
            else:
                sig = "ns"
            
            row.append(f"p={pvalue:.4f} {sig}")
        
        test_results.append(row)
    
    table = ax.table(
        cellText=test_results,
        colLabels=['KPI', 'None vs Mod', 'None vs Exc', 'Mod vs Exc'],
        cellLoc='center',
        loc='center',
        colWidths=[0.3, 0.23, 0.23, 0.23]
    )
    table.auto_set_font_size(False)
    table.set_fontsize(9)
    table.scale(1, 2.5)
    
    for i in range(4):
        table[(0, i)].set_facecolor('#34495e')
        table[(0, i)].set_text_props(weight='bold', color='white')
    
    plt.title('Statistical Significance Tests (Mann-Whitney U)\n*** p<0.001  ** p<0.01  * p<0.05  ns = not significant',
              fontsize=12, fontweight='bold', pad=20)
    plt.savefig(f"{output_dir}/analysis_significance.png", dpi=150, bbox_inches='tight')
    plt.close()
    print(f"   Saved: {output_dir}/analysis_significance.png")
    
    # 5. TEXT REPORT
    print("[5/5] Creating text report...")
    with open(f"{output_dir}/analysis_report.txt", 'w') as f:
        f.write("="*70 + "\n")
        f.write("MARKET ABM - STATISTICAL ANALYSIS REPORT\n")
        f.write("="*70 + "\n")
        f.write(f"Dataset: 90 simulations (3 policies x 30 seeds)\n")
        f.write(f"Policies: {', '.join(policies_order)}\n")
        f.write("="*70 + "\n\n")
        
        for kpi_col, kpi_name in kpis:
            f.write(f"\n{kpi_name}:\n")
            f.write("-" * 50 + "\n")
            for policy in policies_order:
                data = df[df['policy'] == policy][kpi_col].values
                f.write(f"  {policy.upper():12s}: {np.mean(data):10.4f} ± {np.std(data):8.4f} ")
                f.write(f"[min={np.min(data):.4f}, max={np.max(data):.4f}]\n")
    
    print(f"   Saved: {output_dir}/analysis_report.txt")
    
    print("\n" + "="*70)
    print("ANALYSIS COMPLETE!")
    print("="*70)
    print(f"Created 5 output files in '{output_dir}/':")
    print(f"  1. analysis_boxplots.png      - Boxplot comparison of all KPIs")
    print(f"  2. analysis_distributions.png - Histograms for key KPIs")
    print(f"  3. analysis_statistics.png    - Table with mean ± std")
    print(f"  4. analysis_significance.png  - Statistical significance tests")
    print(f"  5. analysis_report.txt        - Full text report")
    print("="*70)

if __name__ == "__main__":
    # Load results from run.py
    df = load_results()
    
    # Create comprehensive analysis
    create_comprehensive_analysis(df)
