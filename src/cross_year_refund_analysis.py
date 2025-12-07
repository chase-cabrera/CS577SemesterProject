"""
Cross-Year Refund Analysis
Identifies contributors who donated to a committee in one year but received a refund the following year.
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from datetime import datetime
import os
import sys

sys.path.append(os.path.dirname(os.path.abspath(__file__)))
from db_utils import db_connection

OUTPUT_DIR = 'report/fraud_detection/cross_year_refunds'
os.makedirs(OUTPUT_DIR, exist_ok=True)

sns.set_style("whitegrid")
plt.rcParams['figure.figsize'] = (14, 10)
plt.rcParams['font.size'] = 11


def create_yearly_aggregation_tables(conn):
    """Create temp tables with yearly aggregations per contributor-committee pair."""
    cursor = conn.cursor()
    
    print("Creating yearly aggregation tables...")
    
    cursor.execute("DROP TABLE IF EXISTS tmp_yearly_donations")
    cursor.execute("DROP TABLE IF EXISTS tmp_yearly_refunds")
    conn.commit()
    
    print("  Aggregating yearly donations...")
    cursor.execute("""
        CREATE TABLE tmp_yearly_donations AS
        SELECT 
            contributor_id, cmte_id,
            YEAR(transaction_dt) as donation_year,
            COUNT(*) as donation_count,
            SUM(transaction_amt) as total_donated,
            MIN(transaction_dt) as first_donation,
            MAX(transaction_dt) as last_donation
        FROM contributions
        WHERE transaction_amt > 0 AND transaction_dt IS NOT NULL
        GROUP BY contributor_id, cmte_id, YEAR(transaction_dt)
    """)
    conn.commit()
    
    print("  Adding indexes to donations table...")
    cursor.execute("ALTER TABLE tmp_yearly_donations ADD INDEX idx_donor_cmte_year (contributor_id, cmte_id, donation_year)")
    cursor.execute("ALTER TABLE tmp_yearly_donations ADD INDEX idx_year (donation_year)")
    conn.commit()
    
    print("  Aggregating yearly refunds...")
    cursor.execute("""
        CREATE TABLE tmp_yearly_refunds AS
        SELECT 
            contributor_id, cmte_id,
            YEAR(transaction_dt) as refund_year,
            COUNT(*) as refund_count,
            SUM(ABS(transaction_amt)) as total_refunded,
            MIN(transaction_dt) as first_refund,
            MAX(transaction_dt) as last_refund
        FROM contributions
        WHERE transaction_amt < 0 AND transaction_dt IS NOT NULL
        GROUP BY contributor_id, cmte_id, YEAR(transaction_dt)
    """)
    conn.commit()
    
    print("  Adding indexes to refunds table...")
    cursor.execute("ALTER TABLE tmp_yearly_refunds ADD INDEX idx_donor_cmte_year (contributor_id, cmte_id, refund_year)")
    cursor.execute("ALTER TABLE tmp_yearly_refunds ADD INDEX idx_year (refund_year)")
    conn.commit()
    
    cursor.execute("SELECT COUNT(*) FROM tmp_yearly_donations")
    donation_rows = cursor.fetchone()[0]
    cursor.execute("SELECT COUNT(*) FROM tmp_yearly_refunds")
    refund_rows = cursor.fetchone()[0]
    
    print(f"  Created {donation_rows:,} yearly donation records")
    print(f"  Created {refund_rows:,} yearly refund records")
    
    cursor.close()
    return donation_rows, refund_rows


def find_cross_year_refunds(conn, year_gap=1):
    """Find contributors who donated in year X and received refund in year X+year_gap."""
    print(f"\nFinding donations followed by refunds {year_gap} year(s) later...")
    
    query = f"""
    SELECT 
        d.contributor_id, d.cmte_id, d.donation_year, r.refund_year,
        d.donation_count, d.total_donated, d.first_donation, d.last_donation,
        r.refund_count, r.total_refunded, r.first_refund, r.last_refund,
        (r.total_refunded / d.total_donated * 100) as refund_pct
    FROM tmp_yearly_donations d
    JOIN tmp_yearly_refunds r 
        ON d.contributor_id = r.contributor_id 
        AND d.cmte_id = r.cmte_id
        AND r.refund_year = d.donation_year + {year_gap}
    ORDER BY r.total_refunded DESC
    """
    
    df = pd.read_sql(query, conn)
    print(f"  Found {len(df):,} cross-year refund patterns")
    return df


def enrich_with_contributor_details(conn, df, sample_size=50000):
    """Add contributor and committee details to results using batch fetching."""
    if len(df) == 0:
        return df
    
    print("\nEnriching results with contributor and committee details...")
    
    if len(df) > sample_size:
        print(f"  Sampling {sample_size:,} records from {len(df):,} total")
        df_sample = df.head(sample_size).copy()
    else:
        df_sample = df.copy()
    
    contributor_ids = df_sample['contributor_id'].unique().tolist()
    print(f"  Fetching details for {len(contributor_ids):,} contributors...")
    
    batch_size = 10000
    contributor_data = []
    
    for i in range(0, len(contributor_ids), batch_size):
        batch_ids = contributor_ids[i:i+batch_size]
        id_str = ','.join(map(str, batch_ids))
        
        query = f"""
        SELECT id as contributor_id, first_name, last_name, city, state, zip_code,
               employer, occupation, primary_party,
               total_donations as lifetime_donations, total_amount as lifetime_amount
        FROM contributors WHERE id IN ({id_str})
        """
        contributor_data.append(pd.read_sql(query, conn))
    
    if contributor_data:
        df_contributors = pd.concat(contributor_data, ignore_index=True)
        df_sample = df_sample.merge(df_contributors, on='contributor_id', how='left')
    
    committee_ids = df_sample['cmte_id'].unique().tolist()
    if committee_ids:
        cmte_str = "','".join(committee_ids)
        query = f"""
        SELECT cmte_id, cmte_nm as committee_name, cmte_pty_affiliation as committee_party,
               cmte_tp as committee_type, cmte_st as committee_state
        FROM committees WHERE cmte_id IN ('{cmte_str}')
        """
        df_committees = pd.read_sql(query, conn)
        df_sample = df_sample.merge(df_committees, on='cmte_id', how='left')
    
    print(f"  Enriched {len(df_sample):,} records")
    return df_sample


def get_summary_statistics(conn, df):
    """Calculate comprehensive summary statistics."""
    print("\nCalculating summary statistics...")
    
    stats = {
        'total_patterns': len(df),
        'unique_contributors': df['contributor_id'].nunique(),
        'unique_committees': df['cmte_id'].nunique(),
        'total_donated': df['total_donated'].sum(),
        'total_refunded': df['total_refunded'].sum(),
        'avg_donated': df['total_donated'].mean(),
        'avg_refunded': df['total_refunded'].mean(),
        'median_donated': df['total_donated'].median(),
        'median_refunded': df['total_refunded'].median(),
        'avg_refund_pct': df['refund_pct'].mean(),
        'full_refunds': (df['refund_pct'] >= 100).sum(),
        'partial_refunds': ((df['refund_pct'] > 0) & (df['refund_pct'] < 100)).sum(),
        'excess_refunds': (df['refund_pct'] > 100).sum(),
        'min_donation_year': df['donation_year'].min(),
        'max_donation_year': df['donation_year'].max(),
    }
    
    cursor = conn.cursor()
    cursor.execute("SELECT COUNT(*), SUM(transaction_amt) FROM contributions WHERE transaction_amt > 0")
    result = cursor.fetchone()
    stats['total_contribution_count'] = result[0]
    stats['total_contribution_amount'] = float(result[1] or 0)
    cursor.close()
    
    for key in ['total_donated', 'total_refunded', 'avg_donated', 'avg_refunded', 
                'median_donated', 'median_refunded', 'avg_refund_pct']:
        if key in stats and stats[key] is not None:
            stats[key] = float(stats[key])
    
    return stats


def analyze_by_year(df):
    """Analyze cross-year refund patterns by donation year."""
    print("\nAnalyzing patterns by year...")
    
    year_analysis = df.groupby('donation_year').agg({
        'contributor_id': 'nunique',
        'cmte_id': 'nunique',
        'total_donated': ['sum', 'mean'],
        'total_refunded': ['sum', 'mean'],
        'refund_pct': 'mean'
    }).round(2)
    
    year_analysis.columns = [
        'unique_contributors', 'unique_committees',
        'total_donated', 'avg_donated',
        'total_refunded', 'avg_refunded',
        'avg_refund_pct'
    ]
    return year_analysis.reset_index()


def analyze_by_state(df):
    """Analyze cross-year refund patterns by contributor state."""
    if 'state' not in df.columns:
        return pd.DataFrame()
    
    print("\nAnalyzing patterns by state...")
    
    state_analysis = df.groupby('state').agg({
        'contributor_id': 'nunique',
        'total_donated': 'sum',
        'total_refunded': 'sum',
        'refund_pct': 'mean'
    }).round(2)
    
    state_analysis.columns = ['unique_contributors', 'total_donated', 'total_refunded', 'avg_refund_pct']
    return state_analysis.sort_values('total_refunded', ascending=False).reset_index()


def analyze_by_committee_party(df):
    """Analyze cross-year refund patterns by committee party."""
    if 'committee_party' not in df.columns:
        return pd.DataFrame()
    
    print("\nAnalyzing patterns by committee party...")
    
    party_analysis = df.groupby('committee_party').agg({
        'contributor_id': 'nunique',
        'cmte_id': 'nunique',
        'total_donated': 'sum',
        'total_refunded': 'sum',
        'refund_pct': 'mean'
    }).round(2)
    
    party_analysis.columns = ['unique_contributors', 'unique_committees', 'total_donated', 'total_refunded', 'avg_refund_pct']
    return party_analysis.sort_values('total_refunded', ascending=False).reset_index()


def analyze_refund_timing(df):
    """Analyze the timing of refunds relative to donations."""
    print("\nAnalyzing refund timing patterns...")
    
    df_timing = df.copy()
    df_timing['last_donation'] = pd.to_datetime(df_timing['last_donation'])
    df_timing['first_refund'] = pd.to_datetime(df_timing['first_refund'])
    df_timing['days_to_refund'] = (df_timing['first_refund'] - df_timing['last_donation']).dt.days
    
    timing_stats = {
        'avg_days_to_refund': df_timing['days_to_refund'].mean(),
        'median_days_to_refund': df_timing['days_to_refund'].median(),
        'min_days_to_refund': df_timing['days_to_refund'].min(),
        'max_days_to_refund': df_timing['days_to_refund'].max(),
        'refunds_within_30_days': (df_timing['days_to_refund'] <= 30).sum(),
        'refunds_within_90_days': (df_timing['days_to_refund'] <= 90).sum(),
        'refunds_within_180_days': (df_timing['days_to_refund'] <= 180).sum(),
        'refunds_within_365_days': (df_timing['days_to_refund'] <= 365).sum(),
    }
    return timing_stats, df_timing


def create_visualizations(df, df_enriched, stats, year_analysis, state_analysis, party_analysis, timing_stats):
    """Create comprehensive EDA visualizations."""
    print("\nGenerating visualizations...")
    
    fig, axes = plt.subplots(2, 3, figsize=(18, 12))
    
    ax = axes[0, 0]
    year_data = year_analysis.set_index('donation_year')
    ax.bar(year_data.index, year_data['total_refunded'] / 1e6, color='crimson', alpha=0.7, edgecolor='black')
    ax.set_xlabel('Donation Year', fontweight='bold')
    ax.set_ylabel('Total Refunded (Millions $)', fontweight='bold')
    ax.set_title('Cross-Year Refunds by Donation Year', fontweight='bold', fontsize=12)
    ax.grid(axis='y', alpha=0.3)
    
    ax = axes[0, 1]
    refund_pcts = df['refund_pct'].clip(upper=200)
    ax.hist(refund_pcts, bins=50, color='orangered', alpha=0.7, edgecolor='black')
    ax.axvline(x=100, color='black', linestyle='--', linewidth=2, label='100% (Full Refund)')
    ax.set_xlabel('Refund Percentage', fontweight='bold')
    ax.set_ylabel('Frequency', fontweight='bold')
    ax.set_title('Distribution of Refund Percentages', fontweight='bold', fontsize=12)
    ax.legend()
    ax.grid(alpha=0.3)
    
    ax = axes[0, 2]
    sample = df.sample(min(5000, len(df)))
    ax.scatter(sample['total_donated'], sample['total_refunded'], alpha=0.3, s=20, c='darkred')
    ax.plot([0, sample['total_donated'].max()], [0, sample['total_donated'].max()], 'k--', label='1:1 Line', linewidth=2)
    ax.set_xlabel('Total Donated ($)', fontweight='bold')
    ax.set_ylabel('Total Refunded ($)', fontweight='bold')
    ax.set_title('Donation vs Refund Amounts', fontweight='bold', fontsize=12)
    ax.legend()
    ax.grid(alpha=0.3)
    ax.set_xscale('log')
    ax.set_yscale('log')
    
    ax = axes[1, 0]
    categories = ['Partial\n(<100%)', 'Full\n(100%)', 'Excess\n(>100%)']
    values = [stats['partial_refunds'], stats['full_refunds'], stats['excess_refunds']]
    bars = ax.bar(categories, values, color=['gold', 'orange', 'red'], edgecolor='black', linewidth=2)
    ax.set_ylabel('Count', fontweight='bold')
    ax.set_title('Refund Categories', fontweight='bold', fontsize=12)
    ax.grid(axis='y', alpha=0.3)
    for bar, val in zip(bars, values):
        ax.text(bar.get_x() + bar.get_width()/2, bar.get_height(), f'{val:,}', ha='center', va='bottom', fontweight='bold')
    
    ax = axes[1, 1]
    if len(state_analysis) > 0:
        top_states = state_analysis.head(15)
        ax.barh(range(len(top_states)), top_states['total_refunded'] / 1e6, color='firebrick', alpha=0.7, edgecolor='black')
        ax.set_yticks(range(len(top_states)))
        ax.set_yticklabels(top_states['state'])
        ax.set_xlabel('Total Refunded (Millions $)', fontweight='bold')
        ax.set_title('Top 15 States by Cross-Year Refunds', fontweight='bold', fontsize=12)
        ax.grid(axis='x', alpha=0.3)
        ax.invert_yaxis()
    
    ax = axes[1, 2]
    if len(party_analysis) > 0:
        party_colors = {'DEM': '#3498db', 'REP': '#e74c3c', 'IND': '#95a5a6', 'LIB': '#f39c12', 'GRE': '#2ecc71', 'UNK': '#7f8c8d'}
        top_parties = party_analysis.head(6)
        colors = [party_colors.get(p, 'gray') for p in top_parties['committee_party']]
        ax.pie(top_parties['total_refunded'], labels=top_parties['committee_party'],
               autopct='%1.1f%%', colors=colors, startangle=90, textprops={'fontsize': 11, 'weight': 'bold'})
        ax.set_title('Cross-Year Refunds by Committee Party', fontweight='bold', fontsize=12)
    
    plt.suptitle('Cross-Year Refund Analysis: Donations in Year X, Refunds in Year X+1', fontsize=16, fontweight='bold', y=1.02)
    plt.tight_layout()
    plt.savefig(f'{OUTPUT_DIR}/cross_year_refunds_overview.png', dpi=300, bbox_inches='tight')
    print(f"  Saved {OUTPUT_DIR}/cross_year_refunds_overview.png")
    plt.close()
    
    if 'days_to_refund' in df.columns or timing_stats:
        fig, axes = plt.subplots(1, 2, figsize=(14, 5))
        
        ax = axes[0]
        if 'days_to_refund' in df.columns:
            days_clipped = df['days_to_refund'].dropna().clip(upper=730)
            ax.hist(days_clipped, bins=50, color='steelblue', alpha=0.7, edgecolor='black')
            ax.axvline(x=365, color='red', linestyle='--', linewidth=2, label='1 Year')
            ax.set_xlabel('Days from Last Donation to First Refund', fontweight='bold')
            ax.set_ylabel('Frequency', fontweight='bold')
            ax.set_title('Time Gap Between Donation and Refund', fontweight='bold', fontsize=12)
            ax.legend()
            ax.grid(alpha=0.3)
        
        ax = axes[1]
        time_labels = ['30 days', '90 days', '180 days', '1 year', '2 years']
        cumulative = [
            timing_stats.get('refunds_within_30_days', 0),
            timing_stats.get('refunds_within_90_days', 0),
            timing_stats.get('refunds_within_180_days', 0),
            timing_stats.get('refunds_within_365_days', 0),
            stats['total_patterns']
        ]
        ax.bar(time_labels, cumulative, color='teal', alpha=0.7, edgecolor='black')
        ax.set_xlabel('Time Since Last Donation', fontweight='bold')
        ax.set_ylabel('Cumulative Refunds', fontweight='bold')
        ax.set_title('Cumulative Cross-Year Refunds by Timing', fontweight='bold', fontsize=12)
        ax.grid(axis='y', alpha=0.3)
        
        plt.tight_layout()
        plt.savefig(f'{OUTPUT_DIR}/refund_timing_analysis.png', dpi=300, bbox_inches='tight')
        print(f"  Saved {OUTPUT_DIR}/refund_timing_analysis.png")
        plt.close()
    
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    
    ax = axes[0]
    year_data = year_analysis.set_index('donation_year')
    ax.plot(year_data.index, year_data['unique_contributors'], 'o-', color='navy', linewidth=2, markersize=8, label='Contributors')
    ax.set_xlabel('Donation Year', fontweight='bold')
    ax.set_ylabel('Unique Contributors', fontweight='bold', color='navy')
    ax.tick_params(axis='y', labelcolor='navy')
    
    ax2 = ax.twinx()
    ax2.plot(year_data.index, year_data['unique_committees'], 's--', color='darkgreen', linewidth=2, markersize=8, label='Committees')
    ax2.set_ylabel('Unique Committees', fontweight='bold', color='darkgreen')
    ax2.tick_params(axis='y', labelcolor='darkgreen')
    ax.set_title('Contributors and Committees with Cross-Year Refunds', fontweight='bold', fontsize=12)
    ax.grid(alpha=0.3)
    
    ax = axes[1]
    ax.plot(year_data.index, year_data['avg_donated'], 'o-', color='green', linewidth=2, markersize=8, label='Avg Donated')
    ax.plot(year_data.index, year_data['avg_refunded'], 's-', color='red', linewidth=2, markersize=8, label='Avg Refunded')
    ax.set_xlabel('Donation Year', fontweight='bold')
    ax.set_ylabel('Average Amount ($)', fontweight='bold')
    ax.set_title('Average Donation and Refund Amounts by Year', fontweight='bold', fontsize=12)
    ax.legend()
    ax.grid(alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(f'{OUTPUT_DIR}/year_over_year_trends.png', dpi=300, bbox_inches='tight')
    print(f"  Saved {OUTPUT_DIR}/year_over_year_trends.png")
    plt.close()


def cleanup_temp_tables(conn):
    """Remove temporary tables."""
    cursor = conn.cursor()
    cursor.execute("DROP TABLE IF EXISTS tmp_yearly_donations")
    cursor.execute("DROP TABLE IF EXISTS tmp_yearly_refunds")
    conn.commit()
    cursor.close()
    print("\nCleaned up temporary tables.")


def main():
    print("CROSS-YEAR REFUND ANALYSIS")
    print("Finding donors who contributed in Year X and received refunds in Year X+1\n")
    
    with db_connection() as conn:
        try:
            create_yearly_aggregation_tables(conn)
            df_cross_year = find_cross_year_refunds(conn, year_gap=1)
            
            if len(df_cross_year) == 0:
                print("\nNo cross-year refund patterns found.")
                cleanup_temp_tables(conn)
                return
            
            df_enriched = enrich_with_contributor_details(conn, df_cross_year)
            stats = get_summary_statistics(conn, df_cross_year)
            
            year_analysis = analyze_by_year(df_cross_year)
            state_analysis = analyze_by_state(df_enriched)
            party_analysis = analyze_by_committee_party(df_enriched)
            timing_stats, df_with_timing = analyze_refund_timing(df_cross_year)
            
            if 'days_to_refund' in df_with_timing.columns:
                df_enriched = df_enriched.merge(
                    df_with_timing[['contributor_id', 'cmte_id', 'donation_year', 'days_to_refund']],
                    on=['contributor_id', 'cmte_id', 'donation_year'],
                    how='left'
                )
            
            create_visualizations(df_cross_year, df_enriched, stats, year_analysis, state_analysis, party_analysis, timing_stats)
            cleanup_temp_tables(conn)
            
        except Exception as e:
            print(f"\nError: {e}")
            import traceback
            traceback.print_exc()
            cleanup_temp_tables(conn)
            raise
    
    print(f"\nAnalysis complete. Output: {OUTPUT_DIR}/")


if __name__ == "__main__":
    main()
