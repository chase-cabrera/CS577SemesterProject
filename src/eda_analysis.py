"""FEC Campaign Finance - Exploratory Data Analysis"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from matplotlib.patches import Patch
from db_utils import db_connection
import os

sns.set_style("whitegrid")
plt.rcParams['figure.figsize'] = (16, 10)

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.dirname(SCRIPT_DIR)
OUTPUT_DIR = os.path.join(PROJECT_ROOT, "report")
os.makedirs(OUTPUT_DIR, exist_ok=True)

PARTY_COLORS = {'Democrat': 'blue', 'Republican': 'red', 'Other/Independent': 'grey'}
PARTY_COLORS_SHORT = {'DEM': 'blue', 'REP': 'red', 'OTHER': 'grey'}


def execute_query(query, use_dict=True):
    """Execute SQL query and return results."""
    with db_connection(use_dict_cursor=use_dict) as conn:
        if use_dict:
            cursor = conn.cursor()
            cursor.execute(query)
            return cursor.fetchall()
        return pd.read_sql(query, conn)


def get_summary_statistics():
    """Get summary statistics from contributors table."""
    query = """
    SELECT 
        COUNT(*) as total_contributors,
        SUM(total_amount) as total_donated,
        AVG(total_amount) as avg_total_per_contributor,
        SUM(total_donations) as total_transactions,
        AVG(total_donations) as avg_donations_per_contributor,
        AVG(avg_donation_amount) as avg_donation_amount
    FROM contributors
    WHERE total_amount > 0
    """
    return execute_query(query)[0]


def load_distribution_data():
    """Load contributor data for distribution visualizations."""
    print("Loading contributor data for distributions...")
    query = """
    SELECT total_donations, total_amount, avg_donation_amount, primary_party
    FROM contributors WHERE total_amount > 0
    """
    df = execute_query(query, use_dict=False)
    print(f"Loaded {len(df):,} contributors")
    return df


def get_party_label(party):
    """Convert party code to full label."""
    if pd.notna(party) and str(party).upper() in ['DEM', 'D']:
        return 'Democrat'
    elif pd.notna(party) and str(party).upper() in ['REP', 'R']:
        return 'Republican'
    return 'Other/Independent'


def create_donation_amount_distribution(df):
    """Create histogram of total donation amounts (log scale)."""
    print("Creating donation amount distribution...")
    
    total_amount = np.array(df['total_amount'], dtype=float)
    log_amounts = np.log10(np.clip(total_amount, 1, None))
    
    fig, ax = plt.subplots(figsize=(12, 8))
    ax.hist(log_amounts, bins=50, color='steelblue', edgecolor='black', alpha=0.7)
    ax.set_xlabel('Log10(Total Amount Donated)', fontsize=12)
    ax.set_ylabel('Number of Donors', fontsize=12)
    ax.set_title('Distribution of Total Donation Amount per Donor', fontsize=14, fontweight='bold')
    ax.grid(True, alpha=0.3)
    
    median_val = np.median(total_amount)
    mean_val = np.mean(total_amount)
    ax.axvline(np.log10(median_val), color='red', linestyle='--', linewidth=2, label=f'Median: ${median_val:,.0f}')
    ax.axvline(np.log10(mean_val), color='green', linestyle='--', linewidth=2, label=f'Mean: ${mean_val:,.0f}')
    ax.legend(fontsize=10)
    
    plt.tight_layout()
    plt.savefig(os.path.join(OUTPUT_DIR, 'donation_amount_distribution.png'), dpi=300, bbox_inches='tight')
    print("Saved: donation_amount_distribution.png")
    plt.close()


def create_donations_per_donor(df):
    """Create histogram of number of donations per donor."""
    print("Creating donations per donor distribution...")
    
    total_donations = np.array(df['total_donations'], dtype=float)
    bins = min(int(np.max(total_donations)), 200)
    
    fig, ax = plt.subplots(figsize=(12, 8))
    ax.hist(total_donations, bins=bins, color='steelblue', edgecolor='black', alpha=0.7)
    ax.set_xlabel('Number of Donations', fontsize=12)
    ax.set_ylabel('Number of Donors (log scale)', fontsize=12)
    ax.set_yscale('log')
    ax.set_title('Distribution of Donation Frequency per Donor', fontsize=14, fontweight='bold')
    ax.grid(True, alpha=0.3)
    
    one_time = np.sum(total_donations == 1)
    repeat = np.sum(total_donations > 1)
    ax.text(0.95, 0.95, f'One-time donors: {one_time:,} ({one_time/len(df)*100:.1f}%)\nRepeat donors: {repeat:,} ({repeat/len(df)*100:.1f}%)',
            transform=ax.transAxes, fontsize=11, verticalalignment='top', horizontalalignment='right',
            bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))
    
    plt.tight_layout()
    plt.savefig(os.path.join(OUTPUT_DIR, 'donations_per_donor.png'), dpi=300, bbox_inches='tight')
    print("Saved: donations_per_donor.png")
    plt.close()


def create_avg_donation_distribution(df):
    """Create histogram of average donation amounts."""
    print("Creating average donation distribution...")
    
    avg_donation_amount = np.array(df['avg_donation_amount'], dtype=float)
    log_avg = np.log10(np.clip(avg_donation_amount, 1, None))
    
    fig, ax = plt.subplots(figsize=(12, 8))
    ax.hist(log_avg, bins=50, color='steelblue', edgecolor='black', alpha=0.7)
    ax.set_xlabel('Log10(Average Donation Amount)', fontsize=12)
    ax.set_ylabel('Number of Donors', fontsize=12)
    ax.set_title('Distribution of Average Donation Size per Donor', fontsize=14, fontweight='bold')
    ax.grid(True, alpha=0.3)
    
    median_val = np.median(avg_donation_amount)
    ax.axvline(np.log10(median_val), color='red', linestyle='--', linewidth=2, label=f'Median: ${median_val:,.0f}')
    ax.legend(fontsize=10)
    
    plt.tight_layout()
    plt.savefig(os.path.join(OUTPUT_DIR, 'avg_donation_distribution.png'), dpi=300, bbox_inches='tight')
    print("Saved: avg_donation_distribution.png")
    plt.close()


def create_donations_vs_amount_scatter(df):
    """Create scatter plot of donations count vs total amount, colored by party."""
    print("Creating donations vs amount scatter plot...")
    
    total_donations = np.array(df['total_donations'], dtype=float)
    total_amount = np.array(df['total_amount'], dtype=float)
    parties = df['primary_party'].fillna('OTHER').values
    
    fig, ax = plt.subplots(figsize=(14, 10))
    
    for party_set, color, label in [({'DEM'}, 'blue', 'Democrat'),
                                     ({'REP'}, 'red', 'Republican'), 
                                     ({'OTHER', 'LIB', 'GRE', 'IND', ''}, 'grey', 'Other/Independent')]:
        mask = np.array([str(p).upper() in party_set for p in parties])
        if np.sum(mask) > 0:
            ax.scatter(total_donations[mask], 
                      np.log10(np.clip(total_amount[mask], 1, None)),
                      alpha=0.15, s=1, color=color, label=label, rasterized=True)
    
    ax.set_xlabel('Number of Donations (log scale)', fontsize=12)
    ax.set_ylabel('Log10(Total Amount)', fontsize=12)
    ax.set_xscale('log')
    ax.set_title(f'Donation Frequency vs Total Amount by Party ({len(df):,} donors)', fontsize=14, fontweight='bold')
    ax.grid(True, alpha=0.3, which='both')
    ax.legend(markerscale=10, fontsize=11, loc='lower right')
    
    plt.tight_layout()
    plt.savefig(os.path.join(OUTPUT_DIR, 'donations_vs_amount_scatter.png'), dpi=300, bbox_inches='tight')
    print("Saved: donations_vs_amount_scatter.png")
    plt.close()


def create_annual_trends():
    """Create annual donation trends visualization."""
    print("Creating annual trends visualization...")
    
    query = """
    SELECT 
        YEAR(last_donation_date) as year,
        COUNT(*) as num_donors,
        SUM(total_amount) as total_donated,
        AVG(total_amount) as avg_donation
    FROM contributors
    WHERE total_amount > 0 
      AND last_donation_date IS NOT NULL
      AND YEAR(last_donation_date) BETWEEN 2010 AND 2025
    GROUP BY YEAR(last_donation_date)
    ORDER BY year
    """
    
    df = pd.DataFrame(execute_query(query))
    for col in ['num_donors', 'total_donated', 'avg_donation', 'year']:
        df[col] = pd.to_numeric(df[col])
    
    fig, axes = plt.subplots(2, 2, figsize=(16, 12))
    fig.suptitle('Annual Donation Trends (2010-2025)', fontsize=16, fontweight='bold')
    
    axes[0, 0].bar(df['year'], df['total_donated'] / 1e9, color='steelblue', edgecolor='black')
    axes[0, 0].set_xlabel('Year')
    axes[0, 0].set_ylabel('Total Donated (billions $)')
    axes[0, 0].set_title('Total Amount Donated by Year', fontweight='bold')
    axes[0, 0].grid(True, alpha=0.3, axis='y')
    
    axes[0, 1].bar(df['year'], df['num_donors'] / 1e6, color='forestgreen', edgecolor='black')
    axes[0, 1].set_xlabel('Year')
    axes[0, 1].set_ylabel('Number of Donors (millions)')
    axes[0, 1].set_title('Number of Active Donors by Year', fontweight='bold')
    axes[0, 1].grid(True, alpha=0.3, axis='y')
    
    axes[1, 0].plot(df['year'], df['avg_donation'], marker='o', linewidth=2, markersize=8, color='purple')
    axes[1, 0].set_xlabel('Year')
    axes[1, 0].set_ylabel('Average Total Donation ($)')
    axes[1, 0].set_title('Average Donation per Donor by Year', fontweight='bold')
    axes[1, 0].grid(True, alpha=0.3)
    
    df['yoy_growth'] = df['total_donated'].pct_change() * 100
    colors = ['green' if x >= 0 else 'red' for x in df['yoy_growth'].fillna(0)]
    axes[1, 1].bar(df['year'], df['yoy_growth'].fillna(0), color=colors, edgecolor='black')
    axes[1, 1].set_xlabel('Year')
    axes[1, 1].set_ylabel('Year-over-Year Growth (%)')
    axes[1, 1].set_title('Annual Donation Growth Rate', fontweight='bold')
    axes[1, 1].axhline(y=0, color='black', linestyle='-', linewidth=0.5)
    axes[1, 1].grid(True, alpha=0.3, axis='y')
    
    plt.tight_layout()
    plt.savefig(os.path.join(OUTPUT_DIR, 'annual_trends.png'), dpi=300, bbox_inches='tight')
    print("Saved: annual_trends.png")
    plt.close()


def create_election_cycle_analysis():
    """Create election cycle analysis (presidential vs midterm years)."""
    print("Creating election cycle analysis...")
    
    query = """
    SELECT 
        YEAR(last_donation_date) as year,
        SUM(total_amount) as total_donated,
        COUNT(*) as num_donors,
        AVG(total_amount) as avg_donation
    FROM contributors
    WHERE total_amount > 0 
      AND last_donation_date IS NOT NULL
      AND YEAR(last_donation_date) BETWEEN 2010 AND 2024
    GROUP BY YEAR(last_donation_date)
    ORDER BY year
    """
    
    df = pd.DataFrame(execute_query(query))
    for col in df.columns:
        df[col] = pd.to_numeric(df[col])
    
    presidential_years = [2012, 2016, 2020, 2024]
    midterm_years = [2010, 2014, 2018, 2022]
    
    df['cycle'] = df['year'].apply(lambda y: 'Presidential' if y in presidential_years 
                                    else ('Midterm' if y in midterm_years else 'Off-Year'))
    
    fig, axes = plt.subplots(1, 3, figsize=(18, 6))
    fig.suptitle('Election Cycle Analysis', fontsize=16, fontweight='bold')
    
    cycle_colors = {'Presidential': 'darkblue', 'Midterm': 'darkorange', 'Off-Year': 'grey'}
    colors = [cycle_colors[c] for c in df['cycle']]
    
    axes[0].bar(df['year'], df['total_donated'] / 1e9, color=colors, edgecolor='black')
    axes[0].set_xlabel('Year')
    axes[0].set_ylabel('Total Donated (billions $)')
    axes[0].set_title('Donations by Election Cycle', fontweight='bold')
    axes[0].legend(handles=[Patch(facecolor=c, label=l) for l, c in cycle_colors.items()], loc='upper left')
    axes[0].grid(True, alpha=0.3, axis='y')
    
    cycle_summary = df[df['cycle'].isin(['Presidential', 'Midterm'])].groupby('cycle').agg({
        'total_donated': 'mean', 'num_donors': 'mean', 'avg_donation': 'mean'
    })
    
    x = np.arange(2)
    width = 0.35
    axes[1].bar(x, cycle_summary['total_donated'] / 1e9, width, 
                color=[cycle_colors['Presidential'], cycle_colors['Midterm']], edgecolor='black')
    axes[1].set_xticks(x)
    axes[1].set_xticklabels(['Presidential', 'Midterm'])
    axes[1].set_ylabel('Average Total Donated (billions $)')
    axes[1].set_title('Avg Donations: Presidential vs Midterm', fontweight='bold')
    axes[1].grid(True, alpha=0.3, axis='y')
    
    axes[2].bar(x, cycle_summary['num_donors'] / 1e6, width,
                color=[cycle_colors['Presidential'], cycle_colors['Midterm']], edgecolor='black')
    axes[2].set_xticks(x)
    axes[2].set_xticklabels(['Presidential', 'Midterm'])
    axes[2].set_ylabel('Average Number of Donors (millions)')
    axes[2].set_title('Avg Donor Count: Presidential vs Midterm', fontweight='bold')
    axes[2].grid(True, alpha=0.3, axis='y')
    
    plt.tight_layout()
    plt.savefig(os.path.join(OUTPUT_DIR, 'election_cycle_analysis.png'), dpi=300, bbox_inches='tight')
    print("Saved: election_cycle_analysis.png")
    plt.close()


def create_occupation_employer_analysis():
    """Create occupation and employer analysis."""
    print("Creating occupation/employer analysis...")
    
    occ_query = """
    SELECT occupation, COUNT(*) as num_donors, SUM(total_amount) as total_donated, AVG(total_amount) as avg_donation
    FROM contributors
    WHERE total_amount > 0 AND occupation IS NOT NULL AND occupation != ''
      AND occupation NOT IN ('INFORMATION REQUESTED', 'NONE', 'N/A', 'NA')
    GROUP BY occupation ORDER BY total_donated DESC LIMIT 15
    """
    
    emp_query = """
    SELECT employer, COUNT(*) as num_donors, SUM(total_amount) as total_donated, AVG(total_amount) as avg_donation
    FROM contributors
    WHERE total_amount > 0 AND employer IS NOT NULL AND employer != ''
      AND employer NOT IN ('INFORMATION REQUESTED', 'NONE', 'N/A', 'NA', 'SELF', 'SELF-EMPLOYED', 'SELF EMPLOYED')
    GROUP BY employer ORDER BY total_donated DESC LIMIT 15
    """
    
    occ_df = pd.DataFrame(execute_query(occ_query))
    emp_df = pd.DataFrame(execute_query(emp_query))
    
    for df in [occ_df, emp_df]:
        for col in ['num_donors', 'total_donated', 'avg_donation']:
            df[col] = pd.to_numeric(df[col])
    
    fig, axes = plt.subplots(2, 2, figsize=(18, 14))
    fig.suptitle('Occupation & Employer Analysis', fontsize=16, fontweight='bold')
    
    occ_sorted = occ_df.sort_values('total_donated')
    axes[0, 0].barh(range(len(occ_sorted)), occ_sorted['total_donated'] / 1e9, color='steelblue', edgecolor='black')
    axes[0, 0].set_yticks(range(len(occ_sorted)))
    axes[0, 0].set_yticklabels([o[:30] for o in occ_sorted['occupation']], fontsize=9)
    axes[0, 0].set_xlabel('Total Donated (billions $)')
    axes[0, 0].set_title('Top 15 Occupations by Total Donated', fontweight='bold')
    axes[0, 0].grid(True, alpha=0.3, axis='x')
    
    emp_sorted = emp_df.sort_values('total_donated')
    axes[0, 1].barh(range(len(emp_sorted)), emp_sorted['total_donated'] / 1e9, color='forestgreen', edgecolor='black')
    axes[0, 1].set_yticks(range(len(emp_sorted)))
    axes[0, 1].set_yticklabels([e[:30] for e in emp_sorted['employer']], fontsize=9)
    axes[0, 1].set_xlabel('Total Donated (billions $)')
    axes[0, 1].set_title('Top 15 Employers by Total Donated', fontweight='bold')
    axes[0, 1].grid(True, alpha=0.3, axis='x')
    
    occ_by_avg = occ_df.nlargest(15, 'avg_donation').sort_values('avg_donation')
    axes[1, 0].barh(range(len(occ_by_avg)), occ_by_avg['avg_donation'], color='purple', edgecolor='black')
    axes[1, 0].set_yticks(range(len(occ_by_avg)))
    axes[1, 0].set_yticklabels([o[:30] for o in occ_by_avg['occupation']], fontsize=9)
    axes[1, 0].set_xlabel('Average Donation ($)')
    axes[1, 0].set_title('Top 15 Occupations by Avg Donation', fontweight='bold')
    axes[1, 0].grid(True, alpha=0.3, axis='x')
    
    occ_by_count = occ_df.nlargest(15, 'num_donors').sort_values('num_donors')
    axes[1, 1].barh(range(len(occ_by_count)), occ_by_count['num_donors'] / 1e3, color='darkorange', edgecolor='black')
    axes[1, 1].set_yticks(range(len(occ_by_count)))
    axes[1, 1].set_yticklabels([o[:30] for o in occ_by_count['occupation']], fontsize=9)
    axes[1, 1].set_xlabel('Number of Donors (thousands)')
    axes[1, 1].set_title('Top 15 Occupations by Donor Count', fontweight='bold')
    axes[1, 1].grid(True, alpha=0.3, axis='x')
    
    plt.tight_layout()
    plt.savefig(os.path.join(OUTPUT_DIR, 'occupation_employer_analysis.png'), dpi=300, bbox_inches='tight')
    print("Saved: occupation_employer_analysis.png")
    plt.close()


def create_donor_size_segments():
    """Create small vs medium vs large donor analysis."""
    print("Creating donor size segment analysis...")
    
    query = """
    SELECT 
        CASE 
            WHEN total_amount < 200 THEN 'Small (<$200)'
            WHEN total_amount BETWEEN 200 AND 2900 THEN 'Medium ($200-$2,900)'
            ELSE 'Large (>$2,900)'
        END as donor_segment,
        primary_party,
        COUNT(*) as num_donors,
        SUM(total_amount) as total_donated,
        AVG(total_amount) as avg_donation
    FROM contributors
    WHERE total_amount > 0
    GROUP BY donor_segment, primary_party
    """
    
    df = pd.DataFrame(execute_query(query))
    for col in ['num_donors', 'total_donated', 'avg_donation']:
        df[col] = pd.to_numeric(df[col])
    
    segment_order = ['Small (<$200)', 'Medium ($200-$2,900)', 'Large (>$2,900)']
    
    fig, axes = plt.subplots(2, 2, figsize=(16, 12))
    fig.suptitle('Donor Size Segment Analysis (FEC Thresholds)', fontsize=16, fontweight='bold')
    
    segment_totals = df.groupby('donor_segment')['num_donors'].sum().reindex(segment_order)
    axes[0, 0].bar(range(len(segment_order)), segment_totals / 1e6, color='steelblue', edgecolor='black')
    axes[0, 0].set_xticks(range(len(segment_order)))
    axes[0, 0].set_xticklabels(segment_order, fontsize=10)
    axes[0, 0].set_ylabel('Number of Donors (millions)')
    axes[0, 0].set_title('Number of Donors by Segment', fontweight='bold')
    axes[0, 0].grid(True, alpha=0.3, axis='y')
    for i, v in enumerate(segment_totals):
        axes[0, 0].text(i, v/1e6, f'{v/1e6:.1f}M', ha='center', va='bottom')
    
    segment_amounts = df.groupby('donor_segment')['total_donated'].sum().reindex(segment_order)
    axes[0, 1].bar(range(len(segment_order)), segment_amounts / 1e9, color='forestgreen', edgecolor='black')
    axes[0, 1].set_xticks(range(len(segment_order)))
    axes[0, 1].set_xticklabels(segment_order, fontsize=10)
    axes[0, 1].set_ylabel('Total Donated (billions $)')
    axes[0, 1].set_title('Total Amount by Segment', fontweight='bold')
    axes[0, 1].grid(True, alpha=0.3, axis='y')
    for i, v in enumerate(segment_amounts):
        axes[0, 1].text(i, v/1e9, f'${v/1e9:.1f}B', ha='center', va='bottom')
    
    pivot = df.pivot_table(values='num_donors', index='donor_segment', columns='primary_party', aggfunc='sum').fillna(0)
    pivot = pivot.reindex(segment_order)
    party_cols = ['DEM', 'REP', 'OTHER'] if 'OTHER' in pivot.columns else [c for c in pivot.columns if c in ['DEM', 'REP']]
    available_cols = [c for c in party_cols if c in pivot.columns]
    
    bottom = np.zeros(len(segment_order))
    for party in available_cols:
        if party in pivot.columns:
            vals = pivot[party].values / 1e6
            color = PARTY_COLORS_SHORT.get(party, 'grey')
            axes[1, 0].bar(range(len(segment_order)), vals, bottom=bottom, label=party, color=color, edgecolor='black')
            bottom += vals
    
    axes[1, 0].set_xticks(range(len(segment_order)))
    axes[1, 0].set_xticklabels(segment_order, fontsize=10)
    axes[1, 0].set_ylabel('Number of Donors (millions)')
    axes[1, 0].set_title('Party Distribution by Donor Segment', fontweight='bold')
    axes[1, 0].legend()
    axes[1, 0].grid(True, alpha=0.3, axis='y')
    
    segment_avg = df.groupby('donor_segment')['avg_donation'].mean().reindex(segment_order)
    axes[1, 1].bar(range(len(segment_order)), segment_avg, color='purple', edgecolor='black')
    axes[1, 1].set_xticks(range(len(segment_order)))
    axes[1, 1].set_xticklabels(segment_order, fontsize=10)
    axes[1, 1].set_ylabel('Average Donation ($)')
    axes[1, 1].set_title('Average Donation by Segment', fontweight='bold')
    axes[1, 1].grid(True, alpha=0.3, axis='y')
    for i, v in enumerate(segment_avg):
        axes[1, 1].text(i, v, f'${v:,.0f}', ha='center', va='bottom')
    
    plt.tight_layout()
    plt.savefig(os.path.join(OUTPUT_DIR, 'donor_size_segments.png'), dpi=300, bbox_inches='tight')
    print("Saved: donor_size_segments.png")
    plt.close()


def create_party_by_state():
    """Create party preferences by state visualization."""
    print("Creating party by state analysis...")
    
    query = """
    SELECT state, primary_party, COUNT(*) as num_donors, SUM(total_amount) as total_donated
    FROM contributors
    WHERE total_amount > 0 AND state IS NOT NULL AND state != '' AND LENGTH(state) = 2
    GROUP BY state, primary_party
    """
    
    df = pd.DataFrame(execute_query(query))
    for col in ['num_donors', 'total_donated']:
        df[col] = pd.to_numeric(df[col])
    
    pivot = df.pivot_table(values='total_donated', index='state', columns='primary_party', aggfunc='sum').fillna(0)
    
    if 'DEM' not in pivot.columns or 'REP' not in pivot.columns:
        print("Warning: Could not create party_by_state - missing DEM or REP data")
        return
    
    pivot['total'] = pivot.sum(axis=1)
    pivot['dem_pct'] = pivot['DEM'] / pivot['total'] * 100
    pivot['rep_pct'] = pivot['REP'] / pivot['total'] * 100
    pivot['lean'] = pivot['dem_pct'] - pivot['rep_pct']
    pivot = pivot[pivot['total'] > 1e6].sort_values('lean')
    
    fig, axes = plt.subplots(1, 2, figsize=(18, 10))
    fig.suptitle('Party Preferences by State', fontsize=16, fontweight='bold')
    
    colors = ['blue' if x > 0 else 'red' for x in pivot['lean']]
    axes[0].barh(range(len(pivot)), pivot['lean'], color=colors, edgecolor='black', alpha=0.7)
    axes[0].set_yticks(range(len(pivot)))
    axes[0].set_yticklabels(pivot.index, fontsize=8)
    axes[0].set_xlabel('Democratic Lean <- -> Republican Lean')
    axes[0].set_title('State Political Lean (by donation amount)', fontweight='bold')
    axes[0].axvline(x=0, color='black', linestyle='-', linewidth=1)
    axes[0].grid(True, alpha=0.3, axis='x')
    
    top_dem = pivot.nlargest(10, 'dem_pct')
    top_rep = pivot.nsmallest(10, 'lean').head(10)
    
    dem_states = top_dem.index.tolist()
    rep_states = top_rep.index.tolist()
    
    y_pos = np.arange(10)
    width = 0.35
    
    axes[1].barh(y_pos - width/2, top_dem['dem_pct'].values, width, label='Top 10 Democratic', color='blue', edgecolor='black')
    axes[1].barh(y_pos + width/2, top_rep['rep_pct'].values, width, label='Top 10 Republican', color='red', edgecolor='black')
    axes[1].set_yticks(y_pos)
    axes[1].set_yticklabels([f"{d} / {r}" for d, r in zip(dem_states, rep_states)], fontsize=9)
    axes[1].set_xlabel('Party Donation Share (%)')
    axes[1].set_title('Most Partisan States (DEM left / REP right)', fontweight='bold')
    axes[1].legend(loc='lower right')
    axes[1].grid(True, alpha=0.3, axis='x')
    
    plt.tight_layout()
    plt.savefig(os.path.join(OUTPUT_DIR, 'party_by_state.png'), dpi=300, bbox_inches='tight')
    print("Saved: party_by_state.png")
    plt.close()


def create_donor_retention():
    """Create donor retention analysis."""
    print("Creating donor retention analysis...")
    
    query = """
    SELECT total_donations, DATEDIFF(last_donation_date, first_donation_date) as donor_tenure_days,
           total_amount, primary_party
    FROM contributors
    WHERE total_amount > 0 AND first_donation_date IS NOT NULL AND last_donation_date IS NOT NULL
    """
    
    df = execute_query(query, use_dict=False)
    for col in ['total_donations', 'donor_tenure_days', 'total_amount']:
        df[col] = pd.to_numeric(df[col])
    
    df['donor_type'] = df['total_donations'].apply(lambda x: 'One-Time' if x == 1 else ('Repeat (2-5)' if x <= 5 else 'Loyal (6+)'))
    
    fig, axes = plt.subplots(2, 2, figsize=(16, 12))
    fig.suptitle('Donor Retention Analysis', fontsize=16, fontweight='bold')
    
    type_counts = df['donor_type'].value_counts()
    type_order = ['One-Time', 'Repeat (2-5)', 'Loyal (6+)']
    type_counts = type_counts.reindex(type_order)
    
    axes[0, 0].pie(type_counts.values, labels=type_counts.index, autopct='%1.1f%%',
                   colors=['lightcoral', 'lightskyblue', 'lightgreen'], startangle=90)
    axes[0, 0].set_title('Donor Type Distribution', fontweight='bold')
    
    ltv = df.groupby('donor_type')['total_amount'].mean().reindex(type_order)
    axes[0, 1].bar(range(len(type_order)), ltv, color=['lightcoral', 'lightskyblue', 'lightgreen'], edgecolor='black')
    axes[0, 1].set_xticks(range(len(type_order)))
    axes[0, 1].set_xticklabels(type_order)
    axes[0, 1].set_ylabel('Average Lifetime Value ($)')
    axes[0, 1].set_title('Average Lifetime Value by Donor Type', fontweight='bold')
    axes[0, 1].grid(True, alpha=0.3, axis='y')
    for i, v in enumerate(ltv):
        axes[0, 1].text(i, v, f'${v:,.0f}', ha='center', va='bottom')
    
    sample = df.sample(min(50000, len(df)))
    axes[1, 0].scatter(sample['donor_tenure_days'] / 365, sample['total_donations'],
                       alpha=0.1, s=1, color='steelblue', rasterized=True)
    axes[1, 0].set_xlabel('Donor Tenure (years)')
    axes[1, 0].set_ylabel('Number of Donations')
    axes[1, 0].set_yscale('log')
    axes[1, 0].set_title('Tenure vs Donation Frequency', fontweight='bold')
    axes[1, 0].grid(True, alpha=0.3)
    
    total_by_type = df.groupby('donor_type')['total_amount'].sum().reindex(type_order)
    axes[1, 1].bar(range(len(type_order)), total_by_type / 1e9, 
                   color=['lightcoral', 'lightskyblue', 'lightgreen'], edgecolor='black')
    axes[1, 1].set_xticks(range(len(type_order)))
    axes[1, 1].set_xticklabels(type_order)
    axes[1, 1].set_ylabel('Total Donated (billions $)')
    axes[1, 1].set_title('Total Contribution by Donor Type', fontweight='bold')
    axes[1, 1].grid(True, alpha=0.3, axis='y')
    for i, v in enumerate(total_by_type):
        axes[1, 1].text(i, v/1e9, f'${v/1e9:.1f}B', ha='center', va='bottom')
    
    plt.tight_layout()
    plt.savefig(os.path.join(OUTPUT_DIR, 'donor_retention.png'), dpi=300, bbox_inches='tight')
    print("Saved: donor_retention.png")
    plt.close()


def create_frequency_segmentation_viz():
    """Create donor frequency segmentation visualization."""
    print("Creating donor frequency segmentation...")
    
    query = """
    SELECT 
        CASE 
            WHEN total_donations = 1 THEN '1x donors'
            WHEN total_donations = 2 THEN '2x donors'
            WHEN total_donations BETWEEN 3 AND 5 THEN '3-5x donors'
            WHEN total_donations BETWEEN 6 AND 12 THEN '6-12x donors'
            ELSE '12+x donors'
        END as frequency_category,
        COUNT(*) as num_donors,
        AVG(total_amount) as avg_total_amount
    FROM contributors WHERE total_amount > 0
    GROUP BY frequency_category
    """
    
    results = execute_query(query)
    freq_stats = pd.DataFrame(results)
    freq_order = ['1x donors', '2x donors', '3-5x donors', '6-12x donors', '12+x donors']
    freq_stats = freq_stats.set_index('frequency_category').reindex(freq_order)
    
    fig, axes = plt.subplots(1, 2, figsize=(16, 6))
    fig.suptitle('Donor Frequency Analysis', fontsize=16, fontweight='bold')
    
    bars1 = axes[0].bar(range(len(freq_order)), freq_stats['num_donors'] / 1e6, color='steelblue', edgecolor='black')
    axes[0].set_xticks(range(len(freq_order)))
    axes[0].set_xticklabels(freq_order)
    axes[0].set_ylabel('Number of Donors (millions)')
    axes[0].set_title('Number of Donors by Donation Frequency', fontweight='bold')
    axes[0].grid(True, alpha=0.3, axis='y')
    for bar in bars1:
        axes[0].text(bar.get_x() + bar.get_width()/2., bar.get_height(), f'{bar.get_height():.1f}M', ha='center', va='bottom')
    
    bars2 = axes[1].bar(range(len(freq_order)), freq_stats['avg_total_amount'], color='forestgreen', edgecolor='black')
    axes[1].set_xticks(range(len(freq_order)))
    axes[1].set_xticklabels(freq_order)
    axes[1].set_ylabel('Average Total Amount ($)')
    axes[1].set_title('Average Total Amount by Donation Frequency', fontweight='bold')
    axes[1].grid(True, alpha=0.3, axis='y')
    for bar in bars2:
        axes[1].text(bar.get_x() + bar.get_width()/2., bar.get_height(), f'${bar.get_height():,.0f}', ha='center', va='bottom')
    
    plt.tight_layout()
    plt.savefig(os.path.join(OUTPUT_DIR, 'donor_frequency_segmentation.png'), dpi=300, bbox_inches='tight')
    print("Saved: donor_frequency_segmentation.png")
    plt.close()


def create_geographic_analysis_viz():
    """Create geographic analysis visualization."""
    print("Creating geographic analysis...")
    
    base_query = """
    SELECT state, COUNT(*) as num_donors, SUM(total_amount) as total_donated
    FROM contributors
    WHERE total_amount > 0 AND state IS NOT NULL AND state != ''
    GROUP BY state ORDER BY {} DESC LIMIT 20
    """
    
    top_by_donors = pd.DataFrame(execute_query(base_query.format('num_donors'))).set_index('state')
    top_by_amount = pd.DataFrame(execute_query(base_query.format('total_donated'))).set_index('state')
    
    for df in [top_by_donors, top_by_amount]:
        df['num_donors'] = pd.to_numeric(df['num_donors'])
        df['total_donated'] = pd.to_numeric(df['total_donated'])
    
    fig, axes = plt.subplots(2, 1, figsize=(14, 12))
    fig.suptitle('Geographic Analysis - Top 20 States', fontsize=16, fontweight='bold')
    
    states_donors = top_by_donors.sort_values('num_donors')
    axes[0].barh(range(len(states_donors)), states_donors['num_donors'] / 1e6, color='steelblue', edgecolor='black')
    axes[0].set_yticks(range(len(states_donors)))
    axes[0].set_yticklabels(states_donors.index)
    axes[0].set_xlabel('Number of Donors (millions)')
    axes[0].set_title('Top 20 States by Number of Donors', fontweight='bold')
    axes[0].grid(True, alpha=0.3, axis='x')
    
    states_amount = top_by_amount.sort_values('total_donated')
    axes[1].barh(range(len(states_amount)), states_amount['total_donated'] / 1e3, color='forestgreen', edgecolor='black')
    axes[1].set_yticks(range(len(states_amount)))
    axes[1].set_yticklabels(states_amount.index)
    axes[1].set_xlabel('Total Amount Donated (thousands $)')
    axes[1].set_title('Top 20 States by Total Donation Amount', fontweight='bold')
    axes[1].grid(True, alpha=0.3, axis='x')
    
    plt.tight_layout()
    plt.savefig(os.path.join(OUTPUT_DIR, 'geographic_analysis.png'), dpi=300, bbox_inches='tight')
    print("Saved: geographic_analysis.png")
    plt.close()


def create_committee_analysis_viz():
    """Create committee fundraising analysis visualization."""
    print("Creating committee analysis visualizations...")
    
    query = """
    SELECT cm.cmte_id, cm.cmte_nm, cm.cmte_tp, cm.cmte_pty_affiliation,
           YEAR(cont.transaction_dt) as year,
           SUM(cont.transaction_amt) as total_raised,
           COUNT(*) as num_contributions,
           AVG(cont.transaction_amt) as avg_contribution
    FROM contributions cont
    JOIN committees cm ON cont.cmte_id = cm.cmte_id
    WHERE cont.transaction_dt IS NOT NULL AND cont.transaction_amt > 0
      AND YEAR(cont.transaction_dt) BETWEEN 1980 AND 2030
    GROUP BY cm.cmte_id, cm.cmte_nm, cm.cmte_tp, cm.cmte_pty_affiliation, YEAR(cont.transaction_dt)
    HAVING total_raised > 0
    """
    
    df = execute_query(query, use_dict=False)
    print(f"Loaded {len(df):,} committee-year records")
    
    for col in ['total_raised', 'num_contributions', 'avg_contribution', 'year']:
        df[col] = pd.to_numeric(df[col])
    df = df[(df['year'] >= 1980) & (df['year'] <= 2030)]
    
    df['party_label'] = df['cmte_pty_affiliation'].apply(get_party_label)
    df['color'] = df['party_label'].map(PARTY_COLORS)
    
    fig = plt.figure(figsize=(18, 16))
    gs = fig.add_gridspec(3, 2, hspace=0.35, wspace=0.25, height_ratios=[1.2, 1, 1])
    
    ax1 = fig.add_subplot(gs[0, :])
    top_cmtes = df.groupby(['cmte_id', 'cmte_nm', 'color', 'party_label']).agg({
        'total_raised': 'sum'
    }).reset_index().nlargest(15, 'total_raised').sort_values('total_raised')
    
    ax1.barh(range(len(top_cmtes)), top_cmtes['total_raised'] / 1e6, color=top_cmtes['color'], edgecolor='black', alpha=0.8)
    ax1.set_yticks(range(len(top_cmtes)))
    ax1.set_yticklabels([name[:50] for name in top_cmtes['cmte_nm']], fontsize=9)
    ax1.set_xlabel('Total Amount Raised (millions $)')
    ax1.set_title('Top 15 Committees by Total Fundraising', fontsize=14, fontweight='bold')
    ax1.grid(True, alpha=0.3, axis='x')
    ax1.legend(handles=[Patch(facecolor=c, edgecolor='black', label=l) for l, c in PARTY_COLORS.items()], loc='lower right')
    
    ax2 = fig.add_subplot(gs[1, :])
    party_yearly = df.groupby(['year', 'party_label'])['total_raised'].sum().reset_index()
    
    for party, color in PARTY_COLORS.items():
        data = party_yearly[party_yearly['party_label'] == party].sort_values('year')
        ax2.plot(data['year'], data['total_raised'] / 1e9, marker='o', linewidth=2, markersize=6, color=color, label=party, alpha=0.8)
    
    ax2.set_xlabel('Year')
    ax2.set_ylabel('Total Raised (billions $)')
    ax2.set_title('Party Fundraising Trends Over Time', fontsize=14, fontweight='bold')
    ax2.legend(loc='upper left')
    ax2.grid(True, alpha=0.3)
    
    years = sorted(party_yearly['year'].unique())
    tick_years = [y for y in years if y % 2 == 0] if len(years) > 20 else years
    ax2.set_xticks(tick_years)
    ax2.set_xticklabels([str(int(y)) for y in tick_years], rotation=45, ha='right')
    
    ax3 = fig.add_subplot(gs[2, 0])
    party_avg = df.groupby('party_label')['avg_contribution'].mean().sort_values(ascending=False)
    bars = ax3.bar(party_avg.index, party_avg.values, color=[PARTY_COLORS[p] for p in party_avg.index], edgecolor='black', alpha=0.8)
    ax3.set_ylabel('Average Contribution ($)')
    ax3.set_title('Average Donation Size by Party', fontsize=13, fontweight='bold')
    ax3.grid(True, alpha=0.3, axis='y')
    for bar in bars:
        ax3.text(bar.get_x() + bar.get_width()/2., bar.get_height(), f'${bar.get_height():,.0f}', ha='center', va='bottom', fontweight='bold')
    
    ax4 = fig.add_subplot(gs[2, 1])
    type_map = {'H': 'House', 'S': 'Senate', 'P': 'Presidential', 'N': 'PAC - Non-Qualified',
                'Q': 'PAC - Qualified', 'O': 'Super PAC', 'U': 'Single-Issue',
                'X': 'Party - Non-Qualified', 'Y': 'Party - Qualified', 'Z': 'National Party'}
    
    df['cmte_type_label'] = df['cmte_tp'].map(type_map).fillna('Other')
    type_totals = df.groupby('cmte_type_label')['total_raised'].sum().nlargest(8).sort_values()
    
    ax4.barh(range(len(type_totals)), type_totals.values / 1e9, color='steelblue', edgecolor='black', alpha=0.8)
    ax4.set_yticks(range(len(type_totals)))
    ax4.set_yticklabels(type_totals.index)
    ax4.set_xlabel('Total Raised (billions $)')
    ax4.set_title('Total Fundraising by Committee Type', fontsize=13, fontweight='bold')
    ax4.grid(True, alpha=0.3, axis='x')
    
    plt.suptitle('Committee Fundraising Analysis', fontsize=18, fontweight='bold', y=0.98)
    plt.savefig(os.path.join(OUTPUT_DIR, 'committee_analysis.png'), dpi=300, bbox_inches='tight')
    print("Saved: committee_analysis.png")
    plt.close()


def print_summary_statistics(stats):
    """Print summary statistics."""
    print("\nSUMMARY STATISTICS")
    print(f"  Total Contributors: {stats['total_contributors']:,}")
    print(f"  Total Amount Donated: ${stats['total_donated']:,.2f}")
    print(f"  Average Total per Contributor: ${stats['avg_total_per_contributor']:,.2f}")
    print(f"  Total Transactions: {stats['total_transactions']:,}")
    print(f"  Average Donations per Contributor: {stats['avg_donations_per_contributor']:.2f}")
    print(f"  Average Donation Amount: ${stats['avg_donation_amount']:,.2f}")


def main():
    """Main execution function."""
    print("FEC CAMPAIGN FINANCE - EXPLORATORY DATA ANALYSIS\n")
    
    stats = get_summary_statistics()
    print_summary_statistics(stats)
    
    df = load_distribution_data()
    
    print("\nGenerating visualizations...")
    
    create_donation_amount_distribution(df)
    create_donations_per_donor(df)
    create_avg_donation_distribution(df)
    create_donations_vs_amount_scatter(df)
    create_frequency_segmentation_viz()
    create_geographic_analysis_viz()
    create_committee_analysis_viz()
    create_annual_trends()
    create_election_cycle_analysis()
    create_occupation_employer_analysis()
    create_donor_size_segments()
    create_party_by_state()
    create_donor_retention()
    
    print("\n" + "=" * 60)
    print(f"EDA COMPLETE! All visualizations saved to: {OUTPUT_DIR}/")
    print("=" * 60)


if __name__ == "__main__":
    main()
