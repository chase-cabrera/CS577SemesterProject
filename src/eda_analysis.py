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


def execute_query(query, use_dict=True):
    """Execute SQL query and return results"""
    with db_connection(use_dict_cursor=use_dict) as conn:
        if use_dict:
            cursor = conn.cursor()
            cursor.execute(query)
            return cursor.fetchall()
        return pd.read_sql(query, conn)


def get_summary_statistics():
    """Get summary statistics directly from SQL"""
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
    results = execute_query(query)
    return results[0]


def load_distribution_data():
    """Load contributor data for distribution visualizations"""
    print("Loading contributor data for distributions...")
    query = """
    SELECT total_donations, total_amount, avg_donation_amount
    FROM contributors WHERE total_amount > 0
    """
    df = execute_query(query, use_dict=False)
    print(f"Loaded {len(df):,} contributors")
    return df


def create_donation_distribution_viz(df):
    """Create 2x2 panel donation distribution visualization"""
    print("Creating donation distribution visualizations...")
    
    total_amount = np.array(df['total_amount'], dtype=float)
    total_donations = np.array(df['total_donations'], dtype=float)
    avg_donation_amount = np.array(df['avg_donation_amount'], dtype=float)
    
    fig, axes = plt.subplots(2, 2, figsize=(16, 12))
    fig.suptitle('Donation Distribution Analysis', fontsize=16, fontweight='bold')
    
    # Distribution of Total Donation Amount (log scale)
    log_amounts = np.log10(np.clip(total_amount, 1, None))
    axes[0, 0].hist(log_amounts, bins=50, color='steelblue', edgecolor='black', alpha=0.7)
    axes[0, 0].set_xlabel('Log10(Total Amount)')
    axes[0, 0].set_ylabel('Frequency')
    axes[0, 0].set_title('Distribution of Total Donation Amount (log scale)', fontweight='bold')
    axes[0, 0].grid(True, alpha=0.3)
    
    # Number of Donations per Donor
    bins = min(int(np.max(total_donations)), 200)
    axes[0, 1].hist(total_donations, bins=bins, color='steelblue', edgecolor='black', alpha=0.7)
    axes[0, 1].set_xlabel('Number of Donations')
    axes[0, 1].set_ylabel('Frequency (log scale)')
    axes[0, 1].set_yscale('log')
    axes[0, 1].set_title('Number of Donations per Donor', fontweight='bold')
    axes[0, 1].grid(True, alpha=0.3)
    
    # Average Donation Amount per Donor (log scale)
    log_avg = np.log10(np.clip(avg_donation_amount, 1, None))
    axes[1, 0].hist(log_avg, bins=50, color='steelblue', edgecolor='black', alpha=0.7)
    axes[1, 0].set_xlabel('Log10(Avg Amount)')
    axes[1, 0].set_ylabel('Frequency')
    axes[1, 0].set_title('Average Donation Amount per Donor (log scale)', fontweight='bold')
    axes[1, 0].grid(True, alpha=0.3)
    
    # Donations Count vs Total Amount scatter
    axes[1, 1].scatter(total_donations, np.log10(np.clip(total_amount, 1, None)),
                       alpha=0.1, s=0.5, color='steelblue', rasterized=True)
    axes[1, 1].set_xlabel('Number of Donations (log scale)')
    axes[1, 1].set_ylabel('Log10(Total Amount)')
    axes[1, 1].set_xscale('log')
    axes[1, 1].set_title(f'Donations Count vs Total Amount ({len(df):,} points)', fontweight='bold')
    axes[1, 1].grid(True, alpha=0.3, which='both')
    
    plt.tight_layout()
    plt.savefig(os.path.join(OUTPUT_DIR, 'donation_distributions.png'), dpi=300, bbox_inches='tight')
    print(f"Saved: donation_distributions.png")
    plt.close()


def create_frequency_segmentation_viz():
    """Create donor frequency segmentation visualization"""
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
    
    # Number of Donors by Frequency
    bars1 = axes[0].bar(range(len(freq_order)), freq_stats['num_donors'] / 1e6, 
                        color='steelblue', edgecolor='black')
    axes[0].set_xticks(range(len(freq_order)))
    axes[0].set_xticklabels(freq_order)
    axes[0].set_ylabel('Number of Donors (millions)')
    axes[0].set_title('Number of Donors by Donation Frequency', fontweight='bold')
    axes[0].grid(True, alpha=0.3, axis='y')
    for bar in bars1:
        axes[0].text(bar.get_x() + bar.get_width()/2., bar.get_height(),
                     f'{bar.get_height():.1f}M', ha='center', va='bottom')
    
    # Average Total Amount by Frequency
    bars2 = axes[1].bar(range(len(freq_order)), freq_stats['avg_total_amount'],
                        color='forestgreen', edgecolor='black')
    axes[1].set_xticks(range(len(freq_order)))
    axes[1].set_xticklabels(freq_order)
    axes[1].set_ylabel('Average Total Amount ($)')
    axes[1].set_title('Average Total Amount by Donation Frequency', fontweight='bold')
    axes[1].grid(True, alpha=0.3, axis='y')
    for bar in bars2:
        axes[1].text(bar.get_x() + bar.get_width()/2., bar.get_height(),
                     f'${bar.get_height():,.0f}', ha='center', va='bottom')
    
    plt.tight_layout()
    plt.savefig(os.path.join(OUTPUT_DIR, 'donor_frequency_segmentation.png'), dpi=300, bbox_inches='tight')
    print(f"Saved: donor_frequency_segmentation.png")
    plt.close()


def create_geographic_analysis_viz():
    """Create geographic analysis visualization"""
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
    
    # By Number of Donors
    states_donors = top_by_donors.sort_values('num_donors')
    axes[0].barh(range(len(states_donors)), states_donors['num_donors'] / 1e6,
                 color='steelblue', edgecolor='black')
    axes[0].set_yticks(range(len(states_donors)))
    axes[0].set_yticklabels(states_donors.index)
    axes[0].set_xlabel('Number of Donors (millions)')
    axes[0].set_title('Top 20 States by Number of Donors', fontweight='bold')
    axes[0].grid(True, alpha=0.3, axis='x')
    
    # By Total Amount
    states_amount = top_by_amount.sort_values('total_donated')
    axes[1].barh(range(len(states_amount)), states_amount['total_donated'] / 1e3,
                 color='forestgreen', edgecolor='black')
    axes[1].set_yticks(range(len(states_amount)))
    axes[1].set_yticklabels(states_amount.index)
    axes[1].set_xlabel('Total Amount Donated (thousands $)')
    axes[1].set_title('Top 20 States by Total Donation Amount', fontweight='bold')
    axes[1].grid(True, alpha=0.3, axis='x')
    
    plt.tight_layout()
    plt.savefig(os.path.join(OUTPUT_DIR, 'geographic_analysis.png'), dpi=300, bbox_inches='tight')
    print(f"Saved: geographic_analysis.png")
    plt.close()


def create_committee_analysis_viz():
    """Create committee fundraising analysis visualization"""
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
    
    def get_party_label(party):
        if pd.notna(party) and str(party).upper() in ['DEM', 'D']:
            return 'Democrat'
        elif pd.notna(party) and str(party).upper() in ['REP', 'R']:
            return 'Republican'
        return 'Other/Independent'
    
    df['party_label'] = df['cmte_pty_affiliation'].apply(get_party_label)
    df['color'] = df['party_label'].map(PARTY_COLORS)
    
    fig = plt.figure(figsize=(18, 16))
    gs = fig.add_gridspec(3, 2, hspace=0.35, wspace=0.25, height_ratios=[1.2, 1, 1])
    
    # Top 15 Committees
    ax1 = fig.add_subplot(gs[0, :])
    top_cmtes = df.groupby(['cmte_id', 'cmte_nm', 'color', 'party_label']).agg({
        'total_raised': 'sum'
    }).reset_index().nlargest(15, 'total_raised').sort_values('total_raised')
    
    ax1.barh(range(len(top_cmtes)), top_cmtes['total_raised'] / 1e6,
             color=top_cmtes['color'], edgecolor='black', alpha=0.8)
    ax1.set_yticks(range(len(top_cmtes)))
    ax1.set_yticklabels([name[:50] for name in top_cmtes['cmte_nm']], fontsize=9)
    ax1.set_xlabel('Total Amount Raised (millions $)')
    ax1.set_title('Top 15 Committees by Total Fundraising', fontsize=14, fontweight='bold')
    ax1.grid(True, alpha=0.3, axis='x')
    ax1.legend(handles=[Patch(facecolor=c, edgecolor='black', label=l) 
                        for l, c in PARTY_COLORS.items()], loc='lower right')
    
    # Party Fundraising Trends
    ax2 = fig.add_subplot(gs[1, :])
    party_yearly = df.groupby(['year', 'party_label'])['total_raised'].sum().reset_index()
    
    for party, color in PARTY_COLORS.items():
        data = party_yearly[party_yearly['party_label'] == party].sort_values('year')
        ax2.plot(data['year'], data['total_raised'] / 1e9, marker='o', linewidth=2,
                 markersize=6, color=color, label=party, alpha=0.8)
    
    ax2.set_xlabel('Year')
    ax2.set_ylabel('Total Raised (billions $)')
    ax2.set_title('Party Fundraising Trends Over Time', fontsize=14, fontweight='bold')
    ax2.legend(loc='upper left')
    ax2.grid(True, alpha=0.3)
    
    years = sorted(party_yearly['year'].unique())
    tick_years = [y for y in years if y % 2 == 0] if len(years) > 20 else years
    ax2.set_xticks(tick_years)
    ax2.set_xticklabels([str(int(y)) for y in tick_years], rotation=45, ha='right')
    
    # Average Donation by Party
    ax3 = fig.add_subplot(gs[2, 0])
    party_avg = df.groupby('party_label')['avg_contribution'].mean().sort_values(ascending=False)
    bars = ax3.bar(party_avg.index, party_avg.values,
                   color=[PARTY_COLORS[p] for p in party_avg.index], edgecolor='black', alpha=0.8)
    ax3.set_ylabel('Average Contribution ($)')
    ax3.set_title('Average Donation Size by Party', fontsize=13, fontweight='bold')
    ax3.grid(True, alpha=0.3, axis='y')
    for bar in bars:
        ax3.text(bar.get_x() + bar.get_width()/2., bar.get_height(),
                 f'${bar.get_height():,.0f}', ha='center', va='bottom', fontweight='bold')
    
    # Committee Type Distribution
    ax4 = fig.add_subplot(gs[2, 1])
    type_map = {'H': 'House', 'S': 'Senate', 'P': 'Presidential', 'N': 'PAC - Non-Qualified',
                'Q': 'PAC - Qualified', 'O': 'Super PAC', 'U': 'Single-Issue',
                'X': 'Party - Non-Qualified', 'Y': 'Party - Qualified', 'Z': 'National Party'}
    
    df['cmte_type_label'] = df['cmte_tp'].map(type_map).fillna('Other')
    type_totals = df.groupby('cmte_type_label')['total_raised'].sum().nlargest(8).sort_values()
    
    ax4.barh(range(len(type_totals)), type_totals.values / 1e9,
             color='steelblue', edgecolor='black', alpha=0.8)
    ax4.set_yticks(range(len(type_totals)))
    ax4.set_yticklabels(type_totals.index)
    ax4.set_xlabel('Total Raised (billions $)')
    ax4.set_title('Total Fundraising by Committee Type', fontsize=13, fontweight='bold')
    ax4.grid(True, alpha=0.3, axis='x')
    
    plt.suptitle('Committee Fundraising Analysis', fontsize=18, fontweight='bold', y=0.98)
    plt.savefig(os.path.join(OUTPUT_DIR, 'committee_analysis.png'), dpi=300, bbox_inches='tight')
    print(f"Saved: committee_analysis.png")
    plt.close()


def print_summary_statistics(stats):
    """Print summary statistics"""
    print("\nSUMMARY STATISTICS")
    print(f"Total Contributors: {stats['total_contributors']:,}")
    print(f"Total Amount Donated: ${stats['total_donated']:,.2f}")
    print(f"Average Total per Contributor: ${stats['avg_total_per_contributor']:,.2f}")
    print(f"Total Transactions: {stats['total_transactions']:,}")
    print(f"Average Donations per Contributor: {stats['avg_donations_per_contributor']:.2f}")
    print(f"Average Donation Amount: ${stats['avg_donation_amount']:,.2f}")


def main():
    """Main execution function"""
    print("FEC CAMPAIGN FINANCE - EXPLORATORY DATA ANALYSIS\n")
    
    stats = get_summary_statistics()
    print_summary_statistics(stats)
    
    df = load_distribution_data()
    
    print("\nGENERATING VISUALIZATIONS")
    create_donation_distribution_viz(df)
    create_frequency_segmentation_viz()
    create_geographic_analysis_viz()
    create_committee_analysis_viz()
    
    print(f"\nEDA COMPLETE! All visualizations saved to: {OUTPUT_DIR}/")


if __name__ == "__main__":
    main()
