"""
Cluster Profiling
Analyze and profile each donor cluster to understand donor archetypes
"""

import pandas as pd
import numpy as np
import pickle
import os
import sys
import json
import matplotlib.pyplot as plt
import seaborn as sns

# Add parent directory to path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from feature_engineering import PARTIES


def load_clustered_data():
    """
    Load data with cluster assignments
    """
    print("Loading clustered data...")
    
    # Determine base directory (project root)
    script_dir = os.path.dirname(os.path.abspath(__file__))
    base_dir = os.path.dirname(os.path.dirname(script_dir))  # Go up to project root
    
    # Load cluster assignments
    cluster_path = os.path.join(base_dir, 'models', 'clustering', 'cluster_assignments', 'train_clusters.csv')
    train_clusters = pd.read_csv(cluster_path)
    
    # Load original data
    data_path = os.path.join(base_dir, 'models', 'data_splits', 'train.parquet')
    train_df = pd.read_parquet(data_path)
    
    # Merge
    train_df = train_df.merge(train_clusters, on='id', how='left')
    
    print(f"Loaded {len(train_df):,} contributors with cluster assignments")
    
    return train_df


def profile_cluster(cluster_id, cluster_data):
    """
    Create comprehensive profile for a single cluster
    
    Args:
        cluster_id: Cluster ID
        cluster_data: DataFrame with contributors in this cluster
        
    Returns:
        Dictionary with cluster profile
    """
    profile = {
        'cluster_id': int(cluster_id),
        'size': len(cluster_data),
        'demographics': {},
        'donation_behavior': {},
        'party_affiliation': {},
        'geographic': {}
    }
    
    # Demographics
    mode_emp = cluster_data['employer'].mode()
    profile['demographics']['avg_employer'] = mode_emp.iloc[0] if len(mode_emp) > 0 else 'UNKNOWN'
    mode_occ = cluster_data['occupation'].mode()
    profile['demographics']['avg_occupation'] = mode_occ.iloc[0] if len(mode_occ) > 0 else 'UNKNOWN'
    profile['demographics']['top_occupations'] = cluster_data['occupation'].value_counts().head(5).to_dict()
    
    # Donation behavior
    profile['donation_behavior']['avg_total_donations'] = float(cluster_data['total_donations'].mean())
    profile['donation_behavior']['avg_total_amount'] = float(cluster_data['total_amount'].mean())
    profile['donation_behavior']['avg_donation_amount'] = float(cluster_data['avg_donation_amount'].mean())
    profile['donation_behavior']['avg_recency_days'] = float(cluster_data['recency_days'].mean())
    profile['donation_behavior']['avg_unique_committees'] = float(cluster_data['unique_committees'].mean())
    
    # Party affiliation
    profile['party_affiliation']['primary_party_distribution'] = cluster_data['primary_party'].value_counts().to_dict()
    mode_party = cluster_data['primary_party'].mode()
    profile['party_affiliation']['dominant_party'] = mode_party.iloc[0] if len(mode_party) > 0 else 'UNKNOWN'
    profile['party_affiliation']['avg_dem_pct'] = float(cluster_data['dem_pct'].mean())
    profile['party_affiliation']['avg_rep_pct'] = float(cluster_data['rep_pct'].mean())
    
    # Party-specific amounts
    for party in PARTIES:
        party_col = f'{party.lower()}_amount'
        if party_col in cluster_data.columns:
            profile['party_affiliation'][f'avg_{party.lower()}_amount'] = float(cluster_data[party_col].mean())
    
    # Geographic
    profile['geographic']['top_states'] = cluster_data['state'].value_counts().head(10).to_dict()
    mode_state = cluster_data['state'].mode()
    profile['geographic']['dominant_state'] = mode_state.iloc[0] if len(mode_state) > 0 else 'UNKNOWN'
    
    return profile


def profile_all_clusters(df):
    """
    Profile all clusters
    
    Args:
        df: DataFrame with cluster assignments
        
    Returns:
        Dictionary mapping cluster_id -> profile
    """
    print("\nPROFILING ALL CLUSTERS")
    
    cluster_profiles = {}
    
    unique_clusters = sorted(df['cluster'].unique())
    
    for cluster_id in unique_clusters:
        cluster_data = df[df['cluster'] == cluster_id]
        profile = profile_cluster(cluster_id, cluster_data)
        cluster_profiles[cluster_id] = profile
        
        if cluster_id % 50 == 0:  # Print progress
            print(f"Profiled cluster {cluster_id}...")
    
    print(f"\nProfiled {len(cluster_profiles)} clusters")
    
    return cluster_profiles


def print_cluster_summary(cluster_profiles):
    """
    Print summary of all clusters
    """
    print("\nCLUSTER SUMMARY")
    
    # Create summary DataFrame
    summary_data = []
    for cluster_id, profile in cluster_profiles.items():
        summary_data.append({
            'Cluster': cluster_id,
            'Size': profile['size'],
            'Dominant Party': profile['party_affiliation']['dominant_party'],
            'Avg Donations': f"{profile['donation_behavior']['avg_total_donations']:.1f}",
            'Avg Amount': f"${profile['donation_behavior']['avg_total_amount']:.0f}",
            'Top Occupation': profile['demographics']['avg_occupation'][:20],
            'Top State': profile['geographic']['dominant_state']
        })
    
    summary_df = pd.DataFrame(summary_data)
    
    # Print first 20 clusters
    print("\nFirst 20 clusters:")
    print(summary_df.head(20).to_string(index=False))
    
    print(f"\n... and {len(summary_df) - 20} more clusters")
    
    # Overall statistics
    print("\nOVERALL STATISTICS")
    
    total_contributors = sum(p['size'] for p in cluster_profiles.values())
    print(f"Total contributors: {total_contributors:,}")
    print(f"Number of clusters: {len(cluster_profiles)}")
    print(f"Average cluster size: {total_contributors / len(cluster_profiles):.0f}")
    
    # Party distribution across clusters
    print("\nDominant party distribution across clusters:")
    dominant_parties = [p['party_affiliation']['dominant_party'] for p in cluster_profiles.values()]
    party_counts = pd.Series(dominant_parties).value_counts()
    for party, count in party_counts.items():
        print(f"  {party}: {count} clusters ({count/len(cluster_profiles)*100:.1f}%)")
    
    return summary_df


def create_cluster_visualization(cluster_profiles, summary_df, save_dir=None):
    if save_dir is None:
        script_dir = os.path.dirname(os.path.abspath(__file__))
        base_dir = os.path.dirname(os.path.dirname(script_dir))
        save_dir = os.path.join(base_dir, 'report', 'model_comparison')
    """
    Create visualizations of cluster characteristics
    """
    print("\nCreating cluster visualizations...")
    
    os.makedirs(save_dir, exist_ok=True)
    
    # Prepare data for visualization
    cluster_ids = []
    sizes = []
    avg_amounts = []
    dominant_parties = []
    
    for cluster_id, profile in sorted(cluster_profiles.items()):
        cluster_ids.append(cluster_id)
        sizes.append(profile['size'])
        avg_amounts.append(profile['donation_behavior']['avg_total_amount'])
        dominant_parties.append(profile['party_affiliation']['dominant_party'])
    
    # Create figure with multiple subplots
    fig, axes = plt.subplots(2, 2, figsize=(16, 12))
    fig.suptitle('Cluster Analysis Overview', fontsize=16, fontweight='bold')
    
    # 1. Cluster sizes
    ax1 = axes[0, 0]
    ax1.bar(range(len(sizes)), sizes, color='steelblue', alpha=0.7)
    ax1.set_xlabel('Cluster ID', fontsize=12)
    ax1.set_ylabel('Number of Contributors', fontsize=12)
    ax1.set_title('Cluster Sizes', fontsize=13, fontweight='bold')
    ax1.grid(True, alpha=0.3, axis='y')
    
    # 2. Average donation amounts by cluster
    ax2 = axes[0, 1]
    colors = ['blue' if p == 'DEM' else 'red' if p == 'REP' else 'gray' for p in dominant_parties]
    ax2.bar(range(len(avg_amounts)), avg_amounts, color=colors, alpha=0.7)
    ax2.set_xlabel('Cluster ID', fontsize=12)
    ax2.set_ylabel('Average Total Amount ($)', fontsize=12)
    ax2.set_title('Average Donation Amounts by Cluster', fontsize=13, fontweight='bold')
    ax2.grid(True, alpha=0.3, axis='y')
    
    # 3. Party distribution across clusters
    ax3 = axes[1, 0]
    party_counts = pd.Series(dominant_parties).value_counts()
    ax3.pie(party_counts.values, labels=party_counts.index, autopct='%1.1f%%', startangle=90)
    ax3.set_title('Dominant Party Distribution', fontsize=13, fontweight='bold')
    
    # 4. Cluster size distribution
    ax4 = axes[1, 1]
    ax4.hist(sizes, bins=50, color='forestgreen', alpha=0.7, edgecolor='black')
    ax4.set_xlabel('Cluster Size', fontsize=12)
    ax4.set_ylabel('Number of Clusters', fontsize=12)
    ax4.set_title('Distribution of Cluster Sizes', fontsize=13, fontweight='bold')
    ax4.grid(True, alpha=0.3, axis='y')
    
    plt.tight_layout()
    
    save_path = os.path.join(save_dir, 'cluster_profiles.png')
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    print(f"Cluster visualizations saved to: {save_path}")
    plt.close()


def identify_interesting_clusters(cluster_profiles, top_n=10):
    """
    Identify most interesting/distinctive clusters
    """
    print("\nMOST INTERESTING CLUSTERS")
    
    # Sort by different criteria
    print("\nLargest clusters:")
    sorted_by_size = sorted(cluster_profiles.items(), key=lambda x: x[1]['size'], reverse=True)
    for i, (cluster_id, profile) in enumerate(sorted_by_size[:top_n]):
        print(f"  {i+1}. Cluster {cluster_id}: {profile['size']:,} contributors, "
              f"{profile['party_affiliation']['dominant_party']} lean, "
              f"${profile['donation_behavior']['avg_total_amount']:.0f} avg")
    
    print("\nHighest average donors:")
    sorted_by_amount = sorted(cluster_profiles.items(), 
                              key=lambda x: x[1]['donation_behavior']['avg_total_amount'], 
                              reverse=True)
    for i, (cluster_id, profile) in enumerate(sorted_by_amount[:top_n]):
        print(f"  {i+1}. Cluster {cluster_id}: ${profile['donation_behavior']['avg_total_amount']:.0f} avg, "
              f"{profile['size']:,} contributors, "
              f"{profile['party_affiliation']['dominant_party']} lean")
    
    print("\nMost frequent donors:")
    sorted_by_frequency = sorted(cluster_profiles.items(),
                                 key=lambda x: x[1]['donation_behavior']['avg_total_donations'],
                                 reverse=True)
    for i, (cluster_id, profile) in enumerate(sorted_by_frequency[:top_n]):
        print(f"  {i+1}. Cluster {cluster_id}: {profile['donation_behavior']['avg_total_donations']:.1f} donations avg, "
              f"{profile['size']:,} contributors, "
              f"{profile['party_affiliation']['dominant_party']} lean")


def save_cluster_profiles(cluster_profiles, summary_df):
    """
    Save cluster profiles to files
    """
    print("\nSaving cluster profiles...")
    
    # Determine save directory (absolute path)
    script_dir = os.path.dirname(os.path.abspath(__file__))
    base_dir = os.path.dirname(os.path.dirname(script_dir))
    save_dir = os.path.join(base_dir, 'models', 'clustering')
    os.makedirs(save_dir, exist_ok=True)
    
    # Save full profiles as JSON
    json_path = os.path.join(save_dir, 'cluster_profiles.json')
    with open(json_path, 'w') as f:
        # Convert numpy types to native Python types for JSON serialization
        profiles_serializable = {}
        for cluster_id, profile in cluster_profiles.items():
            profiles_serializable[str(cluster_id)] = profile
        json.dump(profiles_serializable, f, indent=2, default=str)
    
    print(f"Full profiles saved to: {json_path}")
    
    # Save summary as CSV
    csv_path = os.path.join(save_dir, 'cluster_summary.csv')
    summary_df.to_csv(csv_path, index=False)
    print(f"Summary saved to: {csv_path}")


def main():
    """
    Main cluster profiling script
    """
    print("CLUSTER PROFILING")
    
    # Load data
    df = load_clustered_data()
    
    # Profile all clusters
    cluster_profiles = profile_all_clusters(df)
    
    # Print summary
    summary_df = print_cluster_summary(cluster_profiles)
    
    # Identify interesting clusters
    identify_interesting_clusters(cluster_profiles)
    
    # Create visualizations
    create_cluster_visualization(cluster_profiles, summary_df)
    
    # Save profiles
    save_cluster_profiles(cluster_profiles, summary_df)
    
    print("\nCLUSTER PROFILING COMPLETE")


if __name__ == "__main__":
    main()
