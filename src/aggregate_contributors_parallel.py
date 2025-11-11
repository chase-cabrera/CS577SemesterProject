import logging
import time
from multiprocessing import Process, Value
import argparse
from db_utils import get_db_connection

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - Worker %(process)d - %(message)s'
)
logger = logging.getLogger(__name__)


def aggregate_worker(worker_id, start_id, end_id, batch_size, stop_flag, total_updated):
    """
    Worker process to aggregate a range of contributors
    """
    conn = get_db_connection()
    cursor = conn.cursor()
    
    logger.info(f"Starting aggregation from ID {start_id:,} to {end_id:,}")
    
    current_id = start_id
    worker_updated = 0
    
    query = """
        UPDATE contributors c
        JOIN (
            SELECT 
                cont.contributor_id,
                COUNT(*) as total_donations,
                SUM(cont.transaction_amt) as total_amount,
                MIN(cont.transaction_dt) as first_donation_date,
                MAX(cont.transaction_dt) as last_donation_date,
                AVG(cont.transaction_amt) as avg_donation_amount,
                COUNT(DISTINCT cont.cmte_id) as unique_committees,
                
                SUM(CASE WHEN ca.cand_pty_affiliation = 'DEM' THEN 1 ELSE 0 END) as dem_donations,
                SUM(CASE WHEN ca.cand_pty_affiliation = 'DEM' THEN cont.transaction_amt ELSE 0 END) as dem_amount,
                
                SUM(CASE WHEN ca.cand_pty_affiliation = 'REP' THEN 1 ELSE 0 END) as rep_donations,
                SUM(CASE WHEN ca.cand_pty_affiliation = 'REP' THEN cont.transaction_amt ELSE 0 END) as rep_amount,
                
                SUM(CASE WHEN ca.cand_pty_affiliation IN ('LIB', 'LBL', 'LBU', 'LP', 'LPF') THEN 1 ELSE 0 END) as lib_donations,
                SUM(CASE WHEN ca.cand_pty_affiliation IN ('LIB', 'LBL', 'LBU', 'LP', 'LPF') THEN cont.transaction_amt ELSE 0 END) as lib_amount,
                
                SUM(CASE WHEN ca.cand_pty_affiliation IN ('GRE', 'GRN', 'GLP', 'DGR') THEN 1 ELSE 0 END) as gre_donations,
                SUM(CASE WHEN ca.cand_pty_affiliation IN ('GRE', 'GRN', 'GLP', 'DGR') THEN cont.transaction_amt ELSE 0 END) as gre_amount,
                
                SUM(CASE WHEN ca.cand_pty_affiliation IN ('IND', 'IDP', 'ICD') THEN 1 ELSE 0 END) as ind_donations,
                SUM(CASE WHEN ca.cand_pty_affiliation IN ('IND', 'IDP', 'ICD') THEN cont.transaction_amt ELSE 0 END) as ind_amount,
                
                SUM(CASE WHEN ca.cand_pty_affiliation NOT IN ('DEM', 'REP', 'LIB', 'LBL', 'LBU', 'LP', 'LPF', 
                                                                'GRE', 'GRN', 'GLP', 'DGR', 'IND', 'IDP', 'ICD')
                              OR ca.cand_pty_affiliation IS NULL 
                         THEN 1 ELSE 0 END) as other_donations,
                SUM(CASE WHEN ca.cand_pty_affiliation NOT IN ('DEM', 'REP', 'LIB', 'LBL', 'LBU', 'LP', 'LPF',
                                                                'GRE', 'GRN', 'GLP', 'DGR', 'IND', 'IDP', 'ICD')
                              OR ca.cand_pty_affiliation IS NULL
                         THEN cont.transaction_amt ELSE 0 END) as other_amount
            FROM contributions cont
            LEFT JOIN committees cm ON cont.cmte_id = cm.cmte_id
            LEFT JOIN candidate_committee_links ccl ON cm.cmte_id = ccl.cmte_id
            LEFT JOIN candidates ca ON ccl.cand_id = ca.cand_id
            WHERE cont.contributor_id >= %s 
              AND cont.contributor_id < %s
            GROUP BY cont.contributor_id
        ) stats ON c.id = stats.contributor_id
        SET 
            c.total_donations = stats.total_donations,
            c.total_amount = stats.total_amount,
            c.first_donation_date = stats.first_donation_date,
            c.last_donation_date = stats.last_donation_date,
            c.avg_donation_amount = stats.avg_donation_amount,
            c.unique_committees = stats.unique_committees,
            c.recency_days = DATEDIFF(CURDATE(), stats.last_donation_date),
            c.dem_donations = stats.dem_donations,
            c.dem_amount = stats.dem_amount,
            c.rep_donations = stats.rep_donations,
            c.rep_amount = stats.rep_amount,
            c.lib_donations = stats.lib_donations,
            c.lib_amount = stats.lib_amount,
            c.gre_donations = stats.gre_donations,
            c.gre_amount = stats.gre_amount,
            c.ind_donations = stats.ind_donations,
            c.ind_amount = stats.ind_amount,
            c.other_donations = stats.other_donations,
            c.other_amount = stats.other_amount,
            c.dem_pct = ROUND(IFNULL(stats.dem_amount, 0) / NULLIF(stats.total_amount, 0) * 100, 2),
            c.rep_pct = ROUND(IFNULL(stats.rep_amount, 0) / NULLIF(stats.total_amount, 0) * 100, 2),
            c.primary_party = CASE 
                WHEN stats.dem_amount > stats.rep_amount THEN 'DEM'
                WHEN stats.rep_amount > stats.dem_amount THEN 'REP'
                ELSE 'OTHER'
            END
    """
    
    try:
        while current_id < end_id and not stop_flag.value:
            batch_end = min(current_id + batch_size, end_id)
            
            cursor.execute(query, (current_id, batch_end))
            rows_updated = cursor.rowcount
            conn.commit()
            
            worker_updated += rows_updated
            
            with total_updated.get_lock():
                total_updated.value += rows_updated
            
            if worker_updated % 1000 == 0 or rows_updated > 0:
                logger.info(f"Processed up to ID {batch_end:,} | Updated {worker_updated:,} (Total: {total_updated.value:,})")
            
            current_id = batch_end
            time.sleep(0.01)  # Brief pause
            
    except Exception as e:
        logger.error(f"Error: {e}")
        import traceback
        traceback.print_exc()
    finally:
        cursor.close()
        conn.close()
        logger.info(f"Completed range {start_id:,}-{end_id:,} | Updated {worker_updated:,} contributors")


def main():
    parser = argparse.ArgumentParser(description='Parallel contributor aggregation')
    parser.add_argument('--workers', type=int, default=4, help='Number of parallel workers')
    parser.add_argument('--batch-size', type=int, default=1000, help='Contributors per batch')
    args = parser.parse_args()
    
    print("PARALLEL CONTRIBUTOR AGGREGATION")
    print(f"Workers: {args.workers}")
    print(f"Batch size: {args.batch_size}")
    print()
    
    # Get max contributor ID
    conn = get_db_connection()
    cursor = conn.cursor()
    cursor.execute("SELECT MAX(id) FROM contributors")
    max_id = cursor.fetchone()[0]
    cursor.close()
    conn.close()
    
    print(f"Total contributors: {max_id:,}")
    print(f"Dividing work among {args.workers} workers...")
    print()
    
    # Divide work among workers
    range_size = max_id // args.workers
    
    stop_flag = Value('i', 0)
    total_updated = Value('i', 0)
    
    workers = []
    start_time = time.time()
    
    for i in range(args.workers):
        start_id = (i * range_size) + 1
        end_id = ((i + 1) * range_size) if i < args.workers - 1 else max_id + 1
        
        p = Process(target=aggregate_worker, args=(i+1, start_id, end_id, args.batch_size, stop_flag, total_updated))
        p.start()
        workers.append(p)
    
    
    try:
        # Wait for all workers to complete
        for p in workers:
            p.join()
    except KeyboardInterrupt:
        print("\n\nStopping workers...")
        stop_flag.value = 1
        for p in workers:
            p.join(timeout=5)
    
    
    print("AGGREGATION COMPLETE!")


if __name__ == "__main__":
    main()

