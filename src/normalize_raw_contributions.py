"""
Normalizes raw FEC contributions into relational schema.
"""

import logging
import time
import argparse
from db_utils import get_db_connection

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(message)s',
    handlers=[
        logging.FileHandler('normalization.log', mode='a'),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

BATCH_SIZE = 2000000


def build_donor_key():
    """Unique identifier for donor matching."""
    return """
        CONCAT(
            UPPER(COALESCE(last_name, '')), '|',
            UPPER(COALESCE(first_name, '')), '|',
            UPPER(COALESCE(city, '')), '|',
            UPPER(COALESCE(state, '')), '|',
            LEFT(COALESCE(zip_code, ''), 5), '|',
            LEFT(UPPER(COALESCE(employer, '')), 20)
        )
    """


def insert_contributors(cursor, conn, source_file=None):
    """Insert unique donors from raw data in batches."""
    where_clause = "WHERE status = 0"
    params = []
    
    if source_file:
        where_clause += " AND source_file = %s"
        params.append(source_file)
    
    # Get ID range
    cursor.execute(f"SELECT MIN(id), MAX(id) FROM raw_contributions {where_clause}",
                   params if params else None)
    min_id, max_id = cursor.fetchone()
    
    if min_id is None:
        return 0
    
    total_inserted = 0
    current_id = min_id
    
    while current_id <= max_id:
        batch_end = current_id + BATCH_SIZE
        batch_where = f"WHERE status = 0 AND id >= {current_id} AND id < {batch_end}"
        if source_file:
            batch_where += f" AND source_file = %s"
        
        query = f"""
            INSERT IGNORE INTO contributors (
                donor_key, first_name, last_name, middle_name, name_suffix,
                city, state, zip_code, employer, occupation
            )
            SELECT DISTINCT
                {build_donor_key()} as donor_key,
                first_name, last_name, middle_name, name_suffix,
                city, state, zip_code, employer, occupation
            FROM raw_contributions
            {batch_where}
        """
        
        cursor.execute(query, (source_file,) if source_file else None)
        batch_inserted = cursor.rowcount
        total_inserted += batch_inserted
        conn.commit()
        
        if batch_inserted > 0:
            logger.info(f"Inserted {batch_inserted:,} contributors (total: {total_inserted:,})")
        
        current_id = batch_end
    
    return total_inserted


def insert_contributions(cursor, conn, source_file=None):
    """Link contributions to donors and insert."""
    where_clause = "WHERE r.status = 0"
    params = []
    
    if source_file:
        where_clause += " AND r.source_file = %s"
        params.append(source_file)
    
    donor_key = build_donor_key()
    for col in ['last_name', 'first_name', 'city', 'state', 'zip_code', 'employer']:
        donor_key = donor_key.replace(col, f'r.{col}')
    
    cursor.execute(f"SELECT MIN(id), MAX(id) FROM raw_contributions r {where_clause}", 
                   params if params else None)
    min_id, max_id = cursor.fetchone()
    
    if min_id is None:
        return 0
    
    total_inserted = 0
    current_id = min_id
    
    while current_id <= max_id:
        batch_end = current_id + BATCH_SIZE
        batch_where = f"WHERE r.status = 0 AND r.id >= {current_id} AND r.id < {batch_end}"
        if source_file:
            batch_where += " AND r.source_file = %s"
        
        query = f"""
            INSERT INTO contributions (
                contributor_id, cmte_id, transaction_pgi, transaction_tp,
                transaction_dt, transaction_amt, amndt_ind, rpt_tp, image_num,
                entity_tp, other_id, tran_id, file_num, sub_id, memo_cd, memo_text
            )
            SELECT c.id, r.cmte_id, r.transaction_pgi, r.transaction_tp, r.transaction_dt,
                   r.transaction_amt, r.amndt_ind, r.rpt_tp, r.image_num, r.entity_tp,
                   r.other_id, r.tran_id, r.file_num, r.sub_id, r.memo_cd, r.memo_text
            FROM raw_contributions r
            JOIN contributors c ON c.donor_key = {donor_key}
            {batch_where}
        """
        
        cursor.execute(query, (source_file,) if source_file else None)
        batch_inserted = cursor.rowcount
        total_inserted += batch_inserted
        
        cursor.execute(f"UPDATE raw_contributions r SET status = 1, processed_at = NOW() {batch_where}",
                      (source_file,) if source_file else None)
        conn.commit()
        
        if batch_inserted > 0:
            logger.info(f"Inserted {batch_inserted:,} (total: {total_inserted:,})")
        
        current_id = batch_end
    
    return total_inserted


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--source-file', type=str)
    parser.add_argument('--batch-size', type=int, default=2000000)
    args = parser.parse_args()
    
    global BATCH_SIZE
    BATCH_SIZE = args.batch_size
    
    conn = get_db_connection()
    cursor = conn.cursor()
    
    cursor.execute("SELECT COUNT(*) FROM raw_contributions WHERE status = 0")
    pending = cursor.fetchone()[0]
    logger.info(f"Records to process: {pending:,}")
    
    if pending == 0:
        return
    
    start_time = time.time()
    
    logger.info("Inserting contributors...")
    num_contributors = insert_contributors(cursor, conn, args.source_file)
    logger.info(f"Contributors: {num_contributors:,}")
    
    logger.info("Inserting contributions...")
    num_contributions = insert_contributions(cursor, conn, args.source_file)
    logger.info(f"Contributions: {num_contributions:,}")
    
    logger.info(f"Done in {(time.time() - start_time)/60:.1f} min")
    
    cursor.close()
    conn.close()


if __name__ == "__main__":
    main()
