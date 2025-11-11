import sys
import os
from pathlib import Path
import time
from db_utils import get_db_connection

def import_committees(cursor, file_path):
    """Import committee master file"""
    print(f"\nImporting committees from {file_path.name}...")
    
    # Column mapping for committees from cm.txt
    insert_query = """
        INSERT INTO committees (
            cmte_id, cmte_nm, tres_nm, cmte_st1, cmte_st2,
            cmte_city, cmte_st, cmte_zip, cmte_dsgn, cmte_tp,
            cmte_pty_affiliation, cmte_filing_freq, org_tp,
            connected_org_nm, cand_id
        ) VALUES (
            %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s
        )
        ON DUPLICATE KEY UPDATE
            cmte_nm = VALUES(cmte_nm),
            tres_nm = VALUES(tres_nm),
            cmte_city = VALUES(cmte_city),
            cmte_st = VALUES(cmte_st),
            cmte_zip = VALUES(cmte_zip)
    """
    
    batch = []
    count = 0
    
    with open(file_path, 'r', encoding='latin-1') as f:
        for line in f:
            if not line.strip():
                continue
            
            fields = line.strip().split('|')
            
            # Map fields 15 committee master file fields
            record = [
                fields[0] if len(fields) > 0 else None,  # cmte_id
                fields[1] if len(fields) > 1 else None,  # cmte_nm
                fields[2] if len(fields) > 2 else None,  # tres_nm
                fields[3] if len(fields) > 3 else None,  # cmte_st1
                fields[4] if len(fields) > 4 else None,  # cmte_st2
                fields[5] if len(fields) > 5 else None,  # cmte_city
                fields[6] if len(fields) > 6 else None,  # cmte_st
                fields[7] if len(fields) > 7 else None,  # cmte_zip
                fields[8] if len(fields) > 8 else None,  # cmte_dsgn
                fields[9] if len(fields) > 9 else None,  # cmte_tp
                fields[10] if len(fields) > 10 else None,  # cmte_pty_affiliation
                fields[11] if len(fields) > 11 else None,  # cmte_filing_freq
                fields[12] if len(fields) > 12 else None,  # org_tp
                fields[13] if len(fields) > 13 else None,  # connected_org_nm
                fields[14] if len(fields) > 14 else None,  # cand_id
            ]
            
            batch.append(record)
            
            if len(batch) >= 1000:
                cursor.executemany(insert_query, batch)
                count += len(batch)
                batch = []
    
    if batch:
        cursor.executemany(insert_query, batch)
        count += len(batch)
    
    print(f"Imported {count:,} committees")
    return count


def import_candidates(cursor, file_path):
    """Import candidate master file"""
    print(f"\nImporting candidates from {file_path.name}...")
    
    # Column mapping for candidates from cn.txt
    insert_query = """
        INSERT INTO candidates (
            cand_id, cand_name, cand_pty_affiliation, cand_election_yr,
            cand_office_st, cand_office, cand_office_district, cand_ici,
            cand_status, cand_pcc, cand_st1, cand_st2, cand_city, cand_st, cand_zip
        ) VALUES (
            %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s
        )
        ON DUPLICATE KEY UPDATE
            cand_name = VALUES(cand_name),
            cand_pty_affiliation = VALUES(cand_pty_affiliation),
            cand_election_yr = VALUES(cand_election_yr),
            cand_office = VALUES(cand_office),
            cand_status = VALUES(cand_status)
    """
    
    batch = []
    count = 0
    
    with open(file_path, 'r', encoding='latin-1') as f:
        for line in f:
            if not line.strip():
                continue
            
            fields = line.strip().split('|')
            
            # Map fields 15 candidate master file fields
            record = [
                fields[0] if len(fields) > 0 else None,  # cand_id
                fields[1] if len(fields) > 1 else None,  # cand_name
                fields[2] if len(fields) > 2 else None,  # cand_pty_affiliation
                int(fields[3]) if len(fields) > 3 and fields[3] else None,  # cand_election_yr
                fields[4] if len(fields) > 4 else None,  # cand_office_st
                fields[5] if len(fields) > 5 else None,  # cand_office
                fields[6] if len(fields) > 6 else None,  # cand_office_district
                fields[7] if len(fields) > 7 else None,  # cand_ici
                fields[8] if len(fields) > 8 else None,  # cand_status
                fields[9] if len(fields) > 9 else None,  # cand_pcc
                fields[10] if len(fields) > 10 else None,  # cand_st1
                fields[11] if len(fields) > 11 else None,  # cand_st2
                fields[12] if len(fields) > 12 else None,  # cand_city
                fields[13] if len(fields) > 13 else None,  # cand_st
                fields[14] if len(fields) > 14 else None,  # cand_zip
            ]
            
            batch.append(record)
            
            if len(batch) >= 1000:
                cursor.executemany(insert_query, batch)
                count += len(batch)
                batch = []
    
    if batch:
        cursor.executemany(insert_query, batch)
        count += len(batch)
    
    print(f"Imported {count:,} candidates")
    return count


def import_linkages(cursor, file_path):
    """Import candidate-committee linkages"""
    print(f"\nImporting candidate-committee links from {file_path.name}...")
    
    # Column mapping for linkages from ccl.txt
    insert_query = """
        INSERT INTO candidate_committee_links (
            cand_id, cand_election_yr, fec_election_yr, cmte_id,
            cmte_tp, cmte_dsgn, linkage_id
        ) VALUES (
            %s, %s, %s, %s, %s, %s, %s
        )
        ON DUPLICATE KEY UPDATE
            cmte_tp = VALUES(cmte_tp),
            cmte_dsgn = VALUES(cmte_dsgn),
            fec_election_yr = VALUES(fec_election_yr)
    """
    
    batch = []
    count = 0
    
    with open(file_path, 'r', encoding='latin-1') as f:
        for line in f:
            if not line.strip():
                continue
            
            fields = line.strip().split('|')
            
            # Map fields (7 fields total)
            record = [
                fields[0] if len(fields) > 0 else None,  # cand_id
                int(fields[1]) if len(fields) > 1 and fields[1] else None,  # cand_election_yr
                int(fields[2]) if len(fields) > 2 and fields[2] else None,  # fec_election_yr
                fields[3] if len(fields) > 3 else None,  # cmte_id
                fields[4] if len(fields) > 4 else None,  # cmte_tp
                fields[5] if len(fields) > 5 else None,  # cmte_dsgn
                int(fields[6]) if len(fields) > 6 and fields[6] else None,  # linkage_id
            ]
            
            batch.append(record)
            
            if len(batch) >= 1000:
                cursor.executemany(insert_query, batch)
                count += len(batch)
                batch = []
    
    if batch:
        cursor.executemany(insert_query, batch)
        count += len(batch)
    
    print(f"Imported {count:,} linkages")
    return count


def main():
    print("CAMPAIGN FINANCE DATA IMPORT")
    print()
    
    conn = get_db_connection()
    cursor = conn.cursor()
    
    data_dir = Path(__file__).parent.parent / 'data'
    
    if not data_dir.exists():
        print(f"ERROR: Campaign data directory not found: {data_dir}")
        return
    
    start_time = time.time()
    
    # Import in order: candidates, committees, then linkages
    totals = {}
    
    # 1. Candidates
    cn_file = data_dir / 'cn.txt'
    if cn_file.exists():
        totals['candidates'] = import_candidates(cursor, cn_file)
        conn.commit()
    else:
        print(f"Candidate file not found: {cn_file}")
    
    # 2. Committees
    cm_file = data_dir / 'cm.txt'
    if cm_file.exists():
        totals['committees'] = import_committees(cursor, cm_file)
        conn.commit()
    else:
        print(f"Committee file not found: {cm_file}")
    
    # 3. Linkages
    ccl_file = data_dir / 'ccl.txt'
    if ccl_file.exists():
        totals['linkages'] = import_linkages(cursor, ccl_file)
        conn.commit()
    else:
        print(f"Linkage file not found: {ccl_file}")
    
    elapsed = time.time() - start_time
    
    print()
    print("IMPORT COMPLETE")
    for key, value in totals.items():
        print(f"{key.capitalize():20s}: {value:>10,} records")
    print(f"\nTotal Time: {elapsed:.1f} seconds")
    
    cursor.close()
    conn.close()


if __name__ == "__main__":
    main()








