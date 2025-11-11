import sys
import os
from pathlib import Path
import time
from name_parser import NameParser
from db_utils import get_db_connection

BATCH_SIZE = 5000  # Number of records to insert at once


def import_contributions_file(cursor, file_path, parser):
    """Import contributions from a single itcont file"""
    print(f"\nImporting from {file_path.name}...")
    
    insert_query = """
        INSERT INTO raw_contributions (
            cmte_id, amndt_ind, rpt_tp, transaction_pgi, image_num,
            transaction_tp, entity_tp, name, city, state, zip_code,
            employer, occupation, transaction_dt, transaction_amt,
            other_id, tran_id, file_num, memo_cd, memo_text, sub_id,
            first_name, last_name, middle_name, name_suffix, source_file, status
        ) VALUES (
            %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s,
            %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s
        )
    """
    
    batch = []
    count = 0
    skipped = 0
    
    with open(file_path, 'r', encoding='latin-1') as f:
        for line in f:
            if not line.strip():
                continue
            
            fields = line.strip().split('|')
            
            # FEC itcont format has 21 fields
            if len(fields) < 21:
                skipped += 1
                continue
            
            # Parse name (field 7)
            name = fields[7] if len(fields) > 7 else None
            parsed_name = parser.parse(name)
            
            # Parse transaction date (MMDDYYYY format)
            trans_date = None
            if fields[13] and len(fields[13]) == 8:
                try:
                    month = fields[13][0:2]
                    day = fields[13][2:4]
                    year = fields[13][4:8]
                    trans_date = f"{year}-{month}-{day}"
                except:
                    pass
            
            # Parse transaction amount
            trans_amt = None
            if fields[14]:
                try:
                    trans_amt = float(fields[14])
                except:
                    pass
            
            # Parse file_num
            file_num = None
            if fields[17]:
                try:
                    file_num = int(fields[17])
                except:
                    pass
            
            # Parse sub_id
            sub_id = None
            if fields[20]:
                try:
                    sub_id = int(fields[20])
                except:
                    pass
            
            record = [
                fields[0] if len(fields) > 0 else None,   # cmte_id
                fields[1] if len(fields) > 1 else None,   # amndt_ind
                fields[2] if len(fields) > 2 else None,   # rpt_tp
                fields[3] if len(fields) > 3 else None,   # transaction_pgi
                fields[4] if len(fields) > 4 else None,   # image_num
                fields[5] if len(fields) > 5 else None,   # transaction_tp
                fields[6] if len(fields) > 6 else None,   # entity_tp
                name,                                      # name
                fields[8] if len(fields) > 8 else None,   # city
                fields[9] if len(fields) > 9 else None,   # state
                fields[10] if len(fields) > 10 else None, # zip_code
                fields[11] if len(fields) > 11 else None, # employer
                fields[12] if len(fields) > 12 else None, # occupation
                trans_date,                                # transaction_dt
                trans_amt,                                 # transaction_amt
                fields[15] if len(fields) > 15 else None, # other_id
                fields[16] if len(fields) > 16 else None, # tran_id
                file_num,                                  # file_num
                fields[18] if len(fields) > 18 else None, # memo_cd
                fields[19] if len(fields) > 19 else None, # memo_text
                sub_id,                                    # sub_id
                parsed_name['first_name'],                 # first_name
                parsed_name['last_name'],                  # last_name
                parsed_name['middle_name'],                # middle_name
                parsed_name['name_suffix'],                # name_suffix
                file_path.name,                            # source_file
                0                                          # status (0 = pending)
            ]
            
            batch.append(record)
            
            if len(batch) >= BATCH_SIZE:
                cursor.executemany(insert_query, batch)
                count += len(batch)
                print(f"  Imported {count:,} records...", end='\r')
                batch = []
    
    if batch:
        cursor.executemany(insert_query, batch)
        count += len(batch)
    
    print(f"  Imported {count:,} records from {file_path.name}")
    if skipped > 0:
        print(f"  Skipped {skipped:,} invalid records")
    
    return count


def main():
    import argparse
    
    parser_arg = argparse.ArgumentParser(description='Import raw contributions with name parsing')
    parser_arg.add_argument('--file', type=str, help='Specific file to import (e.g., itcont 2024.txt)')
    parser_arg.add_argument('--pattern', type=str, default='itcont*.txt', help='File pattern to match (default: itcont*.txt)')
    
    args = parser_arg.parse_args()
    
    print("RAW CONTRIBUTIONS IMPORT")
    print()
    
    conn = get_db_connection()
    cursor = conn.cursor()
    parser = NameParser()
    
    # Look for files in data directory
    data_dir = Path(__file__).parent.parent / 'data'
    
    if not data_dir.exists():
        print(f"ERROR: Data directory not found: {data_dir}")
        return
    
    # Determine which files to import
    if args.file:
        # Import specific file
        file_path = data_dir / args.file
        if not file_path.exists():
            print(f"ERROR: File not found: {file_path}")
            return
        itcont_files = [file_path]
        print(f"Importing specific file: {args.file}")
    else:
        # Find all matching files
        itcont_files = sorted(data_dir.glob(args.pattern))
        if not itcont_files:
            print(f"No files matching '{args.pattern}' found in {data_dir}")
            return
        print(f"Found {len(itcont_files)} file(s) to import")
    
    
    start_time = time.time()
    total_records = 0
    
    for file_path in itcont_files:
        count = import_contributions_file(cursor, file_path, parser)
        conn.commit()
    
    
    print("IMPORT COMPLETE")
    
    cursor.close()
    conn.close()


if __name__ == "__main__":
    main()

