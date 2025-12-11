"""
Parallel Import for Large FEC Files
Imports from multiple positions in the file simultaneously to speed up import
"""

import sys
import os
from pathlib import Path
import time
import multiprocessing as mp
from multiprocessing import Process, Value, Lock
import argparse

sys.path.append(os.path.dirname(os.path.abspath(__file__)))
from db_utils import get_db_connection
from name_parser import NameParser

BATCH_SIZE = 25000
COMMIT_INTERVAL = 100000


def skip_to_line(file_path, skip_lines):
    """
    Find the byte position after skipping N lines
    
    Returns byte position to start reading from
    """
    print(f"Skipping first {skip_lines:,} lines...")
    
    with open(file_path, 'rb') as f:
        for i in range(skip_lines):
            f.readline()
            if i > 0 and i % 1000000 == 0:
                print(f"  Skipped {i:,} lines...")
        
        start_byte = f.tell()
    
    print(f"  Starting at byte position: {start_byte:,}")
    return start_byte


def get_file_chunks(file_path, num_chunks, start_byte=0):
    """
    Divide file into chunks by finding line boundaries
    
    Args:
        file_path: Path to file
        num_chunks: Number of chunks to create
        start_byte: Byte position to start from (for resuming)
    
    Returns list of (start_byte, end_byte) tuples
    """
    file_size = os.path.getsize(file_path)
    remaining_size = file_size - start_byte
    chunk_size = remaining_size // num_chunks
    
    chunks = []
    
    with open(file_path, 'rb') as f:
        current_start = start_byte
        
        for i in range(num_chunks):
            if i == num_chunks - 1:
                # Last chunk goes to end of file
                end = file_size
            else:
                # Seek to approximate chunk boundary
                f.seek(current_start + chunk_size)
                # Read until we find a newline (complete the current line)
                f.readline()
                end = f.tell()
            
            chunks.append((current_start, end))
            current_start = end
    
    return chunks


def import_chunk(worker_id, file_path, start_byte, end_byte, total_inserted, lock, source_file):
    """
    Import a chunk of the file
    
    Args:
        worker_id: Worker identifier
        file_path: Path to the file
        start_byte: Starting byte position
        end_byte: Ending byte position
        total_inserted: Shared counter for total records
        lock: Lock for thread-safe counter updates
        source_file: Source file name for database
    """
    conn = get_db_connection()
    cursor = conn.cursor()
    parser = NameParser()
    
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
    last_commit = 0
    
    with open(file_path, 'r', encoding='latin-1') as f:
        # Seek to start position
        f.seek(start_byte)
        
        # If not at start of file, skip partial first line
        if start_byte > 0:
            f.readline()
        
        while f.tell() < end_byte:
            line = f.readline()
            if not line:
                break
            
            if not line.strip():
                continue
            
            fields = line.strip().split('|')
            
            if len(fields) < 21:
                continue
            
            # Parse name
            name = fields[7] if len(fields) > 7 else None
            parsed_name = parser.parse(name)
            
            # Parse transaction date
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
                fields[0] if len(fields) > 0 else None,
                fields[1] if len(fields) > 1 else None,
                fields[2] if len(fields) > 2 else None,
                fields[3] if len(fields) > 3 else None,
                fields[4] if len(fields) > 4 else None,
                fields[5] if len(fields) > 5 else None,
                fields[6] if len(fields) > 6 else None,
                name,
                fields[8] if len(fields) > 8 else None,
                fields[9] if len(fields) > 9 else None,
                fields[10] if len(fields) > 10 else None,
                fields[11] if len(fields) > 11 else None,
                fields[12] if len(fields) > 12 else None,
                trans_date,
                trans_amt,
                fields[15] if len(fields) > 15 else None,
                fields[16] if len(fields) > 16 else None,
                file_num,
                fields[18] if len(fields) > 18 else None,
                fields[19] if len(fields) > 19 else None,
                sub_id,
                parsed_name['first_name'],
                parsed_name['last_name'],
                parsed_name['middle_name'],
                parsed_name['name_suffix'],
                source_file,
                0
            ]
            
            batch.append(record)
            
            if len(batch) >= BATCH_SIZE:
                cursor.executemany(insert_query, batch)
                count += len(batch)
                batch = []
                
                # Commit periodically
                if count - last_commit >= COMMIT_INTERVAL:
                    conn.commit()
                    last_commit = count
                    
                    with lock:
                        total_inserted.value += COMMIT_INTERVAL
                        print(f"Worker {worker_id}: {count:,} records | Total: {total_inserted.value:,}")
    
    # Insert remaining batch
    if batch:
        cursor.executemany(insert_query, batch)
        count += len(batch)
    
    conn.commit()
    
    with lock:
        remaining = count - last_commit
        total_inserted.value += remaining
        print(f"Worker {worker_id} DONE: {count:,} records")
    
    cursor.close()
    conn.close()
    
    return count


def main():
    parser = argparse.ArgumentParser(description='Parallel import of FEC contribution files')
    parser.add_argument('--file', type=str, required=True, help='File to import (e.g., "itcont 2022.txt")')
    parser.add_argument('--workers', type=int, default=4, help='Number of parallel workers (default: 4)')
    parser.add_argument('--skip-lines', type=int, default=0, help='Number of lines to skip (for resuming)')
    
    args = parser.parse_args()
    
    data_dir = Path(__file__).parent.parent / 'data'
    file_path = data_dir / args.file
    
    if not file_path.exists():
        print(f"ERROR: File not found: {file_path}")
        return
    
    file_size = os.path.getsize(file_path)
    print(f"PARALLEL IMPORT: {args.file}")
    print(f"File size: {file_size / (1024**3):.2f} GB")
    print(f"Workers: {args.workers}")
    
    # Skip lines if resuming
    start_byte = 0
    if args.skip_lines > 0:
        start_byte = skip_to_line(str(file_path), args.skip_lines)
        remaining_size = file_size - start_byte
        print(f"Remaining to import: {remaining_size / (1024**3):.2f} GB")
    
    # Divide file into chunks
    chunks = get_file_chunks(str(file_path), args.workers, start_byte)
    
    print(f"\nFile chunks:")
    for i, (start, end) in enumerate(chunks):
        chunk_mb = (end - start) / (1024**2)
        print(f"  Worker {i}: bytes {start:,} - {end:,} ({chunk_mb:.1f} MB)")
    
    # Shared counter
    total_inserted = Value('i', 0)
    lock = Lock()
    
    start_time = time.time()
    
    # Start worker processes
    processes = []
    for i, (start, end) in enumerate(chunks):
        p = Process(target=import_chunk, 
                   args=(i, str(file_path), start, end, total_inserted, lock, args.file))
        processes.append(p)
        p.start()
    
    # Wait for all workers to complete
    for p in processes:
        p.join()
    
    elapsed = time.time() - start_time
    
    print(f"\nIMPORT COMPLETE")
    print(f"Total records: {total_inserted.value:,}")
    print(f"Total time: {elapsed/60:.1f} minutes ({elapsed:.0f}s)")
    print(f"Records/second: {total_inserted.value/elapsed:,.0f}")


if __name__ == "__main__":
    main()

