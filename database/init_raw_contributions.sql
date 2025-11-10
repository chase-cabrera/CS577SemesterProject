-- Raw Contributions Table 

CREATE TABLE IF NOT EXISTS raw_contributions (
    id BIGINT AUTO_INCREMENT PRIMARY KEY,
    
    -- FEC Fields
    cmte_id VARCHAR(9),
    amndt_ind VARCHAR(1),
    rpt_tp VARCHAR(3),
    transaction_pgi VARCHAR(5),
    image_num VARCHAR(18),
    transaction_tp VARCHAR(3),
    entity_tp VARCHAR(3),
    name VARCHAR(200),
    city VARCHAR(30),
    state VARCHAR(2),
    zip_code VARCHAR(9),
    employer VARCHAR(38),
    occupation VARCHAR(38),
    transaction_dt DATE,
    transaction_amt DECIMAL(14,2),
    other_id VARCHAR(9),
    tran_id VARCHAR(32),
    file_num INT,
    memo_cd VARCHAR(1),
    memo_text VARCHAR(100),
    sub_id BIGINT,
    first_name VARCHAR(100),
    last_name VARCHAR(100),
    middle_name VARCHAR(100),
    name_suffix VARCHAR(20),
    source_file VARCHAR(255),
    status TINYINT DEFAULT 0 NOT NULL,
    error_message TEXT,
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    processed_at TIMESTAMP NULL,
    
    INDEX idx_status (status),
    INDEX idx_transaction_amt (transaction_amt),
    INDEX idx_cmte_id (cmte_id),
    INDEX idx_transaction_dt (transaction_dt),
    INDEX idx_source_file (source_file),
    INDEX idx_parsed_name (last_name, first_name),
    INDEX idx_status_created (status, created_at)
) ENGINE=InnoDB DEFAULT CHARSET=utf8mb4
PARTITION BY HASH(id) PARTITIONS 8; 










