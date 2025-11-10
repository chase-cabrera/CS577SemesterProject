-- Contributions Table

CREATE TABLE IF NOT EXISTS contributions (
    id BIGINT AUTO_INCREMENT PRIMARY KEY,
    
    -- Foreign key to contributors table
    contributor_id BIGINT NOT NULL,
    
    -- Transaction details
    cmte_id VARCHAR(9),
    transaction_pgi VARCHAR(5),
    transaction_tp VARCHAR(3),
    transaction_dt DATE,
    transaction_amt DECIMAL(14,2),
    amndt_ind VARCHAR(1),
    rpt_tp VARCHAR(3),
    image_num VARCHAR(18),
    entity_tp VARCHAR(3),
    other_id VARCHAR(9),
    tran_id VARCHAR(32),
    file_num INT,
    sub_id BIGINT,
    memo_cd VARCHAR(1),
    memo_text VARCHAR(100),
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    
    -- Foreign key constraint
    FOREIGN KEY (contributor_id) REFERENCES contributors(id),
    
    -- Indexes for queries
    INDEX idx_contributor (contributor_id),
    INDEX idx_committee (cmte_id),
    INDEX idx_date (transaction_dt),
    INDEX idx_amount (transaction_amt),
    INDEX idx_pgi (transaction_pgi),
    INDEX idx_sub_id (sub_id),
    INDEX idx_contributor_cmte (contributor_id, cmte_id)
) ENGINE=InnoDB DEFAULT CHARSET=utf8mb4;








