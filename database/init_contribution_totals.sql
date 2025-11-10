-- Contribution Totals Table

CREATE TABLE IF NOT EXISTS contribution_totals (
    id BIGINT AUTO_INCREMENT PRIMARY KEY,
    
    -- Aggregation keys
    contributor_id BIGINT NOT NULL,
    cmte_id VARCHAR(9) NOT NULL,
    transaction_pgi VARCHAR(5),
    total_amount DECIMAL(14,2),
    contribution_count INT,
    first_contribution_date DATE,
    last_contribution_date DATE,
    average_contribution DECIMAL(14,2),
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP ON UPDATE CURRENT_TIMESTAMP,
    
    -- Foreign key constraint
    FOREIGN KEY (contributor_id) REFERENCES contributors(id),
    
    -- Unique constraint - one row per contributor+committee+pgi combination
    UNIQUE KEY unique_contributor_cmte_pgi (contributor_id, cmte_id, transaction_pgi),
    
    -- Indexes for queries
    INDEX idx_contributor (contributor_id),
    INDEX idx_committee (cmte_id),
    INDEX idx_total_amount (total_amount),
    INDEX idx_pgi (transaction_pgi)
) ENGINE=InnoDB DEFAULT CHARSET=utf8mb4;








