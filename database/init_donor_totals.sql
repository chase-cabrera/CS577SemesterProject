-- Unified Contributor Totals Table
-- One row per contributor-committee pair
-- Candidate info denormalized in (NULL if committee not linked to candidate)

DROP TABLE IF EXISTS donor_totals;

CREATE TABLE donor_totals (
    id BIGINT AUTO_INCREMENT PRIMARY KEY,
    
    -- Contributor
    contributor_id BIGINT NOT NULL,
    
    -- Committee (always present)
    cmte_id VARCHAR(9) NOT NULL,
    cmte_name VARCHAR(200),
    cmte_type VARCHAR(1),
    cmte_party VARCHAR(3),
    cmte_state VARCHAR(2),
    
    -- Candidate (NULL if committee not linked to a candidate)
    cand_id VARCHAR(9),
    cand_name VARCHAR(200),
    cand_party VARCHAR(3),
    cand_office VARCHAR(1),
    cand_state VARCHAR(2),
    
    -- Aggregated donation metrics (contributor -> committee)
    total_donations INT DEFAULT 0,
    total_amount DECIMAL(14,2) DEFAULT 0,
    avg_amount DECIMAL(14,2) DEFAULT 0,
    min_amount DECIMAL(14,2) DEFAULT 0,
    max_amount DECIMAL(14,2) DEFAULT 0,
    stddev_amount DECIMAL(14,2) DEFAULT 0,
    
    -- Temporal features
    first_donation_date DATE,
    last_donation_date DATE,
    days_since_first INT,
    days_since_last INT,
    donation_span_days INT,
    avg_days_between_donations DECIMAL(10,2),
    
    -- Recent activity
    donations_last_1year INT DEFAULT 0,
    amount_last_1year DECIMAL(14,2) DEFAULT 0,
    donations_last_2years INT DEFAULT 0,
    amount_last_2years DECIMAL(14,2) DEFAULT 0,
    
    -- Election cycle patterns
    donations_primary INT DEFAULT 0,
    donations_general INT DEFAULT 0,
    amount_primary DECIMAL(14,2) DEFAULT 0,
    amount_general DECIMAL(14,2) DEFAULT 0,
    
    -- Trend and pattern indicators
    amount_trend DECIMAL(10,4),
    is_recurring TINYINT DEFAULT 0,
    
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP ON UPDATE CURRENT_TIMESTAMP,
    
    -- Unique constraint
    UNIQUE KEY idx_contributor_committee (contributor_id, cmte_id),
    
    -- Foreign key
    FOREIGN KEY (contributor_id) REFERENCES contributors(id),
    
    -- Core lookup indexes
    INDEX idx_contributor (contributor_id),
    INDEX idx_committee (cmte_id),
    INDEX idx_candidate (cand_id),
    
    -- Query optimization indexes
    INDEX idx_total_donations (total_donations),
    INDEX idx_total_amount (total_amount),
    INDEX idx_last_donation (last_donation_date),
    INDEX idx_days_since_last (days_since_last),
    INDEX idx_recurring (is_recurring),
    INDEX idx_cmte_party (cmte_party),
    INDEX idx_cand_party (cand_party),
    
    -- Composite indexes for prediction queries
    INDEX idx_contributor_recency (contributor_id, days_since_last),
    INDEX idx_cmte_amount (cmte_id, total_amount DESC),
    INDEX idx_cand_amount (cand_id, total_amount DESC),
    INDEX idx_active_donors (cmte_id, donations_last_2years DESC),
    INDEX idx_contributor_cand (contributor_id, cand_id)
) ENGINE=InnoDB DEFAULT CHARSET=utf8mb4;


-- Convenience view: Only rows linked to candidates
CREATE OR REPLACE VIEW donor_candidate_totals AS
SELECT 
    id,
    contributor_id,
    cmte_id,
    cmte_name,
    cand_id,
    cand_name,
    cand_party,
    cand_office,
    cand_state,
    total_donations,
    total_amount,
    avg_amount,
    first_donation_date,
    last_donation_date,
    days_since_last,
    donations_last_2years,
    amount_last_2years,
    is_recurring
FROM donor_totals
WHERE cand_id IS NOT NULL;


-- Convenience view: Aggregate by contributor-candidate (sum across all their committees)
CREATE OR REPLACE VIEW donor_candidate_summary AS
SELECT 
    contributor_id,
    cand_id,
    MAX(cand_name) as cand_name,
    MAX(cand_party) as cand_party,
    MAX(cand_office) as cand_office,
    MAX(cand_state) as cand_state,
    COUNT(DISTINCT cmte_id) as committees_donated_to,
    SUM(total_donations) as total_donations,
    SUM(total_amount) as total_amount,
    AVG(avg_amount) as avg_amount,
    MIN(first_donation_date) as first_donation_date,
    MAX(last_donation_date) as last_donation_date,
    MIN(days_since_last) as days_since_last,
    SUM(donations_last_2years) as donations_last_2years,
    SUM(amount_last_2years) as amount_last_2years,
    MAX(is_recurring) as is_recurring
FROM donor_totals
WHERE cand_id IS NOT NULL
GROUP BY contributor_id, cand_id;
