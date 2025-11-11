-- Contributors Table with Aggregated Donor Features

CREATE TABLE IF NOT EXISTS contributors (
    id BIGINT AUTO_INCREMENT PRIMARY KEY,
    
    -- Identity
    donor_key VARCHAR(255) UNIQUE,
    first_name VARCHAR(100),
    last_name VARCHAR(100),
    middle_name VARCHAR(100),
    name_suffix VARCHAR(20),
    city VARCHAR(30),
    state VARCHAR(2),
    zip_code VARCHAR(9),
    employer VARCHAR(38),
    occupation VARCHAR(38),
    
    -- Aggregated activity metrics
    total_donations INT DEFAULT 0,
    total_amount DECIMAL(14,2) DEFAULT 0,
    donations_last_2years INT DEFAULT 0,
    amount_last_2years DECIMAL(14,2) DEFAULT 0,
    recency_days INT,
    avg_donation_amount DECIMAL(14,2),
    first_donation_date DATE,
    last_donation_date DATE,
    unique_committees INT DEFAULT 0,
    
    -- Party affiliation metrics
    dem_donations INT DEFAULT 0,
    dem_amount DECIMAL(14,2) DEFAULT 0,
    rep_donations INT DEFAULT 0,
    rep_amount DECIMAL(14,2) DEFAULT 0,
    lib_donations INT DEFAULT 0,
    lib_amount DECIMAL(14,2) DEFAULT 0,
    gre_donations INT DEFAULT 0,
    gre_amount DECIMAL(14,2) DEFAULT 0,
    ind_donations INT DEFAULT 0,
    ind_amount DECIMAL(14,2) DEFAULT 0,
    other_donations INT DEFAULT 0,
    other_amount DECIMAL(14,2) DEFAULT 0,
    
    -- Computed scores
    dem_pct DECIMAL(5,2),
    rep_pct DECIMAL(5,2),
    primary_party VARCHAR(10),
    
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP ON UPDATE CURRENT_TIMESTAMP,
    
    -- Indexes for lookups and analysis
    INDEX idx_donor_key (donor_key),
    INDEX idx_name (last_name, first_name),
    INDEX idx_location (state, city),
    INDEX idx_state (state),
    INDEX idx_zip (zip_code),
    INDEX idx_employer (employer),
    INDEX idx_occupation (occupation),
    INDEX idx_dem_pct (dem_pct),
    INDEX idx_rep_pct (rep_pct),
    INDEX idx_primary_party (primary_party),
    INDEX idx_total_amount (total_amount),
    INDEX idx_last_donation (last_donation_date),
    INDEX idx_fuzzy_match (zip_code, employer, first_name)
) ENGINE=InnoDB DEFAULT CHARSET=utf8mb4;








