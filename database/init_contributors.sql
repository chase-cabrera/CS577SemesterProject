-- Contributors Table

CREATE TABLE IF NOT EXISTS contributors (
    id BIGINT AUTO_INCREMENT PRIMARY KEY,
    
    first_name VARCHAR(100),
    last_name VARCHAR(100),
    middle_name VARCHAR(100),
    name_suffix VARCHAR(20),
    city VARCHAR(30),
    state VARCHAR(2),
    zip_code VARCHAR(9),
    employer VARCHAR(38),
    occupation VARCHAR(38),
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP ON UPDATE CURRENT_TIMESTAMP,
    
    -- Indexes for lookups and deduplication
    INDEX idx_name (last_name, first_name),
    INDEX idx_location (state, city),
    INDEX idx_employer (employer),
    INDEX idx_occupation (occupation)
) ENGINE=InnoDB DEFAULT CHARSET=utf8mb4;








