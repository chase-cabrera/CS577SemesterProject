-- Candidates Table

CREATE TABLE IF NOT EXISTS candidates (
    cand_id VARCHAR(9) PRIMARY KEY,
    cand_name VARCHAR(200),
    cand_pty_affiliation VARCHAR(3),
    cand_election_yr INT,
    cand_office_st VARCHAR(2),
    cand_office VARCHAR(1),
    cand_office_district VARCHAR(2),
    cand_ici VARCHAR(1),
    cand_status VARCHAR(1),
    cand_pcc VARCHAR(9),
    cand_st1 VARCHAR(34),
    cand_st2 VARCHAR(34),
    cand_city VARCHAR(30),
    cand_st VARCHAR(2),
    cand_zip VARCHAR(9),
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP ON UPDATE CURRENT_TIMESTAMP,
    
    -- Indexes
    INDEX idx_name (cand_name),
    INDEX idx_party (cand_pty_affiliation),
    INDEX idx_election_yr (cand_election_yr),
    INDEX idx_office (cand_office, cand_office_st),
    INDEX idx_status (cand_status)
) ENGINE=InnoDB DEFAULT CHARSET=utf8mb4;








