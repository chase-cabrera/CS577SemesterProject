-- Candidate Committee Links Table

CREATE TABLE IF NOT EXISTS candidate_committee_links (
    id BIGINT AUTO_INCREMENT PRIMARY KEY,
    
    -- Linkage
    cand_id VARCHAR(9),
    cand_election_yr INT,
    fec_election_yr INT,
    cmte_id VARCHAR(9),
    cmte_tp VARCHAR(1),
    cmte_dsgn VARCHAR(1),
    linkage_id BIGINT,
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    
    -- Unique constraint
    UNIQUE KEY unique_link (cand_id, cmte_id, cand_election_yr),
    
    -- Indexes
    INDEX idx_cand_id (cand_id),
    INDEX idx_cmte_id (cmte_id),
    INDEX idx_election_yr (cand_election_yr),
    INDEX idx_linkage_id (linkage_id)
) ENGINE=InnoDB DEFAULT CHARSET=utf8mb4;








