-- Committees Table

CREATE TABLE IF NOT EXISTS committees (
    cmte_id VARCHAR(9) PRIMARY KEY,
    
    -- Committee information
    cmte_nm VARCHAR(200),
    tres_nm VARCHAR(90),
    cmte_st1 VARCHAR(34),
    cmte_st2 VARCHAR(34),
    cmte_city VARCHAR(30),
    cmte_st VARCHAR(2),
    cmte_zip VARCHAR(9),
    cmte_dsgn VARCHAR(1),
    cmte_tp VARCHAR(1),
    cmte_pty_affiliation VARCHAR(3),
    cmte_filing_freq VARCHAR(1),
    org_tp VARCHAR(1),
    connected_org_nm VARCHAR(200),
    cand_id VARCHAR(9),
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP ON UPDATE CURRENT_TIMESTAMP,
    
    -- Indexes
    INDEX idx_name (cmte_nm),
    INDEX idx_type (cmte_tp),
    INDEX idx_party (cmte_pty_affiliation),
    INDEX idx_cand_id (cand_id),
    INDEX idx_state (cmte_st)
) ENGINE=InnoDB DEFAULT CHARSET=utf8mb4;








