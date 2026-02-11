CREATE TABLE IF NOT EXISTS memory_claims (
  id INT AUTO_INCREMENT PRIMARY KEY,
  user_id VARCHAR(255) NOT NULL,
  claim_type VARCHAR(64) NOT NULL,
  claim_text TEXT NOT NULL,
  confidence FLOAT NOT NULL DEFAULT 0.5,
  last_seen_at TIMESTAMP NULL DEFAULT CURRENT_TIMESTAMP,
  status VARCHAR(32) NOT NULL DEFAULT 'active',
  superseded_by_id INT NULL,
  INDEX idx_memory_claims_user_id (user_id),
  INDEX idx_memory_claims_status (status)
);

ALTER TABLE memory_items
  ADD COLUMN type VARCHAR(64) NOT NULL DEFAULT 'chat_message',
  ADD COLUMN role VARCHAR(32) NULL,
  ADD COLUMN extra TEXT NULL;

CREATE INDEX idx_memory_items_user_type_created ON memory_items (user_id, type, created_at);
CREATE INDEX idx_memory_items_chat_created ON memory_items (chat_id, created_at);

ALTER TABLE memory_items
  ADD COLUMN content TEXT NULL;

-- Optional, only if your table is missing these (your SHOW CREATE shows they exist)
ALTER TABLE memory_items
  ADD COLUMN type VARCHAR(64) NOT NULL DEFAULT 'chat_message',
  ADD COLUMN role VARCHAR(32) NULL,
  ADD COLUMN extra TEXT NULL;

