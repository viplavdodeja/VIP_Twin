CREATE TABLE IF NOT EXISTS memory_items (
  id INT AUTO_INCREMENT PRIMARY KEY,
  user_id VARCHAR(255) NOT NULL,
  chat_id VARCHAR(255) NULL,
  type VARCHAR(64) NOT NULL,
  role VARCHAR(32) NULL,
  content TEXT NOT NULL,
  created_at DATETIME NOT NULL,
  extra TEXT NULL,
  INDEX idx_memory_items_user_type_created (user_id, type, created_at),
  INDEX idx_memory_items_chat_created (chat_id, created_at)
);
