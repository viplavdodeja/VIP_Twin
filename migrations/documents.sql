CREATE TABLE IF NOT EXISTS documents (
  id INT AUTO_INCREMENT PRIMARY KEY,
  user_id VARCHAR(255) NOT NULL,
  title VARCHAR(255) NOT NULL,
  doc_type VARCHAR(64) NOT NULL,
  source VARCHAR(255) NULL,
  created_at TIMESTAMP NULL DEFAULT CURRENT_TIMESTAMP,
  INDEX idx_documents_user_id (user_id)
);

CREATE TABLE IF NOT EXISTS document_chunks (
  id INT AUTO_INCREMENT PRIMARY KEY,
  document_id INT NOT NULL,
  chunk_index INT NOT NULL,
  chunk_text TEXT NOT NULL,
  metadata_json TEXT NULL,
  created_at TIMESTAMP NULL DEFAULT CURRENT_TIMESTAMP,
  INDEX idx_document_chunks_doc (document_id),
  FOREIGN KEY (document_id) REFERENCES documents(id)
);

CREATE TABLE IF NOT EXISTS retrieval_runs (
  id INT AUTO_INCREMENT PRIMARY KEY,
  user_id VARCHAR(255) NOT NULL,
  layer VARCHAR(16) NOT NULL,
  query_text TEXT NOT NULL,
  top_k INT NOT NULL,
  created_at TIMESTAMP NULL DEFAULT CURRENT_TIMESTAMP,
  INDEX idx_retrieval_runs_user (user_id),
  INDEX idx_retrieval_runs_layer (layer)
);

ALTER TABLE documents
  ADD COLUMN title VARCHAR(255) NOT NULL,
  ADD COLUMN doc_type VARCHAR(64) NOT NULL;
  
ALTER TABLE documents
  MODIFY COLUMN source TEXT NULL;

SHOW CREATE TABLE documents;

SHOW CREATE TABLE document_chunks;



