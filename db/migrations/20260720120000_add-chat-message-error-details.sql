-- Up Migration
ALTER TABLE chat_messages
  ADD COLUMN error_message TEXT;

-- Down Migration
ALTER TABLE chat_messages
  DROP COLUMN error_message;
