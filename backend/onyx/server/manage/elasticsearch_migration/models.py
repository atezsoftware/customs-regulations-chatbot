from datetime import datetime

from pydantic import BaseModel


class ElasticsearchMigrationStatusResponse(BaseModel):
    model_config = {"frozen": True}
    total_chunks_migrated: int
    created_at: datetime | None
    migration_completed_at: datetime | None
    approx_chunk_count_in_vespa: int | None


class ElasticsearchRetrievalStatusRequest(BaseModel):
    model_config = {"frozen": True}
    enable_elasticsearch_retrieval: bool


class ElasticsearchRetrievalStatusResponse(BaseModel):
    model_config = {"frozen": True}
    enable_elasticsearch_retrieval: bool
    toggling_retrieval_is_disabled: bool = False
