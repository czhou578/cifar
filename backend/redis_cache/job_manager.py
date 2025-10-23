'''
redis cache job manager for long running tasks

System design:

    - use redis to store job status and results
    - use a background worker to process jobs from a queue
    - implement a retry mechanism for failed jobs
    - provide a way to cancel jobs
    - include progress tracking for long-running tasks
    - ensure idempotency of job processing

    batchjob class

JobManager Class to handle job creation, status updates, result storage, and retrieval

- initialize redis connection
- save job to redis
- update job status
- retrieve job by id
- delete job
- recover interrupted jobs on startup
'''
from enum import Enum
import logging
from typing import Optional, Any, Dict, List, Generator
from datetime import datetime
import redis
import json
from dataclasses import dataclass, field
import uuid

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class JobStatus(Enum):
    PENDING = "PENDING"
    IN_PROGRESS = "IN_PROGRESS"
    COMPLETED = "COMPLETED"
    FAILED = "FAILED"
    CANCELED = "CANCELED"

@dataclass
class BatchJob:
    job_id: str
    status: JobStatus
    processed_images: int = 0
    total_images: int = 0
    created_at: str = None
    updated_at: str = None
    results: List[Dict] = field(default_factory=list)
    error_message: Optional[str] = None
    captions: List[str] = field(default_factory=list)

    def __post_init__(self):
        if self.results is None:
            self.results = []

        if self.captions is None:
            self.captions = []

        if self.created_at is None:
            self.created_at = datetime.utcnow().isoformat()

        if self.updated_at is None:
            self.updated_at = self.created_at

    def to_dict(self) -> Dict[str, Any]:

        def make_serializable(obj):
            if isinstance(obj, (str, int, float, bool, type(None))):
                return obj
            elif isinstance(obj, dict):
                return {k: make_serializable(v) for k, v in obj.items()}
            elif isinstance(obj, (list, tuple)):
                return [make_serializable(i) for i in obj]
            elif hasattr(obj, '__dict__'):
                return str(obj)
            else:
                return str(obj)
            
        return {
            "job_id": self.job_id,
            "status": self.status.value,
            "processed_images": self.processed_images,
            "total_images": self.total_images,
            "captions": make_serializable(self.captions),
            "created_at": self.created_at,
            "updated_at": self.updated_at,
            "results": make_serializable(self.results),
            "error_message": self.error_message
        }
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'BatchJob':
        job = cls(
            job_id=data["job_id"],
            status=JobStatus(data["status"]),
            processed_images=data.get("processed_images", 0),
            total_images=data.get("total_images", 0),
            created_at=data.get("created_at"),
            updated_at=data.get("updated_at"),
            results=data.get("results", []),
            error_message=data.get("error_message"),
            captions=data.get("captions", [])
        )
        return job
    
class JobManager:
    def __init__(self, host='localhost', port=6379):
        try:
            self.redis = redis.Redis(
                host=host, 
                port=port, 
                db=0, 
                decode_responses=True,
                socket_connect_timeout=5, 
                socket_timeout=5
            )
            self.redis.ping()
            self.redis_available = True
            logger.info(f"✅ Connected to Redis at {host}:{port}")
        except redis.ConnectionError as e:
            self.redis_available = False
            logger.error(f"❌ Failed to connect to Redis: {e}")
            raise
    
    def _safe_json_dumps(self, data: Dict) -> str:
        """Safely serialize to JSON with fallback"""
        try:
            return json.dumps(data)
        except (TypeError, ValueError) as e:
            logger.error(f"JSON serialization error: {e}")
            # Try again with default=str to convert non-serializable objects
            try:
                return json.dumps(data, default=str)
            except Exception as e2:
                logger.error(f"JSON serialization failed even with default=str: {e2}")
                raise ValueError(f"Cannot serialize job data: {e2}")
    
    def create_job(self, total_images: int) -> str:
        """Create a new job and return its ID"""
        job_id = str(uuid.uuid4())
        job = BatchJob(
            job_id=job_id,
            status=JobStatus.PENDING,
            total_images=total_images
        )
        self.save_job(job)
        return job_id
    
    def save_job(self, job: BatchJob):
        """Save job to Redis with safe serialization"""
        try:
            job_dict = job.to_dict()
            job_json = self._safe_json_dumps(job_dict)
            self.redis.set(job.job_id, job_json)
            logger.info(f"Job {job.job_id} saved to Redis")
        except Exception as e:
            logger.error(f"Failed to save job {job.job_id}: {e}")
            raise

    def get_job(self, job_id: str) -> Optional[BatchJob]:
        """Retrieve job from Redis"""
        try:
            job_data = self.redis.get(job_id)
            if job_data:
                return BatchJob.from_dict(json.loads(job_data))
            return None
        except Exception as e:
            logger.error(f"Failed to get job {job_id}: {e}")
            return None
    
    def update_job(self, job: BatchJob):
        """Update existing job in Redis"""
        try:
            job.updated_at = datetime.utcnow().isoformat()
            job_dict = job.to_dict()
            job_json = self._safe_json_dumps(job_dict)
            self.redis.set(job.job_id, job_json)
            logger.info(f"Job {job.job_id} updated in Redis")
        except Exception as e:
            logger.error(f"Failed to update job {job.job_id}: {e}")
            raise
    
    def update_job_status(self, job_id: str, status: JobStatus, error_message: Optional[str] = None):
        """Update job status"""
        job = self.get_job(job_id)
        if job:
            job.status = status
            if error_message:
                job.error_message = error_message
            self.update_job(job)
            logger.info(f"Job {job_id} status updated to {status.value} in Redis")
        else:
            logger.warning(f"Job {job_id} not found for status update")

    def delete_job(self, job_id: str):
        """Delete job from Redis"""
        self.redis.delete(job_id)
        logger.info(f"Job {job_id} deleted from Redis")
    
    def recover_jobs(self) -> Generator[BatchJob, None, None]:
        for key in self.redis.scan_iter():
            job_data = self.redis.get(key)
            if job_data:
                job = BatchJob.from_dict(json.loads(job_data))
                if job.status in {JobStatus.PENDING, JobStatus.IN_PROGRESS}:
                    logger.info(f"Recovering job {job.job_id} with status {job.status}")
                    yield job

job_manager = JobManager(host='localhost', port=6379)