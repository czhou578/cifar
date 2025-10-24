import asyncio
import logging
from dataclasses import dataclass
from datetime import datetime
from typing import Callable
import uuid

logger = logging.getLogger(__name__)

@dataclass
class QueueJob:
    job_id: str
    status: str
    args: tuple
    created_at: str = None
    priority: int = 0

    def __post_init__(self):
        if self.created_at is None:
            self.created_at = datetime.utcnow().isoformat()

    def __lt__(self, other):
        return self.priority < other.priority

class JobQueue:
    """
        A class to handle Job queues
    """

    def __init__(self, num_workers = 4, max_queue_size = 100):
        self.num_workers = num_workers
        self.max_queue_size = max_queue_size
        self.job_queue = asyncio.PriorityQueue(maxsize=max_queue_size)
        self.workers: list[asyncio.Task] = []
        self.is_running = False

        self.total_jobs_processed = 0
        self.active_jobs = 0

        logger.info(f"Initialized JobQueue with {num_workers} workers and max size {max_queue_size}")

    async def start(self):
        if self.is_running:
            logger.info("JobQueue already running")
            return
        
        self.is_running = True

        for i in range(self.num_workers):
            worker_task = asyncio.create_task(self.worker(i))
            self.workers.append(worker_task)

    async def stop(self):
        if not self.is_running:
            logger.info("JobQueue not running")
            return
        
        self.is_running = False

        for worker in self.workers:
            worker.cancel()
        
        await asyncio.gather(*self.workers, return_exceptions=True)

        logger.info("JobQueue stopped")
    
    async def submit(self, task_func: Callable, *args, priority: int = 0, **kwargs) -> str:

        job_id = str(uuid.uuid4())
        job = QueueJob(job_id=job_id, status="pending", args=(task_func, *args), priority=priority)

        try:
            await self.job_queue.put(job)
            logger.info(f"Submitted job {job_id} to the queue")
            return job_id
        
        except asyncio.QueueFull:
            logger.warning(f"Job queue is full. Job {job_id} was not submitted.")
            return job_id

        except asyncio.TimeoutError:
            logger.warning(f"Job queue submission timed out. Job {job_id} was not submitted.")
            return job_id
    
    async def worker(self, worker_id: int):
        logger.info(f"Worker {worker_id} started")
        while self.is_running:
            try:
                job: QueueJob = await self.job_queue.get()
                self.active_jobs += 1
                logger.info(f"Worker {worker_id} picked up job {job.job_id}")

                task_func, *task_args = job.args
                await task_func(*task_args)

                job.status = "completed"
                self.total_jobs_processed += 1
                self.active_jobs -= 1
                self.job_queue.task_done()
                logger.info(f"Worker {worker_id} completed job {job.job_id}")

            except asyncio.CancelledError:
                logger.info(f"Worker {worker_id} cancelled")
                break

            except Exception as e:
                logger.error(f"Error in worker {worker_id} while processing job {job.job_id}: {e}")
                job.status = "failed"
                self.active_jobs -= 1
                self.job_queue.task_done()

job_queue = None

def get_job_queue() -> JobQueue:
    global job_queue
    if job_queue is None:
        job_queue = JobQueue(num_workers=4, max_queue_size=100)
        asyncio.create_task(job_queue.start())
    return job_queue
