import multiprocessing as mp
import logging
import time
from typing import Optional, Dict, Any
import threading

from workers.caption_worker import caption_worker_process

logger = logging.getLogger(__name__)

class CaptionWorkerManager:
    """
    Manages caption worker process lifecycle
    
    Responsibilities:
    - Start/stop worker process
    - Monitor worker health
    - Restart crashed workers
    - Route messages to/from worker
    """
    
    def __init__(self, device: str = "cuda"):
        self.device = device
        
        # Communication queues
        self.request_queue: Optional[mp.Queue] = None
        self.response_queue: Optional[mp.Queue] = None
        
        # Worker process
        self.worker_process: Optional[mp.Process] = None
        self.worker_pid: Optional[int] = None
        
        # Health monitoring
        self.last_heartbeat: Optional[float] = None
        self.heartbeat_thread: Optional[threading.Thread] = None
        self.running = False
        
    def start(self):
        """Start the worker process"""
        if self.running:
            logger.warning("Worker already running")
            return
        
        logger.info("Starting caption worker process...")
        
        # Create queues
        self.request_queue = mp.Queue(maxsize=100)
        self.response_queue = mp.Queue(maxsize=100)
        
        # Start worker process
        self.worker_process = mp.Process(
            target=caption_worker_process,
            args=(self.request_queue, self.response_queue, self.device),
            daemon=False  # Don't die with parent
        )
        self.worker_process.start()
        
        # Wait for ready signal
        try:
            ready_msg = self.response_queue.get(timeout=60)  # 60 sec for model loading
            if ready_msg.get("type") == "ready":
                self.worker_pid = ready_msg.get("pid")
                self.running = True
                logger.info(f"✅ Caption worker ready (PID: {self.worker_pid})")
                
                # Start heartbeat monitoring
                self.start_heartbeat_monitor()
            else:
                raise RuntimeError(f"Unexpected ready message: {ready_msg}")
                
        except Exception as e:
            logger.error(f"Worker failed to start: {e}")
            self.stop()
            raise
    
    def stop(self):
        """Stop the worker process gracefully"""
        if not self.running:
            return
        
        logger.info("Stopping caption worker...")
        self.running = False
        
        # Stop heartbeat monitoring
        if self.heartbeat_thread:
            self.heartbeat_thread.join(timeout=5)
        
        # Send shutdown signal
        if self.request_queue:
            try:
                self.request_queue.put({"type": "shutdown"}, timeout=5)
            except:
                pass
        
        # Wait for process to stop
        if self.worker_process:
            self.worker_process.join(timeout=10)
            
            # Force kill if still alive
            if self.worker_process.is_alive():
                logger.warning("Force killing worker process")
                self.worker_process.terminate()
                self.worker_process.join(timeout=5)
                
                if self.worker_process.is_alive():
                    self.worker_process.kill()
        
        logger.info("Caption worker stopped")
    
    def submit_caption_request(
        self,
        job_id: str,
        image_bytes: bytes,
        streaming: bool = False
    ):
        """
        Submit caption request to worker
        
        Non-blocking - doesn't wait for response
        """
        if not self.running:
            raise RuntimeError("Worker not running")
        
        message = {
            "type": "caption_request",
            "job_id": job_id,
            "image_bytes": image_bytes,
            "streaming": streaming
        }
        
        try:
            self.request_queue.put(message, timeout=5)
        except:
            raise RuntimeError("Worker queue full - too many pending requests")
    
    def get_response(self, timeout: float = 1.0) -> Optional[Dict[str, Any]]:
        """
        Get response from worker (non-blocking with timeout)
        
        Returns None if no response available
        """
        try:
            return self.response_queue.get(timeout=timeout)
        except:
            return None
    
    def start_heartbeat_monitor(self):
        """Start background thread to monitor worker health"""
        def heartbeat_loop():
            while self.running:
                try:
                    # Send heartbeat
                    self.request_queue.put({"type": "heartbeat"})
                    
                    # Wait for ack
                    response = self.get_response(timeout=5)
                    if response and response.get("type") == "heartbeat_ack":
                        self.last_heartbeat = time.time()
                    else:
                        logger.warning("No heartbeat ack from worker")
                        # Check if worker died
                        if not self.worker_process.is_alive():
                            logger.error("Worker process died!")
                            # TODO: Implement auto-restart
                    
                    time.sleep(10)  # Check every 10 seconds
                    
                except Exception as e:
                    logger.error(f"Heartbeat error: {e}")
        
        self.heartbeat_thread = threading.Thread(target=heartbeat_loop, daemon=True)
        self.heartbeat_thread.start()
    
    def get_stats(self) -> Dict[str, Any]:
        """Get worker statistics"""
        return {
            "running": self.running,
            "worker_pid": self.worker_pid,
            "process_alive": self.worker_process.is_alive() if self.worker_process else False,
            "last_heartbeat": self.last_heartbeat,
            "request_queue_size": self.request_queue.qsize() if self.request_queue else 0,
            "response_queue_size": self.response_queue.qsize() if self.response_queue else 0,
        }


# Global instance
worker_manager: Optional[CaptionWorkerManager] = None

def get_worker_manager() -> CaptionWorkerManager:
    """Get or create worker manager"""
    global worker_manager
    if worker_manager is None:
        worker_manager = CaptionWorkerManager()
    return worker_manager