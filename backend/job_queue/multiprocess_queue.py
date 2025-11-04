
"""
request message

job_id, image_bytes, parameters

response message

job_id, caption/error, status

heartbeat message: 'ACK'

JSON for serialization

"""

from multiprocessing import Process, Queue
import time

