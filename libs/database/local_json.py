import json
import threading
import os
from typing import List, Optional
from pydantic import PrivateAttr
from libs.schemas import RequestCompleted
from libs.database.base import Database

class DatabaseLocalJson(Database):
    filepath: str
    _lock: threading.Lock = PrivateAttr(default_factory=threading.Lock)

    def insert_request(self, request: RequestCompleted) -> None:
        with self._lock:
            if os.path.exists(self.filepath):
                with open(self.filepath, 'r') as f:
                    data = json.load(f)
            else:
                data = []

            data.append(request.model_dump())

            with open(self.filepath, 'w') as f:
                json.dump(data, f, indent=2)

    def update_request(self, uuid: str, updated_request: dict) -> None:
        with self._lock:
            if not os.path.exists(self.filepath):
                raise ValueError(f"Database file {self.filepath} does not exist.")

            with open(self.filepath, 'r') as f:
                data = json.load(f)

            for i, req in enumerate(data):
                if req['uuid'] == uuid:
                    data[i].update(updated_request)
                    break
            else:
                raise ValueError(f"Request with uuid {uuid} not found.")

            with open(self.filepath, 'w') as f:
                json.dump(data, f, indent=2)

    def get_request(self, uuid: str) -> Optional[RequestCompleted]:
        with self._lock:
            if not os.path.exists(self.filepath):
                raise ValueError(f"Database file {self.filepath} does not exist.")

            with open(self.filepath, 'r') as f:
                data = json.load(f)

            for req in data:
                if req['uuid'] == uuid:
                    return RequestCompleted(**req)

            return None

    def get_requests(self) -> List[RequestCompleted]:
        with self._lock:
            if not os.path.exists(self.filepath):
                return []

            with open(self.filepath, 'r') as f:
                data = json.load(f)

            return [RequestCompleted(**req) for req in data]

    def get_requests_status_none(self) -> List[RequestCompleted]:
        with self._lock:
            if not os.path.exists(self.filepath):
                return []

            with open(self.filepath, 'r') as f:
                data = json.load(f)

            result = []
            for req in data:
                if 'status' not in req or req['status'] is None:
                    result.append(RequestCompleted(**req))

            return result
