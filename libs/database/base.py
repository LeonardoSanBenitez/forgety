from typing import List, Optional
from pydantic import BaseModel
from libs.schemas import RequestCompleted
from abc import ABC, abstractmethod


class Database(BaseModel, ABC):
    @abstractmethod
    def insert_request(self, request: RequestCompleted) -> None:
        pass

    @abstractmethod
    def update_request(self, uuid: str, updated_request: dict) -> None:
        '''
        Update only the fields that are set in updated_request
        '''
        pass

    @abstractmethod
    def get_request(self, uuid: str) -> Optional[RequestCompleted]:
        pass

    @abstractmethod
    def get_requests(self) -> List[RequestCompleted]:
        pass

    @abstractmethod
    def get_requests_status_none(self) -> List[RequestCompleted]:
        '''
        Retrieve all requests with status None or not set (not yet completed).
        That is, we still need to check the cluster in order to get the latest status and update the database.
        '''
        pass
