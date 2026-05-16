from typing import List, Optional, Union
from pydantic import BaseModel
from libs.schemas import RequestLaunched, RequestCompleted
from abc import ABC, abstractmethod


class Database(BaseModel, ABC):
    @abstractmethod
    def insert_request(self, request: Union[RequestLaunched, RequestCompleted]) -> None:
        """Persist a request at any lifecycle stage."""
        pass

    @abstractmethod
    def update_request(self, uuid: str, updated_request: dict) -> None:
        '''
        Update only the fields that are set in updated_request
        '''
        pass

    @abstractmethod
    def get_request(self, uuid: str) -> Optional[Union[RequestLaunched, RequestCompleted]]:
        pass

    @abstractmethod
    def get_requests(self) -> List[Union[RequestLaunched, RequestCompleted]]:
        pass

    @abstractmethod
    def get_requests_status_none(self) -> List[Union[RequestLaunched, RequestCompleted]]:
        '''
        Retrieve all requests with status None or not set (not yet completed).
        That is, we still need to check the cluster in order to get the latest status and update the database.
        '''
        pass
