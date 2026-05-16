from typing import List, Optional, Union
from libs.schemas import RequestLaunched, RequestCompleted
from libs.database.base import Database


class DatabaseMongo(Database):
    """Placeholder — not implemented."""

    def insert_request(self, request: Union[RequestLaunched, RequestCompleted]) -> None:
        raise NotImplementedError

    def update_request(self, uuid: str, updated_request: dict) -> None:
        raise NotImplementedError

    def get_request(self, uuid: str) -> Optional[Union[RequestLaunched, RequestCompleted]]:
        raise NotImplementedError

    def get_requests(self) -> List[Union[RequestLaunched, RequestCompleted]]:
        raise NotImplementedError

    def get_requests_status_none(self) -> List[Union[RequestLaunched, RequestCompleted]]:
        raise NotImplementedError
