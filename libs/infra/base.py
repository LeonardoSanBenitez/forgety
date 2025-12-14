from typing import List, Optional
from pydantic import BaseModel
from abc import ABC, abstractmethod
from libs.schemas.request import RequestInferred, RequestLaunched, RequestCompleted

class Infra(BaseModel, ABC):
    @abstractmethod
    def launch(self, request: RequestInferred) -> RequestLaunched:
        pass

    @abstractmethod
    def stop(self, request: RequestLaunched) -> None:
        pass

    @abstractmethod
    def status(self, request: RequestLaunched) -> RequestLaunched | RequestCompleted:
        pass
