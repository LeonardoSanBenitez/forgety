from libs.infra.base import Infra
from libs.schemas.request import RequestInferred, RequestLaunched, RequestCompleted


class InfraSageMaker(Infra):
    """Placeholder — not implemented."""

    def launch(self, request: RequestInferred) -> RequestLaunched:
        raise NotImplementedError

    def stop(self, request: RequestLaunched) -> None:
        raise NotImplementedError

    def status(self, request: RequestLaunched) -> RequestLaunched | RequestCompleted:
        raise NotImplementedError
