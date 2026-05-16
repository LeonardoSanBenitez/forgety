"""
InfraFake — deterministic test double for the Infra ABC.

Behaviour:
- launch(): immediately returns RequestLaunched with a fixed job ID and timestamp.
- status(): always returns RequestCompleted with status=SUCCEEDED.
- stop(): no-op.

This lives in libs/ so it can be imported by tests and potentially by
future integration harnesses without depending on the test runner.
"""
from libs.infra.base import Infra
from libs.schemas.request import RequestInferred, RequestLaunched, RequestCompleted

_FAKE_JOB_ID = "fake-job-000"
_FAKE_TIMESTAMP = 1_700_000_000


class InfraFake(Infra):
    """Synchronous, side-effect-free test double for Infra."""

    def launch(self, request: RequestInferred) -> RequestLaunched:
        result = request.model_dump()
        result.update({
            'timestamp_started': _FAKE_TIMESTAMP,
            'slurm_job_id': _FAKE_JOB_ID,
        })
        return RequestLaunched(**result)

    def stop(self, request: RequestLaunched) -> None:
        return None

    def status(self, request: RequestLaunched) -> RequestLaunched | RequestCompleted:
        result = request.model_dump()
        result.update({
            'status': 'SUCCEEDED',
            'metrics': [{'name': 'clip', 'value': 0.95}],
            'credits_consumed': 1.0,
        })
        return RequestCompleted(**result)
