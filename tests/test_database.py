from libs.database import DatabaseLocalJson
from libs.schemas import Request
import tempfile
import os
import pytest 


def test_all():
    with tempfile.NamedTemporaryFile(suffix=".json", delete=False) as tmpfile:
        if os.path.exists(tmpfile.name):
            os.remove(tmpfile.name)
        database = DatabaseLocalJson(filepath=tmpfile.name)

        result = database.get_requests()
        print(result)
        assert len(result) == 0
        

        request = Request(
            customer_id="123",
            experiment_name="test_experiment",
            model_base_name='SD',
            concept_forget='parachute',
            concept_overwrite='airplane',
            unlearning_algorithm='FADE',
            model_output_hf_id='somewhere',
        )
        database.insert_request(request)

        result = database.get_request(uuid=request.uuid)
        print(result)

        result = database.get_requests()
        assert len(result) == 1
        print(result)

        result = database.get_requests_status_none()
        assert len(result) == 1

        database.update_request(uuid=request.uuid, updated_request={'status': 'SUCCEEDED'})
        result = database.get_request(uuid=request.uuid)
        assert result.status == 'SUCCEEDED'  # TODO I should know the Request stage to be returned??? Maybe save everythign as last stage?
        print(result)

        result = database.get_requests()
        assert len(result) == 1
        print(result)

        result = database.get_requests_status_none()
        assert len(result) == 0
        
        if os.path.exists(tmpfile.name):
            os.remove(tmpfile.name)