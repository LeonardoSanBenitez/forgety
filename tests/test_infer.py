from libs import infer_request, Request
import tempfile
import os
import pytest 


def test_auto():
    result = infer_request(Request(
        customer_id="test-customer",
        experiment_name="test-experiment",
        model_base_name="test-model",
        concept_forget=None,
        concept_overwrite=None,
        concept_retain=None,
        unlearning_algorithm="Automatic (recommended)",
        model_output_hf_id="test-output-model"
    ), num_forget_images=10, num_retain_images=10)
    assert result.unlearning_algorithm == "Munba"

    result = infer_request(Request(
        customer_id="test-customer",
        experiment_name="test-experiment",
        model_base_name="test-model",
        concept_forget="cat",
        concept_overwrite="dog",
        concept_retain=None,
        unlearning_algorithm="Automatic (recommended)",
        model_output_hf_id="test-output-model"
    ), num_forget_images=10, num_retain_images=10)
    assert result.unlearning_algorithm == "FADE"

    result = infer_request(Request(
        customer_id="test-customer",
        experiment_name="test-experiment",
        model_base_name="test-model",
        concept_forget="cat",
        concept_overwrite=None,
        concept_retain="animal",
        unlearning_algorithm="Automatic (recommended)",
        model_output_hf_id="test-output-model"
    ), num_forget_images=0, num_retain_images=0)
    assert result.unlearning_algorithm == "UCE"

    with pytest.raises(Exception):  
        result = infer_request(Request(
            customer_id="test-customer",
            experiment_name="test-experiment",
            model_base_name="test-model",
            concept_forget="cat",
            concept_overwrite=None,
            concept_retain=None,
            unlearning_algorithm="Automatic (recommended)",
            model_output_hf_id="test-output-model"
        ), num_forget_images=0, num_retain_images=0)

    with pytest.raises(Exception):  
        result = infer_request(Request(
            customer_id="test-customer",
            experiment_name="test-experiment",
            model_base_name="test-model",
            concept_forget="cat",
            concept_overwrite="dog",
            concept_retain=None,
            unlearning_algorithm="Automatic (recommended)",
            model_output_hf_id="test-output-model"
        ), num_forget_images=0, num_retain_images=0)


def test_fade():
    infer_request(Request(
        customer_id="test-customer",
        experiment_name="test-experiment",
        model_base_name="test-model",
        concept_forget="cat",
        concept_overwrite="dog",
        concept_retain=None,
        unlearning_algorithm="FADE",
        model_output_hf_id="test-output-model"
    ), num_forget_images=10, num_retain_images=10)

    with pytest.raises(Exception):    
        infer_request(Request(
            customer_id="test-customer",
            experiment_name="test-experiment",
            model_base_name="test-model",
            concept_forget="cat",
            concept_overwrite=None,
            concept_retain=None,
            unlearning_algorithm="FADE",
            model_output_hf_id="test-output-model"
        ), num_forget_images=10, num_retain_images=10)

    with pytest.raises(Exception):    
        infer_request(Request(
            customer_id="test-customer",
            experiment_name="test-experiment",
            model_base_name="test-model",
            concept_forget=None,
            concept_overwrite="dog",
            concept_retain=None,
            unlearning_algorithm="FADE",
            model_output_hf_id="test-output-model"
        ), num_forget_images=10, num_retain_images=10)

    with pytest.raises(Exception):    
        infer_request(Request(
            customer_id="test-customer",
            experiment_name="test-experiment",
            model_base_name="test-model",
            concept_forget="cat",
            concept_overwrite="dog",
            concept_retain=None,
            unlearning_algorithm="FADE",
            model_output_hf_id="test-output-model"
        ), num_forget_images=0, num_retain_images=0)


def test_uce():
    infer_request(Request(
        customer_id="test-customer",
        experiment_name="test-experiment",
        model_base_name="test-model",
        concept_forget="cat",
        concept_overwrite=None,
        concept_retain="animal",
        unlearning_algorithm="UCE",
        model_output_hf_id="test-output-model"
    ), num_forget_images=None, num_retain_images=None)

    with pytest.raises(Exception):
        infer_request(Request(
            customer_id="test-customer",
            experiment_name="test-experiment",
            model_base_name="test-model",
            concept_forget=None,
            concept_overwrite=None,
            concept_retain="animal",
            unlearning_algorithm="UCE",
            model_output_hf_id="test-output-model"
        ), num_forget_images=None, num_retain_images=None)


    with pytest.raises(Exception):
        infer_request(Request(
            customer_id="test-customer",
            experiment_name="test-experiment",
            model_base_name="test-model",
            concept_forget="cat",
            concept_overwrite=None,
            concept_retain=None,
            unlearning_algorithm="UCE",
            model_output_hf_id="test-output-model"
        ), num_forget_images=None, num_retain_images=None)


def test_munba():
    infer_request(Request(
        customer_id="test-customer",
        experiment_name="test-experiment",
        model_base_name="test-model",
        concept_forget=None,
        concept_overwrite=None,
        concept_retain=None,
        unlearning_algorithm="Munba",
        model_output_hf_id="test-output-model"
    ), num_forget_images=10, num_retain_images=10)

    with pytest.raises(Exception):
        infer_request(Request(
            customer_id="test-customer",
            experiment_name="test-experiment",
            model_base_name="test-model",
            concept_forget=None,
            concept_overwrite=None,
            concept_retain=None,
            unlearning_algorithm="Munba",
            model_output_hf_id="test-output-model"
        ), num_forget_images=0, num_retain_images=0)
