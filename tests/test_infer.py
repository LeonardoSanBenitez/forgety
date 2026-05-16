import math
from libs import infer_request, Request
import pytest


def _req(**kwargs) -> Request:
    """Helper: build a Request with required fields pre-filled."""
    defaults = dict(
        customer_id="test-customer",
        experiment_name="test-experiment",
        model_base_name="test-model",
        concept_forget=None,
        concept_overwrite=None,
        concept_retain=None,
        unlearning_algorithm="UCE",
        model_output_hf_id="test-output-model",
    )
    defaults.update(kwargs)  # type: ignore[arg-type]
    return Request(**defaults)  # type: ignore[arg-type]


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


# ---------------------------------------------------------------------------
# Hyperparameter value tests
# ---------------------------------------------------------------------------

class TestFadeHyperparameters:
    """Test that FADE produces the expected hyperparameter values."""

    def test_learning_rate(self) -> None:
        result = infer_request(
            _req(concept_forget="cat", concept_overwrite="dog", unlearning_algorithm="FADE"),
            num_forget_images=10, num_retain_images=10,
        )
        assert result.hyperparameters is not None
        assert result.hyperparameters["learning_rate"] == 1e-4

    def test_epoch_formula_balanced(self) -> None:
        # 10 forget + 10 retain -> ceil(1600 / min(10,10)) = 160
        result = infer_request(
            _req(concept_forget="cat", concept_overwrite="dog", unlearning_algorithm="FADE"),
            num_forget_images=10, num_retain_images=10,
        )
        assert result.hyperparameters is not None
        assert result.hyperparameters["num_train_epochs"] == math.ceil(1600 / 10)

    def test_epoch_formula_bottleneck_is_forget(self) -> None:
        # 5 forget, 20 retain -> bottleneck is 5 -> ceil(1600/5) = 320
        result = infer_request(
            _req(concept_forget="cat", concept_overwrite="dog", unlearning_algorithm="FADE"),
            num_forget_images=5, num_retain_images=20,
        )
        assert result.hyperparameters is not None
        assert result.hyperparameters["num_train_epochs"] == math.ceil(1600 / 5)

    def test_epoch_formula_bottleneck_is_retain(self) -> None:
        # 20 forget, 5 retain -> bottleneck is 5 -> ceil(1600/5) = 320
        result = infer_request(
            _req(concept_forget="cat", concept_overwrite="dog", unlearning_algorithm="FADE"),
            num_forget_images=20, num_retain_images=5,
        )
        assert result.hyperparameters is not None
        assert result.hyperparameters["num_train_epochs"] == math.ceil(1600 / 5)

    def test_epoch_formula_very_small_dataset(self) -> None:
        # 1 image -> ceil(1600/1) = 1600 epochs
        result = infer_request(
            _req(concept_forget="cat", concept_overwrite="dog", unlearning_algorithm="FADE"),
            num_forget_images=1, num_retain_images=1,
        )
        assert result.hyperparameters is not None
        assert result.hyperparameters["num_train_epochs"] == 1600

    def test_epoch_formula_large_dataset(self) -> None:
        # 1601 images -> ceil(1600/1601) = 1 epoch minimum
        result = infer_request(
            _req(concept_forget="cat", concept_overwrite="dog", unlearning_algorithm="FADE"),
            num_forget_images=1601, num_retain_images=1601,
        )
        assert result.hyperparameters is not None
        assert result.hyperparameters["num_train_epochs"] == 1

    def test_overwrite_concept_in_hyperparameters(self) -> None:
        result = infer_request(
            _req(concept_forget="cat", concept_overwrite="leopard", unlearning_algorithm="FADE"),
            num_forget_images=10, num_retain_images=10,
        )
        assert result.hyperparameters is not None
        assert result.hyperparameters["overwritting_concept"] == "leopard"

    def test_validation_prompt_includes_forget_concept(self) -> None:
        result = infer_request(
            _req(concept_forget="cat", concept_overwrite="dog", unlearning_algorithm="FADE"),
            num_forget_images=10, num_retain_images=10,
        )
        assert result.hyperparameters is not None
        assert "cat" in result.hyperparameters["validation_prompt"]


class TestMunbaHyperparameters:
    """Test that Munba produces the expected hyperparameter values."""

    def test_learning_rate(self) -> None:
        result = infer_request(
            _req(unlearning_algorithm="Munba"),
            num_forget_images=10, num_retain_images=10,
        )
        assert result.hyperparameters is not None
        assert result.hyperparameters["learning_rate"] == 1e-4

    def test_epoch_formula_balanced(self) -> None:
        # 10 forget + 10 retain -> ceil(400 / min(10,10)) = 40
        result = infer_request(
            _req(unlearning_algorithm="Munba"),
            num_forget_images=10, num_retain_images=10,
        )
        assert result.hyperparameters is not None
        assert result.hyperparameters["num_train_epochs"] == math.ceil(400 / 10)

    def test_epoch_formula_very_small_dataset(self) -> None:
        # 1 image -> ceil(400/1) = 400 epochs
        result = infer_request(
            _req(unlearning_algorithm="Munba"),
            num_forget_images=1, num_retain_images=1,
        )
        assert result.hyperparameters is not None
        assert result.hyperparameters["num_train_epochs"] == 400

    def test_epoch_formula_large_dataset(self) -> None:
        # 401 images -> ceil(400/401) = 1 epoch
        result = infer_request(
            _req(unlearning_algorithm="Munba"),
            num_forget_images=401, num_retain_images=401,
        )
        assert result.hyperparameters is not None
        assert result.hyperparameters["num_train_epochs"] == 1


class TestUceHyperparameters:
    """Test that UCE produces the expected hyperparameter values."""

    def test_uce_edit_concept(self) -> None:
        result = infer_request(
            _req(concept_forget="cat", concept_retain="animal", unlearning_algorithm="UCE"),
            num_forget_images=0, num_retain_images=0,
        )
        assert result.hyperparameters is not None
        assert result.hyperparameters["edit_concepts"] == "cat"
        assert result.hyperparameters["preserve_concepts"] == "animal"

    def test_uce_erase_scale(self) -> None:
        result = infer_request(
            _req(concept_forget="cat", concept_retain="animal", unlearning_algorithm="UCE"),
            num_forget_images=0, num_retain_images=0,
        )
        assert result.hyperparameters is not None
        assert result.hyperparameters["erase_scale"] == 0.4
        assert result.hyperparameters["preserve_scale"] == 1.0

    def test_uce_no_num_train_epochs(self) -> None:
        """UCE does not have a num_train_epochs key."""
        result = infer_request(
            _req(concept_forget="cat", concept_retain="animal", unlearning_algorithm="UCE"),
            num_forget_images=0, num_retain_images=0,
        )
        assert result.hyperparameters is not None
        assert "num_train_epochs" not in result.hyperparameters
