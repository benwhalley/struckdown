"""Fixtures for the contrib tests: a priced model, and a clean span context."""

import pytest

from struckdown.contrib.django import spans
from struckdown.llm import set_model_pricing


@pytest.fixture(autouse=True)
def _clean_context():
    """No span or pricing leaks from one test into the next."""
    spans._current.set(None)
    set_model_pricing(None, None)
    from struckdown.ledger import set_model_ref

    set_model_ref(None)
    yield
    spans._current.set(None)
    set_model_pricing(None, None)
    set_model_ref(None)


@pytest.fixture
def credential(db):
    from struckdown.contrib.django.models import Credential

    return Credential.objects.create(
        name="Test", api_key="sk-test", base_url="http://example.invalid/v1"
    )


@pytest.fixture
def priced_model(credential):
    from struckdown.contrib.django.models import AvailableModel

    return AvailableModel.objects.create(
        model_name="stub",
        model_type="llm",
        name="Stub",
        credential=credential,
        input_cost_per_mtok="2.000000",
        output_cost_per_mtok="8.000000",
        cache_read_cost_per_mtok="0.200000",
        prices_updated_manually=True,
        data_residency="eu",
    )
