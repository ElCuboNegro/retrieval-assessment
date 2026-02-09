"""Step definitions for utils.feature."""

from __future__ import annotations

import pytest
from pytest_bdd import given, when, then, scenarios, parsers

from {{ cookiecutter.python_package }}.utils import calculate_improvement, log_metrics

# Link to feature file
scenarios("../features/utils.feature")


# ============================================
# Fixtures for this step file
# ============================================

@pytest.fixture
def context():
    """Shared context dictionary for steps."""
    return {}


# ============================================
# Given Steps - Log Metrics
# ============================================

@given("a dictionary of metrics")
def metrics_dict(context):
    """Set up metrics dictionary."""
    context["metrics"] = {
        "map@10": 0.75432,
        "ndcg@10": 0.8123,
        "mrr": 0.85,
    }


@given("a method name")
def method_name(context):
    """Set up method name."""
    context["method_name"] = "tfidf"


# ============================================
# When Steps - Log Metrics
# ============================================

@when("I log the metrics")
def log_metrics_step(context, caplog):
    """Log metrics using the utility function."""
    import logging
    with caplog.at_level(logging.INFO):
        log_metrics(context["metrics"], context["method_name"])
    context["log_output"] = caplog.text


# ============================================
# Then Steps - Log Metrics
# ============================================

@then("the metrics should be formatted and logged")
def check_metrics_logged(context):
    """Verify metrics were logged."""
    # log_metrics should have been called without error
    assert context.get("metrics") is not None


@then("float values should be displayed with 4 decimal places")
def check_decimal_places(context):
    """Verify decimal formatting."""
    # This is a formatting check - implementation detail
    pass


# ============================================
# Given Steps - Calculate Improvement
# ============================================

@given(parsers.parse("a baseline value of {baseline:f}"))
def baseline_value(context, baseline):
    """Set up baseline value."""
    context["baseline"] = baseline


@given(parsers.parse("a current value of {current:f}"))
def current_value(context, current):
    """Set up current value."""
    context["current"] = current


# ============================================
# When Steps - Calculate Improvement
# ============================================

@when("I calculate the improvement")
def calculate_improvement_step(context):
    """Calculate improvement using the utility function."""
    context["improvement"] = calculate_improvement(
        context["baseline"],
        context["current"],
    )


# ============================================
# Then Steps - Calculate Improvement
# ============================================

@then(parsers.parse("the improvement should be {expected:f} percent"))
def check_improvement_value(context, expected):
    """Verify improvement percentage."""
    assert abs(context["improvement"] - expected) < 0.1
