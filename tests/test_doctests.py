"""Execute the doctests embedded in public docstrings.

Documented examples are part of the user-facing contract, so they are run as tests to
keep them from going stale.
"""

import doctest

import pytest

from pyTOST import results, workflow

MODULES = [workflow, results]


@pytest.mark.parametrize("module", MODULES, ids=lambda m: m.__name__)
def test_module_doctests(module):
    failures, attempted = doctest.testmod(
        module, optionflags=doctest.NORMALIZE_WHITESPACE, verbose=False
    )
    assert failures == 0, f"{failures} doctest failure(s) in {module.__name__}"


def test_run_tost_docstring_documents_every_parameter():
    """Each keyword parameter must appear in the docstring's Parameters section."""
    import inspect

    doc = inspect.getdoc(workflow.run_tost) or ""
    params = inspect.signature(workflow.run_tost).parameters

    missing = [name for name in params if name not in doc]
    assert not missing, f"run_tost docstring omits parameters: {missing}"


def test_run_tost_docstring_documents_the_result_schema():
    import inspect

    doc = inspect.getdoc(workflow.run_tost) or ""

    for key in ("engine", "primary", "sensitivity", "bootstrap"):
        assert key in doc

    for column in ("delta", "mu_hat", "ci_low", "ci_high", "equivalent", "method"):
        assert column in doc, f"result column {column!r} is undocumented"
