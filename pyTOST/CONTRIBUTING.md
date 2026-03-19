# Contributing

Thank you for your interest in improving pyTOST.

## Reporting issues

Please use the repository issue tracker to report bugs, installation problems, documentation errors, or feature requests.

When reporting a bug, include:

- your Python version and operating system
- the pyTOST version or commit hash
- a minimal reproducible example
- the full traceback or error message

## Seeking support

For usage questions, open an issue and label it as a question or support request if your issue template setup supports labels.

## Contributing code or documentation

Contributions are welcome for:

- bug fixes
- tests
- documentation improvements
- notebook cleanup and examples
- engine diagnostics and validation workflows
- additional engines

Recommended workflow:

1. Fork the repository.
2. Create a feature branch.
3. Add or update tests for the change when appropriate.
4. Run the test suite locally.
5. Open a pull request with a concise description of the change.

## Development setup

Install the package in editable mode with test dependencies:

```bash
pip install -e .[test]
pytest -q
```

## Code style

Prefer clear, explicit statistical code and keep public-facing terminology generic unless a domain-specific example is required.

## Scope notes

pyTOST is a library-first package. Please avoid adding CLI-oriented workflows unless they are explicitly discussed and accepted as part of the project roadmap.
