.PHONY: test lint format format-check typecheck build

all: test lint format typecheck

test:
	$(info ****************** running tests ******************)
	uv run --all-packages pytest tests

lint:
	$(info ****************** linting ******************)
	uv run --all-packages pre-commit run -a

format:
	$(info ****************** formatting ******************)
	uv run --all-packages ruff format

format-check:
	$(info ****************** checking formatting ******************)
	uv run --all-packages ruff format --check

typecheck:
	$(info ****************** type checking ******************)
	uv run --all-packages ty check shared/src/fs_explorer_shared/
	uv run --all-packages ty check api/src/fs_explorer_api/
	uv run --all-packages ty check indexer/src/fs_explorer_indexer/

build:
	$(info ****************** building ******************)
	uv build --all-packages
