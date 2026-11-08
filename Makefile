# After `make setup`, targets run inside the uv-managed environment.
# Override to use another interpreter, e.g.:  make test PYTHON=python
PYTHON ?= uv run python

.PHONY: setup test eval-smoke eval mcp

setup:            ## install deps (generates uv.lock on first run)
	uv sync --extra dev

test:             ## offline unit + e2e tests (no network, no API keys)
	$(PYTHON) -m pytest -q

eval-smoke:       ## 5 scenarios, k=1, both variants (needs API keys + AGENT_MODEL/USER_MODEL in .env)
	$(PYTHON) -m evals.run --scenarios smoke --k 1 --variant both --out results/smoke

eval:             ## all 20 scenarios, k=4, both variants; writes results/ and fills the README results block
	$(PYTHON) -m evals.run --scenarios all --k 4 --variant both --out results --update-readme

mcp:              ## run the tool server over stdio
	$(PYTHON) -m tools.mcp_server
