from __future__ import annotations

import os
from pathlib import Path
from typing import cast

import pytest

from model_eval.config import EvalAdapterName, EvalRunConfig


pytestmark = [pytest.mark.live, pytest.mark.llm]


@pytest.mark.asyncio
async def test_manual_model_eval() -> None:
    if os.getenv("RUN_MODEL_EVAL") != "1":
        pytest.skip("set RUN_MODEL_EVAL=1 to run model eval")

    from model_eval.runner import run_eval_suite

    server_root = Path(__file__).resolve().parents[2]
    matrix_path = Path(
        os.getenv(
            "MODEL_EVAL_MATRIX",
            server_root / "evals" / "model_matrix.example.toml",
        )
    )
    adapter = os.getenv("MODEL_EVAL_ADAPTER", "simulated")
    if adapter not in {"simulated", "live-mcp"}:
        raise ValueError(f"unsupported MODEL_EVAL_ADAPTER: {adapter}")
    samples = int(os.getenv("MODEL_EVAL_SAMPLES", "1"))

    result = await run_eval_suite(
        EvalRunConfig(
            matrix_path=matrix_path,
            pack_name=os.getenv("MODEL_EVAL_PACK", "core_robot_commands"),
            adapter=cast(EvalAdapterName, adapter),
            mcp_url=os.getenv(
                "MODEL_EVAL_MCP_URL",
                "http://127.0.0.1:8765/mcp",
            ),
            samples=samples,
            evidence_root=server_root / "evidence" / "model_eval",
        )
    )

    assert result.attempts
    assert result.evidence_dir.exists()
    assert any(summary.correctness_passed for summary in result.summaries)
