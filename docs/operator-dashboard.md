# MAVE operator dashboard

Start the MAVE operator dashboard from the Multi-Actor repository root:

```powershell
cd C:\Users\Samuel\Documents\github\Multi-Actor-Interface-Library
uv run python scripts/run_operator_dashboard.py
```

Open the printed localhost URL with its `token` query parameter. The dashboard can start and monitor Vizor + RViz, MoveIt MCP, and the Pipecat voice agent.

The dashboard service commands are configured in `C:\Users\Samuel\Documents\github\Multi-Actor-Interface-Library\configs\operator_dashboard.example.toml` or the local override `configs\operator_dashboard.local.toml`.
