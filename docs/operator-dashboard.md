# MAVE operator dashboard

Start the MAVE operator dashboard from the repo root:

```powershell
.\Start-MAVE-Workshop.cmd
```

The launcher prints the localhost URL with its `token` query parameter in the same `.cmd` window and opens it in the browser. Keep the window open while the dashboard is running.

The dashboard can start and monitor the repo-local Vizor + RViz Compose stack, MoveIt MCP, Vizor MCP, the verified execution server, and the Pipecat voice agent.

Service commands are configured in `configs/operator_dashboard.example.toml`. Put machine-specific values such as a physical robot IP in `configs/operator_dashboard.local.toml`; that file is ignored.
