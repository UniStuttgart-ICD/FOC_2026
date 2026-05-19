#
# Copyright (c) 2024-2025, Daily
#
# SPDX-License-Identifier: BSD 2-Clause License
#

"""pipecat-agent - Voice-Controlled UR Robot Agent.

Run the bot::

    uv run bot.py
"""

import argparse
import os
import sys
from pathlib import Path

from dotenv import load_dotenv
from loguru import logger
from pipecat.pipeline.runner import PipelineRunner
from pipecat.processors.aggregators.llm_response_universal import (
    AssistantTurnStoppedMessage,
    UserTurnStoppedMessage,
)
from pipecat.runner.types import RunnerArguments, SmallWebRTCRunnerArguments
from pipecat.transports.base_transport import BaseTransport, TransportParams
from pipecat.transports.smallwebrtc.connection import SmallWebRTCConnection
from pipecat.transports.smallwebrtc.transport import SmallWebRTCTransport

from config import load_runtime_config
from pipeline_builder import build_pipeline
from robot_control.job_monitor import start_robot_job_monitor_from_env
from voice_runtime.agent_turn import AgentTurnProcessor

load_dotenv(override=True)


def _patch_small_webrtc_prebuilt_client_history() -> None:
    """Keep the stock /client UI from coalescing separate fast turns."""
    try:
        import pipecat_ai_small_webrtc_prebuilt
    except ImportError:
        return

    package_dir = Path(pipecat_ai_small_webrtc_prebuilt.__file__).resolve().parent
    assets_dir = package_dir / "client" / "dist" / "assets"
    js_files = sorted(assets_dir.glob("index-*.js"))
    if not js_files:
        logger.warning("Small WebRTC prebuilt client asset not found at {}", assets_dir)
        return

    replacements = {
        "const Az=3e4": "const Az=0",
        "&&o<3e4?": "&&o<0?",
        "const qz=2500": "const qz=0",
        "},3e3)},[])))": "},0)},[])))",
    }
    patched = False
    for js_path in js_files:
        try:
            text = js_path.read_text(encoding="utf-8")
        except OSError as exc:
            logger.warning("Could not read Small WebRTC client asset {}: {}", js_path, exc)
            continue

        updated = text
        for old, new in replacements.items():
            updated = updated.replace(old, new)

        if updated == text:
            continue

        try:
            js_path.write_text(updated, encoding="utf-8")
        except OSError as exc:
            logger.warning("Could not patch Small WebRTC client asset {}: {}", js_path, exc)
            continue

        patched = True
        logger.info("Patched Small WebRTC prebuilt client history behavior in {}", js_path)

    if not patched:
        logger.debug("Small WebRTC prebuilt client history behavior already patched")


async def run_bot(transport: BaseTransport, profile_name: str | None = None):
    """Main bot logic."""
    config = load_runtime_config(profile_name=profile_name)
    logger.info(
        "Starting voice robot agent profile={} category={} stt={} tts={} agent={}",
        config.profile_name,
        config.category,
        config.stt.provider,
        config.tts.provider,
        config.agent.provider,
    )

    built = build_pipeline(config, transport)
    task = built.task
    agent_processor = built.agent_processor
    embodiment = built.embodiment
    robot_job_monitor = await start_robot_job_monitor_from_env(
        getattr(agent_processor, "robot_job_board", None)
    )

    @transport.event_handler("on_client_connected")
    async def on_client_connected(transport, client):
        logger.info("Client connected")
        if embodiment is not None:
            await embodiment.start()
        if isinstance(agent_processor, AgentTurnProcessor):
            try:
                await agent_processor.connect()
            except Exception as exc:
                logger.warning(
                    "Agent backend was not ready during client setup: {}",
                    exc,
                )

    @transport.event_handler("on_client_disconnected")
    async def on_client_disconnected(transport, client):
        logger.info("Client disconnected")
        if isinstance(agent_processor, AgentTurnProcessor):
            try:
                await agent_processor.disconnect()
            except Exception:
                logger.exception("Agent backend disconnect failed")
        if embodiment is not None:
            await embodiment.stop()
        await task.cancel()

    @built.user_aggregator.event_handler("on_user_turn_stopped")
    async def on_user_turn_stopped(aggregator, strategy, message: UserTurnStoppedMessage):
        timestamp = f"[{message.timestamp}] " if message.timestamp else ""
        logger.info(f"Transcript: {timestamp}user: {message.content}")

    @built.assistant_aggregator.event_handler("on_assistant_turn_stopped")
    async def on_assistant_turn_stopped(aggregator, message: AssistantTurnStoppedMessage):
        timestamp = f"[{message.timestamp}] " if message.timestamp else ""
        logger.info(f"Transcript: {timestamp}assistant: {message.content}")

    runner = PipelineRunner(handle_sigint=False)
    try:
        await runner.run(task)
    finally:
        if robot_job_monitor is not None:
            await robot_job_monitor.stop()
        if isinstance(agent_processor, AgentTurnProcessor):
            try:
                await agent_processor.disconnect()
            except Exception:
                logger.exception("Agent backend disconnect failed during shutdown")
        if embodiment is not None:
            await embodiment.stop()


async def bot(runner_args: RunnerArguments):
    """Main bot entry point."""
    transport = None

    match runner_args:
        case SmallWebRTCRunnerArguments():
            webrtc_connection: SmallWebRTCConnection = runner_args.webrtc_connection
            transport = SmallWebRTCTransport(
                webrtc_connection=webrtc_connection,
                params=TransportParams(
                    audio_in_enabled=True,
                    audio_out_enabled=True,
                ),
            )
        case _:
            logger.error(f"Unsupported runner arguments type: {type(runner_args)}")
            return

    await run_bot(transport, profile_name=os.getenv("VOICE_PROFILE"))


if __name__ == "__main__":
    _patch_small_webrtc_prebuilt_client_history()

    from pipecat.runner.run import main

    parser = argparse.ArgumentParser(add_help=False)
    parser.add_argument("--profile", dest="profile_name", default=None)
    known, remaining = parser.parse_known_args()
    if known.profile_name:
        os.environ["VOICE_PROFILE"] = known.profile_name
        sys.argv = [sys.argv[0], *remaining]
    main()
