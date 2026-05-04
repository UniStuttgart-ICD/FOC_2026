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
from voice_runtime.agent_turn import AgentTurnProcessor

load_dotenv(override=True)


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

    @transport.event_handler("on_client_connected")
    async def on_client_connected(transport, client):
        logger.info("Client connected")
        if isinstance(agent_processor, AgentTurnProcessor):
            await agent_processor.connect()

    @transport.event_handler("on_client_disconnected")
    async def on_client_disconnected(transport, client):
        logger.info("Client disconnected")
        if isinstance(agent_processor, AgentTurnProcessor):
            await agent_processor.disconnect()
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
    await runner.run(task)


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
    from pipecat.runner.run import main

    parser = argparse.ArgumentParser(add_help=False)
    parser.add_argument("--profile", dest="profile_name", default=None)
    known, remaining = parser.parse_known_args()
    if known.profile_name:
        os.environ["VOICE_PROFILE"] = known.profile_name
        sys.argv = [sys.argv[0], *remaining]
    main()
