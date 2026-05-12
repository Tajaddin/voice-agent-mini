"""``voice-agent`` CLI — runs a synthetic round against mock backends."""

from __future__ import annotations

import asyncio

import click

from voice_agent.audio import generate_silence, generate_speech
from voice_agent.backends import MockLLM, MockSTT, MockTTS
from voice_agent.pipeline import VoicePipeline


async def _stream(chunks):
    for c in chunks:
        yield c
        # tiny sleep so producer/consumer overlap looks realistic
        await asyncio.sleep(0)


async def main_async():
    pipeline = VoicePipeline(MockSTT(), MockLLM(), MockTTS())
    audio = (
        generate_silence(5)
        + generate_speech(30, start_ms=100)
        + generate_silence(15, start_ms=700)
    )
    trace = await pipeline.turn(_stream(audio))
    click.echo(f"transcript: {trace.transcript}")
    click.echo(f"reply:      {trace.reply_text}")
    click.echo(f"voice-to-voice ms: {trace.voice_to_voice_ms}")
    click.echo(f"per-stage latencies: {dict(trace.latency.actuals)}")
    click.echo(f"violations: {[str(v) for v in trace.latency.violations()]}")


@click.group()
def cli():
    """voice-agent-mini CLI."""


@cli.command()
def demo():
    """Run one mock turn end-to-end and print the trace."""
    asyncio.run(main_async())


if __name__ == "__main__":
    cli()
