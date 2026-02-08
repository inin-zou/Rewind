import asyncio
import base64
import json
import logging
import os
from dataclasses import dataclass
from typing import Any, AsyncIterator, Optional

import gradium
import websockets
from dotenv import load_dotenv
from openai import OpenAI


@dataclass(frozen=True)
class GatewayConfig:
    gradium_api_key: str
    openai_api_key: str
    host: str
    port: int
    stt_model: str
    stt_input_format: str
    tts_model: str
    tts_voice_id: str
    tts_output_format: str
    openai_model: str
    openai_prompt: str


def load_config() -> GatewayConfig:
    load_dotenv()
    gradium_api_key = os.getenv("GRADIUM_API_KEY")
    openai_api_key = os.getenv("OPENAI_API_KEY")
    if not gradium_api_key:
        raise RuntimeError("Missing GRADIUM_API_KEY")
    if not openai_api_key:
        raise RuntimeError("Missing OPENAI_API_KEY")

    return GatewayConfig(
        gradium_api_key=gradium_api_key,
        openai_api_key=openai_api_key,
        host=os.getenv("GATEWAY_HOST", "0.0.0.0"),
        port=int(os.getenv("GATEWAY_PORT", "8765")),
        stt_model=os.getenv("GRADIUM_STT_MODEL", "default"),
        stt_input_format=os.getenv("GRADIUM_STT_INPUT_FORMAT", "pcm"),
        tts_model=os.getenv("GRADIUM_TTS_MODEL", "default"),
        tts_voice_id=os.getenv("GRADIUM_TTS_VOICE_ID", "YTpq7expH9539ERJ"),
        tts_output_format=os.getenv("GRADIUM_TTS_OUTPUT_FORMAT", "pcm"),
        openai_model=os.getenv("OPENAI_MODEL", "gpt-4o-mini"),
        openai_prompt=os.getenv(
            "OPENAI_PROMPT",
            "You're an AI therapist.",
        ),
    )


class AudioQueue:
    def __init__(self) -> None:
        self._queue: asyncio.Queue[Optional[bytes]] = asyncio.Queue()

    async def put(self, data: bytes) -> None:
        await self._queue.put(data)

    async def close(self) -> None:
        await self._queue.put(None)

    async def generator(self) -> AsyncIterator[bytes]:
        while True:
            item = await self._queue.get()
            if item is None:
                break
            yield item


async def recv_start_message(ws: websockets.WebSocketServerProtocol) -> dict[str, Any]:
    raw = await ws.recv()
    if isinstance(raw, bytes):
        raise RuntimeError("Expected JSON start message before binary audio frames")
    try:
        msg = json.loads(raw)
    except json.JSONDecodeError as exc:
        raise RuntimeError("Invalid JSON start message") from exc
    if msg.get("type") != "start":
        raise RuntimeError("First message must be a start message")
    return msg


async def send_json(ws: websockets.WebSocketServerProtocol, payload: dict[str, Any]) -> None:
    await ws.send(json.dumps(payload))


async def receive_audio(
    ws: websockets.WebSocketServerProtocol,
    audio_queue: AudioQueue,
) -> None:
    async for raw in ws:
        if isinstance(raw, bytes):
            logging.debug("Received %d bytes of audio", len(raw))
            await audio_queue.put(raw)
            continue

        try:
            msg = json.loads(raw)
        except json.JSONDecodeError:
            await send_json(ws, {"type": "error", "message": "Invalid JSON message"})
            continue

        msg_type = msg.get("type")
        if msg_type == "audio":
            payload = msg.get("audio")
            if not payload:
                continue
            try:
                decoded = base64.b64decode(payload)
                logging.debug("Received %d bytes of base64 audio", len(decoded))
                await audio_queue.put(decoded)
            except (TypeError, ValueError):
                await send_json(ws, {"type": "error", "message": "Invalid base64 audio payload"})
        elif msg_type in {"end", "end_of_stream"}:
            break
        else:
            await send_json(ws, {"type": "error", "message": f"Unknown message type: {msg_type}"})

    await audio_queue.close()


def extract_text_message(message: Any) -> Optional[dict[str, Any]]:
    if isinstance(message, dict):
        return message
    if isinstance(message, str):
        try:
            parsed = json.loads(message)
            if isinstance(parsed, dict):
                return parsed
        except json.JSONDecodeError:
            return None
    text = getattr(message, "text", None)
    if isinstance(text, str) and text:
        return {"type": "text", "text": text, "_raw": message}
    return None


async def run_stt(
    gradium_client: gradium.client.GradiumClient,
    audio_queue: AudioQueue,
    final_texts: asyncio.Queue[Optional[str]],
    ws: websockets.WebSocketServerProtocol,
    config: GatewayConfig,
) -> None:
    logging.info("Starting STT stream (model=%s format=%s)", config.stt_model, config.stt_input_format)
    stream = await gradium_client.stt_stream(
        {"model_name": config.stt_model, "input_format": config.stt_input_format},
        audio_queue.generator(),
    )

    last_text: Optional[str] = None
    last_update_id = 0

    async def flush_after_silence(update_id: int) -> None:
        await asyncio.sleep(2)
        nonlocal last_text
        if update_id == last_update_id and last_text:
            logging.info("STT final (silence): %s", last_text)
            await send_json(ws, {"type": "stt_final", "text": last_text})
            await final_texts.put(last_text)
            last_text = None

    async for message in stream.iter_text():
        payload = extract_text_message(message)
        if not payload:
            logging.info("STT stream message (raw): %s", message)
            continue

        msg_type = payload.get("type")
        if msg_type == "text":
            text = payload.get("text", "")
            if text:
                logging.debug("STT partial: %s", text)
                last_text = text
                last_update_id += 1
                asyncio.create_task(flush_after_silence(last_update_id))
                await send_json(ws, {"type": "stt_partial", "text": text})
        elif msg_type == "end_text":
            if last_text:
                logging.info("STT final: %s", last_text)
                await send_json(ws, {"type": "stt_final", "text": last_text})
                await final_texts.put(last_text)
                last_text = None
                last_update_id += 1
        elif msg_type == "end_of_stream":
            logging.info("STT stream ended")
            break
        elif msg_type == "error":
            logging.error("STT error: %s", payload.get("message"))
            await send_json(ws, {"type": "error", "message": payload.get("message", "STT error")})

    if last_text:
        logging.info("STT final (stream end): %s", last_text)
        await send_json(ws, {"type": "stt_final", "text": last_text})
        await final_texts.put(last_text)

    await final_texts.put(None)


def transform_text_sync(openai_client: OpenAI, config: GatewayConfig, text: str) -> str:
    response = openai_client.responses.create(
        model=config.openai_model,
        input=[
            {
                "role": "system",
                "content": [{"type": "input_text", "text": config.openai_prompt}],
            },
            {
                "role": "user",
                "content": [{"type": "input_text", "text": text}],
            },
        ],
        max_output_tokens=512,
        temperature=0.3,
    )
    output_text = getattr(response, "output_text", None)
    if output_text:
        return output_text
    # Fallback for older SDK response shapes
    try:
        return response.output[0].content[0].text
    except Exception:
        return text


async def run_tts(
    gradium_client: gradium.client.GradiumClient,
    ws: websockets.WebSocketServerProtocol,
    config: GatewayConfig,
    text: str,
) -> None:
    logging.info("Starting TTS stream (model=%s voice=%s format=%s)", config.tts_model, config.tts_voice_id, config.tts_output_format)
    stream = await gradium_client.tts_stream(
        setup={
            "model_name": config.tts_model,
            "voice_id": config.tts_voice_id,
            "output_format": config.tts_output_format,
        },
        text=text,
    )

    await send_json(
        ws,
        {
            "type": "tts_start",
            "format": config.tts_output_format,
            "sample_rate": 48000,
        },
    )

    async for audio_chunk in stream.iter_bytes():
        logging.debug("TTS chunk: %d bytes", len(audio_chunk))
        await ws.send(audio_chunk)

    await send_json(ws, {"type": "tts_end"})


async def run_transform_and_tts(
    openai_client: OpenAI,
    gradium_client: gradium.client.GradiumClient,
    final_texts: asyncio.Queue[Optional[str]],
    ws: websockets.WebSocketServerProtocol,
    config: GatewayConfig,
) -> None:
    while True:
        text = await final_texts.get()
        if text is None:
            break

        logging.info("Transforming text (len=%d)", len(text))
        transformed = await asyncio.to_thread(transform_text_sync, openai_client, config, text)
        logging.info("Transformed text (len=%d)", len(transformed))
        logging.info("Transformed text: %s", transformed)
        await send_json(ws, {"type": "text_final", "text": transformed})
        await run_tts(gradium_client, ws, config, transformed)


async def handler(ws: websockets.WebSocketServerProtocol) -> None:
    config = load_config()
    gradium_client = gradium.client.GradiumClient(api_key=config.gradium_api_key)
    openai_client = OpenAI(api_key=config.openai_api_key)

    try:
        start_msg = await recv_start_message(ws)
    except Exception as exc:
        await send_json(ws, {"type": "error", "message": str(exc)})
        return

    logging.info("Client start message: %s", start_msg)

    stt_format = start_msg.get("input_format") or start_msg.get("format") or config.stt_input_format
    config = GatewayConfig(
        gradium_api_key=config.gradium_api_key,
        openai_api_key=config.openai_api_key,
        host=config.host,
        port=config.port,
        stt_model=start_msg.get("stt_model", config.stt_model),
        stt_input_format=stt_format,
        tts_model=start_msg.get("tts_model", config.tts_model),
        tts_voice_id=start_msg.get("voice_id", config.tts_voice_id),
        tts_output_format=start_msg.get("output_format", config.tts_output_format),
        openai_model=start_msg.get("openai_model", config.openai_model),
        openai_prompt=start_msg.get("prompt", config.openai_prompt),
    )

    await send_json(
        ws,
        {
            "type": "ready",
            "stt_input_format": config.stt_input_format,
            "tts_output_format": config.tts_output_format,
        },
    )

    audio_queue = AudioQueue()
    final_texts: asyncio.Queue[Optional[str]] = asyncio.Queue()

    receive_task = asyncio.create_task(receive_audio(ws, audio_queue))
    stt_task = asyncio.create_task(run_stt(gradium_client, audio_queue, final_texts, ws, config))
    tts_task = asyncio.create_task(
        run_transform_and_tts(openai_client, gradium_client, final_texts, ws, config)
    )

    done, pending = await asyncio.wait(
        {receive_task, stt_task, tts_task},
        return_when=asyncio.FIRST_EXCEPTION,
    )

    for task in pending:
        task.cancel()

    for task in done:
        if task.exception():
            logging.exception("Gateway task failed", exc_info=task.exception())


async def main() -> None:
    config = load_config()
    log_level = os.getenv("LOG_LEVEL", "INFO").upper()
    logging.basicConfig(level=log_level)

    async with websockets.serve(handler, config.host, config.port, max_size=None):
        logging.info("Gateway listening on ws://%s:%s", config.host, config.port)
        await asyncio.Future()


if __name__ == "__main__":
    asyncio.run(main())
