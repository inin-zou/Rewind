import asyncio
import base64
import json
import logging
import os
from typing import Any, AsyncIterator, Optional

import gradium
import websockets
from openai import OpenAI

from config import GatewayConfig, load_config


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


def validate_start_message(start_msg: dict[str, Any], config: GatewayConfig) -> None:
    sample_rate = start_msg.get("sample_rate")
    channels = start_msg.get("channels")
    input_format = start_msg.get("input_format")

    if sample_rate is None or channels is None or input_format is None:
        raise RuntimeError(
            "Start message must include sample_rate, channels, and input_format for validation."
        )

    if sample_rate != 24000:
        raise RuntimeError(f"Unsupported sample_rate {sample_rate}; expected 24000.")
    if channels != 1:
        raise RuntimeError(f"Unsupported channels {channels}; expected 1.")
    if input_format != config.stt_input_format:
        raise RuntimeError(
            f"Unsupported input_format {input_format}; expected {config.stt_input_format}."
        )


async def send_json(ws: websockets.WebSocketServerProtocol, payload: dict[str, Any]) -> None:
    await ws.send(json.dumps(payload))


async def receive_client_messages(
    ws: websockets.WebSocketServerProtocol,
    audio_queue: AudioQueue,
    openai_client: OpenAI,
    conversation_id: str,
) -> None:
    async for raw in ws:
        if isinstance(raw, bytes):
            logging.debug(f"Received {len(raw)} bytes of audio")
            await audio_queue.put(raw)
            continue

        try:
            msg = json.loads(raw)
        except json.JSONDecodeError:
            await send_json(ws, {"type": "error", "message": "Invalid JSON message"})
            continue

        msg_type = msg.get("type")
        if msg_type == "image":
            image_data = msg.get("image")
            if not image_data:
                await send_json(ws, {"type": "error", "message": "Missing image data"})
                continue
            logging.debug(f"Received {len(image_data)} chars of base64 image data")
            # await add_image_item(openai_client, conversation_id, image_data)
            await send_json(ws, {"type": "image_ack"})
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


async def add_image_item(openai_client: OpenAI, conversation_id: str, image_data_url: str) -> None:
    await asyncio.to_thread(
        openai_client.conversations.items.create,
        conversation_id=conversation_id,
        items=[
            {
                "type": "message",
                "role": "user",
                "content": [{"type": "input_image", "image_url": image_data_url}],
            }
        ],
    )


async def add_text_item(openai_client: OpenAI, conversation_id: str, text: str) -> None:
    await asyncio.to_thread(
        openai_client.conversations.items.create,
        conversation_id=conversation_id,
        items=[
            {
                "type": "message",
                "role": "user",
                "content": [{"type": "input_text", "text": text}],
            }
        ],
    )


async def run_stt(
    gradium_client: gradium.client.GradiumClient,
    audio_queue: AudioQueue,
    final_texts: asyncio.Queue[Optional[str]],
    ws: websockets.WebSocketServerProtocol,
    config: GatewayConfig,
) -> None:
    logging.info("Starting STT stream (model=%s format=%s)", config.stt_model, config.stt_input_format)
    stream = await gradium_client.stt_stream(
        {
            "model_name": config.stt_model,
            "input_format": config.stt_input_format,
            "json_config": {"language": config.stt_language},
        },
        audio_queue.generator(),
    )

    last_text: Optional[str] = None
    last_update_id = 0

    async def flush_after_silence(update_id: int) -> None:
        await asyncio.sleep(0.5)
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


def transform_text_sync(
    openai_client: OpenAI,
    config: GatewayConfig,
    conversation_id: str,
    text: str,
) -> str:
    response = openai_client.responses.create(
        model=config.openai_model,
        instructions=config.openai_prompt,
        conversation=conversation_id,
        input=[
            {
                "role": "user",
                "content": [{"type": "input_text", "text": text}],
            }
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
    conversation_id: str,
) -> None:
    while True:
        text = await final_texts.get()
        if text is None:
            break

        logging.info("Transforming text (len=%d)", len(text))
        transformed = await asyncio.to_thread(
            transform_text_sync,
            openai_client,
            config,
            conversation_id,
            text,
        )
        logging.info("Transformed text (len=%d)", len(transformed))
        logging.info("Transformed text: %s", transformed)
        await send_json(ws, {"type": "text_final", "text": transformed})
        await run_tts(gradium_client, ws, config, transformed)


class GatewayHandler:
    def __init__(self, config: GatewayConfig) -> None:
        self._config = config
        self._gradium_client = gradium.client.GradiumClient(api_key=config.gradium_api_key)
        self._openai_client = OpenAI(api_key=config.openai_api_key)

    async def __call__(self, ws: websockets.WebSocketServerProtocol) -> None:
        conversation_id = await self._create_conversation()
        if not await self._prepare_session(ws):
            return

        audio_queue = AudioQueue()
        final_texts: asyncio.Queue[Optional[str]] = asyncio.Queue()
        await self._run_pipeline(ws, conversation_id, audio_queue, final_texts)

    async def _create_conversation(self) -> str:
        conversation = await asyncio.to_thread(self._openai_client.conversations.create)
        conversation_id = conversation.id
        logging.info("OpenAI conversation created: %s", conversation_id)
        return conversation_id

    async def _prepare_session(self, ws: websockets.WebSocketServerProtocol) -> bool:
        try:
            start_msg = await recv_start_message(ws)
        except Exception as exc:
            await send_json(ws, {"type": "error", "message": str(exc)})
            return False

        logging.info("Client start message: %s", start_msg)
        try:
            validate_start_message(start_msg, self._config)
        except Exception as exc:
            await send_json(ws, {"type": "error", "message": str(exc)})
            return False

        await send_json(
            ws,
            {
                "type": "ready",
                "stt_input_format": self._config.stt_input_format,
                "tts_output_format": self._config.tts_output_format,
            },
        )
        return True

    async def _run_pipeline(
        self,
        ws: websockets.WebSocketServerProtocol,
        conversation_id: str,
        audio_queue: AudioQueue,
        final_texts: asyncio.Queue[Optional[str]],
    ) -> None:
        receive_task = asyncio.create_task(
            receive_client_messages(ws, audio_queue, self._openai_client, conversation_id)
        )

        stt_task = asyncio.create_task(
            run_stt(self._gradium_client, audio_queue, final_texts, ws, self._config)
        )
        tts_task = asyncio.create_task(
            run_transform_and_tts(
                self._openai_client,
                self._gradium_client,
                final_texts,
                ws,
                self._config,
                conversation_id,
            )
        )

        done, pending = await asyncio.wait(
            {receive_task}, #stt_task, tts_task},
            return_when=asyncio.FIRST_EXCEPTION,
        )

        for task in pending:
            task.cancel()

        for task in done:
            if task.exception():
                logging.exception("Gateway task failed", exc_info=task.exception())


async def main() -> None:
    config = load_config()
    # logging.basicConfig(level=config.log_level)
    handler = GatewayHandler(config)
    async with websockets.serve(handler, config.host, config.port, max_size=None):
        logging.info("Gateway listening on ws://%s:%s", config.host, config.port)
        await asyncio.Event().wait()


if __name__ == "__main__":
    asyncio.run(main())
