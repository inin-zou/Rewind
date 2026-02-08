import asyncio
import base64
import json
from typing import Optional

import sounddevice as sd
import websockets

SAMPLE_RATE = 24000
CHANNELS = 1
BLOCK_SIZE = 1920  # 80ms at 24kHz
DTYPE = "int16"

OUTPUT_TTS = "tts_output_live.pcm"
TTS_SAMPLE_RATE = 48000
IMAGE_PATH = "image.png"


def load_image_data_url(path: str) -> str:
    with open(path, "rb") as f:
        encoded = base64.b64encode(f.read()).decode("ascii")
    return f"data:image/png;base64,{encoded}"


def _mic_callback(indata, _frames, _time, status):
    if status:
        print("Audio status:", status)
    # indata is a bytes-like object for RawInputStream
    _mic_callback.queue.put_nowait(bytes(indata))


async def send_audio(ws: websockets.WebSocketClientProtocol, queue: asyncio.Queue[bytes]) -> None:
    sent = 0
    while True:
        chunk = await queue.get()
        if chunk is None:
            break
        if not isinstance(chunk, (bytes, bytearray, memoryview)):
            raise RuntimeError("Audio must be sent as binary frames, not base64.")
        await ws.send(chunk)
        sent += len(chunk)
        if sent % (BLOCK_SIZE * 2 * 50) == 0:
            print(f"Sent {sent} bytes of audio")


async def recv_messages(ws: websockets.WebSocketClientProtocol) -> None:
    with open(OUTPUT_TTS, "wb") as out:
        with sd.RawOutputStream(
            samplerate=TTS_SAMPLE_RATE,
            channels=1,
            dtype="int16",
            blocksize=0,
        ) as stream:
            async for msg in ws:
                if isinstance(msg, bytes):
                    out.write(msg)
                    stream.write(msg)
                else:
                    print("JSON:", msg)


async def main() -> None:
    async with websockets.connect("ws://localhost:8765") as ws:
        await ws.send(
            json.dumps(
                {
                    "type": "start",
                    "sample_rate": SAMPLE_RATE,
                    "channels": CHANNELS,
                    "input_format": "pcm",
                }
            )
        )
        await ws.send(
            json.dumps(
                {
                    "type": "image",
                    "image": load_image_data_url(IMAGE_PATH),
                }
            )
        )

        audio_queue: asyncio.Queue[Optional[bytes]] = asyncio.Queue()
        _mic_callback.queue = audio_queue

        recv_task = asyncio.create_task(recv_messages(ws))
        send_task = asyncio.create_task(send_audio(ws, audio_queue))

        try:
            with sd.RawInputStream(
                samplerate=SAMPLE_RATE,
                channels=CHANNELS,
                dtype=DTYPE,
                blocksize=BLOCK_SIZE,
                callback=_mic_callback,
            ):
                print("Streaming mic audio... Press Ctrl+C to stop.")
                while True:
                    await asyncio.sleep(0.1)
        except KeyboardInterrupt:
            print("Stopping...")
        finally:
            await ws.send(json.dumps({"type": "end"}))
            await audio_queue.put(None)
            await send_task
            recv_task.cancel()


if __name__ == "__main__":
    asyncio.run(main())
