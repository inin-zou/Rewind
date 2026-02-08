# Audio Gateway (Gradium STT → OpenAI → Gradium TTS)

This project runs a WebSocket gateway that accepts streaming audio, transcribes it with Gradium, rewrites the text with OpenAI, and streams synthesized speech back via Gradium TTS.

## Setup

Install deps (using your preferred tool, e.g. `uv` or `pip`), then set environment variables (supports `.env`):

- `GRADIUM_API_KEY` (required)
- `OPENAI_API_KEY` (required)
- Optional configuration:
  - `GATEWAY_HOST` (default `0.0.0.0`)
  - `GATEWAY_PORT` (default `8765`)
  - `GRADIUM_STT_MODEL` (default `default`)
  - `GRADIUM_STT_INPUT_FORMAT` (default `pcm`)
  - `GRADIUM_TTS_MODEL` (default `default`)
  - `GRADIUM_TTS_VOICE_ID` (default `YTpq7expH9539ERJ`)
  - `GRADIUM_TTS_OUTPUT_FORMAT` (default `pcm`)
  - `OPENAI_MODEL` (default `gpt-4o-mini`)
  - `OPENAI_PROMPT` (default: rewrite/clean transcript)

## Run

```bash
python main.py
```

The server listens on `ws://<host>:<port>`.
You can set `LOG_LEVEL=DEBUG` for detailed streaming logs.

## WebSocket Protocol

### Client → Server

- JSON start message **required**:

```json
{
  "type": "start",
  "input_format": "pcm",
  "output_format": "pcm",
  "voice_id": "YTpq7expH9539ERJ",
  "stt_model": "default",
  "tts_model": "default",
  "openai_model": "gpt-4o-mini",
  "prompt": "Rewrite the user transcript into concise, well-punctuated text while preserving meaning."
}
```

- Binary audio frames (preferred), or JSON base64 audio:

```json
{ "type": "audio", "audio": "<base64>" }
```

- End stream:

```json
{ "type": "end" }
```

### Server → Client

- Ready:

```json
{ "type": "ready", "stt_input_format": "pcm", "tts_output_format": "pcm" }
```

- Partial STT:

```json
{ "type": "stt_partial", "text": "..." }
```

- Final STT:

```json
{ "type": "stt_final", "text": "..." }
```

- Final transformed text:

```json
{ "type": "text_final", "text": "..." }
```

- TTS audio frames: binary (raw bytes)
- TTS start/end markers:

```json
{ "type": "tts_start", "format": "pcm", "sample_rate": 48000 }
```

```json
{ "type": "tts_end" }
```

- Errors:

```json
{ "type": "error", "message": "..." }
```

## Notes

- `input_format` and `output_format` should match what your Gradium account supports.
- This gateway forwards binary audio frames directly; ensure your client encodes audio in the expected format.

## Quick Tests

### File-based client

Use `ws_test.py` with a raw PCM input file (16-bit, mono). Generate one from WAV:

```bash
ffmpeg -i input.wav -ac 1 -ar 24000 -f s16le input_24k_mono_s16le.pcm
```

Then run:

```bash
python ws_test.py
```

### Live mic client (real-time playback)

`ws_live_mic.py` captures your mic at 24 kHz PCM, streams to STT, and plays TTS back
in real time (48 kHz PCM) while also saving to `tts_output_live.pcm`.

```bash
python ws_live_mic.py
```

If you need to play the saved PCM file:

```bash
ffplay -f s16le -ar 48000 -ac 1 tts_output_live.pcm
```
