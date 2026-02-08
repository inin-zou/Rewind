import os
from dataclasses import dataclass

from dotenv import load_dotenv


@dataclass(frozen=True)
class GatewayConfig:
    gradium_api_key: str
    openai_api_key: str
    host: str
    port: int
    stt_model: str
    stt_input_format: str
    stt_language: str
    tts_model: str
    tts_voice_id: str
    tts_output_format: str
    openai_model: str
    openai_prompt: str
    log_level: str


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
        stt_language=os.getenv("GRADIUM_STT_LANGUAGE", "en"),
        tts_model=os.getenv("GRADIUM_TTS_MODEL", "default"),
        tts_voice_id=os.getenv("GRADIUM_TTS_VOICE_ID", "YTpq7expH9539ERJ"),
        tts_output_format=os.getenv("GRADIUM_TTS_OUTPUT_FORMAT", "pcm"),
        openai_model=os.getenv("OPENAI_MODEL", "gpt-4o-mini"),
        openai_prompt=os.getenv(
            "OPENAI_PROMPT",
            "You're an AI therapist.",
        ),
        log_level=os.getenv("LOG_LEVEL", "INFO").upper(),
    )
