"""Tests for the /api/companion/greet and /api/companion/chat endpoints."""

import base64
from unittest.mock import AsyncMock, MagicMock, patch

import pytest
from fastapi.testclient import TestClient

from main import app


@pytest.fixture
def client():
    return TestClient(app)


def _mock_openai_response(text: str):
    choice = MagicMock()
    choice.message.content = text
    resp = MagicMock()
    resp.choices = [choice]
    return resp


def _mock_gradium_tts_response():
    """Fake WAV audio bytes."""
    fake_audio = b"RIFF" + b"\x00" * 100
    mock_resp = MagicMock()
    mock_resp.content = fake_audio
    mock_resp.raise_for_status = MagicMock()
    return mock_resp


def _mock_gradium_stt_response(text: str):
    mock_resp = MagicMock()
    mock_resp.json.return_value = {"text": text}
    mock_resp.raise_for_status = MagicMock()
    return mock_resp


class TestCompanionGreet:

    @patch("main.httpx.AsyncClient")
    @patch("main.openai_client")
    def test_greet_success(self, mock_openai, mock_httpx_cls, client):
        mock_openai.chat.completions.create.return_value = _mock_openai_response(
            "What brings you back here? This place looks so peaceful."
        )

        tts_resp = _mock_gradium_tts_response()
        mock_client = AsyncMock()
        mock_client.post.return_value = tts_resp
        mock_client.__aenter__ = AsyncMock(return_value=mock_client)
        mock_client.__aexit__ = AsyncMock(return_value=False)
        mock_httpx_cls.return_value = mock_client

        resp = client.post(
            "/api/companion/greet",
            json={"scene_context": "A beach at sunset"},
        )

        assert resp.status_code == 200
        data = resp.json()
        assert "audio_base64" in data
        assert data["text"] == "What brings you back here? This place looks so peaceful."

    @patch("main.openai_client")
    def test_greet_openai_failure(self, mock_openai, client):
        mock_openai.chat.completions.create.side_effect = Exception("API down")

        resp = client.post(
            "/api/companion/greet",
            json={"scene_context": "A park"},
        )
        assert resp.status_code == 500
        assert "error" in resp.json()

    def test_greet_missing_context(self, client):
        resp = client.post("/api/companion/greet", json={})
        assert resp.status_code == 422


class TestCompanionChat:

    @patch("main.httpx.AsyncClient")
    @patch("main.openai_client")
    def test_chat_success(self, mock_openai, mock_httpx_cls, client):
        mock_openai.chat.completions.create.return_value = _mock_openai_response(
            "That sounds like a wonderful memory. What were you feeling in that moment?"
        )

        # First call = STT, second call = TTS
        stt_resp = _mock_gradium_stt_response("I used to come here with my family")
        tts_resp = _mock_gradium_tts_response()

        mock_client = AsyncMock()
        mock_client.post.side_effect = [stt_resp, tts_resp]
        mock_client.__aenter__ = AsyncMock(return_value=mock_client)
        mock_client.__aexit__ = AsyncMock(return_value=False)
        mock_httpx_cls.return_value = mock_client

        fake_audio_b64 = base64.b64encode(b"\x00" * 50).decode()

        resp = client.post(
            "/api/companion/chat",
            json={
                "audio_base64": fake_audio_b64,
                "conversation_history": [
                    {"role": "assistant", "content": "What brings you back here?"},
                ],
                "scene_context": "A beach at sunset",
            },
        )

        assert resp.status_code == 200
        data = resp.json()
        assert data["user_text"] == "I used to come here with my family"
        assert "audio_base64" in data
        assert "text" in data

    @patch("main.httpx.AsyncClient")
    @patch("main.openai_client")
    def test_chat_empty_stt(self, mock_openai, mock_httpx_cls, client):
        """When STT returns empty, should use '(silence)' as user text."""
        mock_openai.chat.completions.create.return_value = _mock_openai_response(
            "I'm here whenever you're ready to talk."
        )

        stt_resp = _mock_gradium_stt_response("")
        tts_resp = _mock_gradium_tts_response()

        mock_client = AsyncMock()
        mock_client.post.side_effect = [stt_resp, tts_resp]
        mock_client.__aenter__ = AsyncMock(return_value=mock_client)
        mock_client.__aexit__ = AsyncMock(return_value=False)
        mock_httpx_cls.return_value = mock_client

        fake_audio_b64 = base64.b64encode(b"\x00" * 50).decode()

        resp = client.post(
            "/api/companion/chat",
            json={
                "audio_base64": fake_audio_b64,
                "conversation_history": [],
                "scene_context": "A park",
            },
        )

        assert resp.status_code == 200
        assert resp.json()["user_text"] == "(silence)"

    @patch("main.httpx.AsyncClient")
    @patch("main.openai_client")
    def test_chat_history_capped_at_10(self, mock_openai, mock_httpx_cls, client):
        """Conversation history should be capped at last 10 messages."""
        mock_openai.chat.completions.create.return_value = _mock_openai_response("OK")

        stt_resp = _mock_gradium_stt_response("hello")
        tts_resp = _mock_gradium_tts_response()

        mock_client = AsyncMock()
        mock_client.post.side_effect = [stt_resp, tts_resp]
        mock_client.__aenter__ = AsyncMock(return_value=mock_client)
        mock_client.__aexit__ = AsyncMock(return_value=False)
        mock_httpx_cls.return_value = mock_client

        # Send 20 messages in history
        long_history = [
            {"role": "user" if i % 2 == 0 else "assistant", "content": f"msg {i}"}
            for i in range(20)
        ]

        fake_audio_b64 = base64.b64encode(b"\x00" * 50).decode()

        resp = client.post(
            "/api/companion/chat",
            json={
                "audio_base64": fake_audio_b64,
                "conversation_history": long_history,
                "scene_context": "A room",
            },
        )

        assert resp.status_code == 200

        # Verify OpenAI was called with capped history (system + 10 history + 1 user = 12)
        call_args = mock_openai.chat.completions.create.call_args
        messages = call_args.kwargs.get("messages") or call_args[1].get("messages")
        # 1 system + 10 history + 1 current user = 12
        assert len(messages) == 12

    def test_chat_missing_audio(self, client):
        resp = client.post(
            "/api/companion/chat",
            json={"scene_context": "A park"},
        )
        assert resp.status_code == 422


class TestCompanionModels:

    def test_greet_request(self):
        from main import CompanionGreetRequest
        r = CompanionGreetRequest(scene_context="beach")
        assert r.scene_context == "beach"

    def test_chat_request_defaults(self):
        from main import CompanionChatRequest
        r = CompanionChatRequest(audio_base64="abc", scene_context="park")
        assert r.conversation_history == []
