"""Tests for the /api/generate-sound endpoint."""

import base64
import io
from unittest.mock import AsyncMock, MagicMock, patch

import pytest
from fastapi.testclient import TestClient
from PIL import Image

from main import app


@pytest.fixture
def client():
    return TestClient(app)


@pytest.fixture
def sample_image_base64():
    """Create a small test image and return its base64 encoding."""
    img = Image.new("RGB", (100, 100), color="blue")
    buf = io.BytesIO()
    img.save(buf, format="JPEG")
    return base64.b64encode(buf.getvalue()).decode()


def _mock_openai_response(text: str):
    """Build a mock OpenAI chat completion response."""
    choice = MagicMock()
    choice.message.content = text
    resp = MagicMock()
    resp.choices = [choice]
    return resp


class TestGenerateSoundEndpoint:
    """Tests for POST /api/generate-sound."""

    @patch("main.httpx.AsyncClient")
    @patch("main.openai_client")
    def test_success(self, mock_openai, mock_httpx_cls, client, sample_image_base64):
        """Full happy-path: OpenAI analyzes image → ElevenLabs returns audio."""
        # Mock OpenAI vision response
        mock_openai.chat.completions.create.return_value = _mock_openai_response(
            "Gentle ocean waves crashing on a sandy beach with distant seagulls"
        )

        # Mock ElevenLabs HTTP response
        fake_audio = b"\xff\xfb\x90\x00" * 100  # fake MP3 bytes
        mock_resp = MagicMock()
        mock_resp.status_code = 200
        mock_resp.content = fake_audio
        mock_resp.raise_for_status = MagicMock()

        mock_client_instance = AsyncMock()
        mock_client_instance.post.return_value = mock_resp
        mock_client_instance.__aenter__ = AsyncMock(return_value=mock_client_instance)
        mock_client_instance.__aexit__ = AsyncMock(return_value=False)
        mock_httpx_cls.return_value = mock_client_instance

        resp = client.post(
            "/api/generate-sound",
            json={"image_base64": sample_image_base64},
        )

        assert resp.status_code == 200
        data = resp.json()
        assert "audio_base64" in data
        assert "sound_prompt" in data
        assert data["sound_prompt"] == "Gentle ocean waves crashing on a sandy beach with distant seagulls"

        # Verify the audio_base64 decodes back to our fake audio
        decoded = base64.b64decode(data["audio_base64"])
        assert decoded == fake_audio

    @patch("main.httpx.AsyncClient")
    @patch("main.openai_client")
    def test_openai_called_with_image(self, mock_openai, mock_httpx_cls, client, sample_image_base64):
        """Verify OpenAI is called with the correct image data."""
        mock_openai.chat.completions.create.return_value = _mock_openai_response("Wind blowing")

        fake_audio = b"\x00" * 50
        mock_resp = MagicMock()
        mock_resp.content = fake_audio
        mock_resp.raise_for_status = MagicMock()

        mock_client_instance = AsyncMock()
        mock_client_instance.post.return_value = mock_resp
        mock_client_instance.__aenter__ = AsyncMock(return_value=mock_client_instance)
        mock_client_instance.__aexit__ = AsyncMock(return_value=False)
        mock_httpx_cls.return_value = mock_client_instance

        client.post("/api/generate-sound", json={"image_base64": sample_image_base64})

        # Check OpenAI was called with the image
        call_args = mock_openai.chat.completions.create.call_args
        messages = call_args.kwargs.get("messages") or call_args[1].get("messages")
        user_msg = messages[1]
        image_content = user_msg["content"][0]
        assert image_content["type"] == "image_url"
        assert sample_image_base64 in image_content["image_url"]["url"]

    @patch("main.httpx.AsyncClient")
    @patch("main.openai_client")
    def test_elevenlabs_called_with_loop(self, mock_openai, mock_httpx_cls, client, sample_image_base64):
        """Verify ElevenLabs is called with loop=True."""
        mock_openai.chat.completions.create.return_value = _mock_openai_response("Rain on a tin roof")

        fake_audio = b"\x00" * 50
        mock_resp = MagicMock()
        mock_resp.content = fake_audio
        mock_resp.raise_for_status = MagicMock()

        mock_client_instance = AsyncMock()
        mock_client_instance.post.return_value = mock_resp
        mock_client_instance.__aenter__ = AsyncMock(return_value=mock_client_instance)
        mock_client_instance.__aexit__ = AsyncMock(return_value=False)
        mock_httpx_cls.return_value = mock_client_instance

        client.post("/api/generate-sound", json={"image_base64": sample_image_base64})

        # Check ElevenLabs call payload
        post_call = mock_client_instance.post.call_args
        json_body = post_call.kwargs.get("json") or post_call[1].get("json")
        assert json_body["text"] == "Rain on a tin roof"
        assert json_body["loop"] is True
        assert json_body["duration_seconds"] == 10
        assert json_body["model_id"] == "eleven_text_to_sound_v2"

    @patch("main.openai_client")
    def test_openai_failure_returns_500(self, mock_openai, client, sample_image_base64):
        """If OpenAI fails, endpoint returns 500."""
        mock_openai.chat.completions.create.side_effect = Exception("OpenAI rate limit")

        resp = client.post(
            "/api/generate-sound",
            json={"image_base64": sample_image_base64},
        )

        assert resp.status_code == 500
        assert "error" in resp.json()

    def test_missing_image_returns_422(self, client):
        """Missing image_base64 should return 422 validation error."""
        resp = client.post("/api/generate-sound", json={})
        assert resp.status_code == 422


class TestSoundModels:
    """Test the request/response models."""

    def test_sound_request_requires_image(self):
        from main import SoundRequest
        with pytest.raises(Exception):
            SoundRequest()

    def test_sound_response_fields(self):
        from main import SoundResponse
        r = SoundResponse(audio_base64="AAAA", sound_prompt="birds chirping")
        assert r.audio_base64 == "AAAA"
        assert r.sound_prompt == "birds chirping"
