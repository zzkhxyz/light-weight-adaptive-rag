from ac_rag.config import Settings


def test_openai_compatible_settings_load_from_environment(monkeypatch):
    monkeypatch.setenv("OPENAI_API_KEY", "test-key")
    monkeypatch.setenv("OPENAI_BASE_URL", "https://example.test/api/v1")
    settings = Settings(_env_file=None)
    assert settings.openai_api_key.get_secret_value() == "test-key"
    assert settings.openai_base_url == "https://example.test/api/v1"
