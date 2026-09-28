"""Central configuration.

All secrets come from environment variables only. Nothing secret is ever
hard-coded, written to disk, or committed to the repo.
"""
import os


def _get(name: str, default: str = "") -> str:
    return os.environ.get(name, default)


class Settings:
    def __init__(self) -> None:
        self.groq_api_key = _get("GROQ_API_KEY")
        self.finnhub_api_key = _get("FINNHUB_API_KEY")
        self.alpha_vantage_api_key = _get("ALPHA_VANTAGE_API_KEY")
        self.xai_api_key = _get("XAI_API_KEY")
        self.llm_provider = (_get("LLM_PROVIDER", "groq") or "groq").lower().strip()
        self.data_dir = _get("DATA_DIR", "/tmp/portfolio-pulse-data")
        self.moderator_model = _get("MODERATOR_MODEL", "")
        self.agent_model = _get("AGENT_MODEL", "")
        self.priority_days = int(_get("PRIORITY_DAYS", "5") or 5)
        self.materiality_threshold = float(_get("MATERIALITY_THRESHOLD", "0.45") or 0.45)

    @property
    def live_news(self) -> bool:
        """True when at least one news API key is configured."""
        return bool(self.finnhub_api_key or self.alpha_vantage_api_key)

    @property
    def live_llm(self) -> bool:
        """True when the selected LLM provider has an API key."""
        if self.llm_provider == "xai":
            return bool(self.xai_api_key)
        return bool(self.groq_api_key)

    def llm_endpoint(self) -> tuple[str, str, str, str]:
        """Return (base_url, api_key, moderator_model, agent_model)."""
        if self.llm_provider == "xai":
            return (
                "https://api.x.ai/v1",
                self.xai_api_key,
                self.moderator_model or "grok-4.3",
                self.agent_model or "grok-4.3",
            )
        return (
            "https://api.groq.com/openai/v1",
            self.groq_api_key,
            self.moderator_model or "openai/gpt-oss-120b",
            self.agent_model or "openai/gpt-oss-120b",
        )


settings = Settings()
