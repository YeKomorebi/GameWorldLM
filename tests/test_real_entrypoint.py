import json

from examples import real_generation
from llm.providers import LLMError


def test_auth_failure_stops_batch_and_preserves_failure_record(tmp_path, monkeypatch):
    class Backend:
        calls = 0
        closed = False
        last_metadata = {}

        def complete(self, messages):
            self.calls += 1
            self.last_metadata = {"http_status": 401}
            raise LLMError("Authentication failed", code="AuthenticationError")

        def close(self):
            self.closed = True

    backend = Backend()
    monkeypatch.setattr(real_generation, "load_dotenv", lambda *args, **kwargs: False)
    monkeypatch.setenv("QWEN_API_KEY", "test-only")
    monkeypatch.setattr(real_generation, "OpenAICompatibleBackend", lambda config: backend)
    assert real_generation.main(["--output-dir", str(tmp_path), "--no-export"]) == 1
    assert backend.calls == 1
    assert backend.closed
    report = json.loads(next((tmp_path / "batches").glob("*.json")).read_text())
    assert report["not_run"] == 9
    assert report["runs"][0]["status"] == "provider_error"
    assert "401" in report["stopped_reason"]


def test_single_prompt_entrypoint_produces_run_and_dataset(forest, tmp_path, monkeypatch):
    class Backend:
        last_metadata = {"actual_model": "test"}

        def complete(self, messages):
            return forest.model_dump_json()

        def close(self):
            pass

    monkeypatch.setattr(real_generation, "load_dotenv", lambda *args, **kwargs: False)
    monkeypatch.setenv("QWEN_API_KEY", "test-only")
    monkeypatch.setattr(real_generation, "OpenAICompatibleBackend", lambda config: Backend())
    assert real_generation.main(["--prompt", "A village", "--output-dir", str(tmp_path)]) == 0
    report = json.loads(next((tmp_path / "batches").glob("*.json")).read_text())
    assert report["succeeded"] == report["planned"] == 1
    dataset = tmp_path / report["dataset"]["path"]
    assert (dataset / "train.jsonl").exists()
    assert report["dataset"]["accepted"] == 1
