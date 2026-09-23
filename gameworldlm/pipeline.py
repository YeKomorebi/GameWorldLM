"""Audited prompt -> validated JSON -> PNG pipeline shared by batch callers."""

from copy import deepcopy
from dataclasses import dataclass
from pathlib import Path

from dataset.storage import RunWriter
from engine.pygame_renderer import PygameRenderer
from gameworldlm import __version__
from llm.providers import LLMBackend, LLMError
from world.evaluation import GenerationExpectations, ValidationReport
from world.generator import GenerationAttempt, WorldGenerationError, WorldGenerator
from world.io import save_world


@dataclass(frozen=True)
class PipelineResult:
    run_id: str
    directory: Path
    status: str
    validation: ValidationReport
    error: str | None = None

    @property
    def success(self) -> bool:
        return self.status == "success"


class GenerationPipeline:
    def __init__(
        self,
        backend: LLMBackend,
        *,
        provider: str,
        model: str,
        output_dir: str | Path,
        source: str = "llm",
        max_attempts: int = 3,
        tile_size: int = 24,
        parameters: dict | None = None,
    ):
        if source not in {"llm", "fixture", "test"}:
            raise ValueError("source must be llm, fixture, or test")
        self.backend = backend
        self.generator = WorldGenerator(backend, max_attempts)
        self.renderer = PygameRenderer(tile_size)
        self.output_dir = Path(output_dir)
        self.metadata = {
            "provider": provider,
            "model": model,
            "source": source,
            "application_version": __version__,
            "max_attempts": max_attempts,
            "parameters": parameters or {},
        }

    def run(
        self,
        prompt: str,
        *,
        case_id: str | None = None,
        expectations: GenerationExpectations | None = None,
    ) -> PipelineResult:
        writer = RunWriter(
            self.output_dir,
            self.metadata
            | {
                "prompt": prompt,
                "case_id": case_id,
                "expectations": expectations.model_dump() if expectations is not None else None,
            },
        )
        last_report = ValidationReport("not_evaluated")

        def capture(attempt: GenerationAttempt) -> None:
            nonlocal last_report
            last_report = attempt.validation
            metadata = deepcopy(getattr(self.backend, "last_metadata", {}))
            writer.save_attempt(attempt, metadata)

        stage = "generation"
        try:
            result = self.generator.generate(prompt, expectations=expectations, on_attempt=capture)
            stage = "render"
            save_world(result.world, writer.directory / "world.json")
            self.renderer.render(result.world, writer.directory / "map.png")
        except (KeyboardInterrupt, SystemExit):
            writer.finish("interrupted", last_report.to_dict(), error="Run interrupted")
            raise
        except Exception as exc:
            if isinstance(exc, LLMError):
                status = "provider_error"
            elif isinstance(exc, WorldGenerationError):
                status = "validation_failed"
            else:
                status = f"{stage}_error"
            error = (
                str(exc)
                if isinstance(exc, (LLMError, WorldGenerationError, ValueError, OSError))
                else type(exc).__name__
            )
            writer.finish(status, last_report.to_dict(), error=error)
            return PipelineResult(writer.run_id, writer.directory, status, last_report, error)
        writer.finish("success", last_report.to_dict())
        return PipelineResult(writer.run_id, writer.directory, "success", last_report)
