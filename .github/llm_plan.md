# OpenAI LLM Rewrite Integration Plan

This document tells Codex exactly how to replace the placeholder rewriting hook with a production-ready integration that calls OpenAI's latest Responses API (e.g., `gpt-4.1` / `gpt-4.1-mini`) to rewrite 500-word windows that violate Lexile thresholds. Follow each numbered section sequentially; every bullet is an action item with explicit file targets or commands.

---

## 0. Goals & Success Criteria

1. **Wire a real OpenAI-powered rewriter** into the existing `process_document` loop so violating windows are rewritten automatically when `rewrite_enabled=True`.
2. **Keep the system deterministic and testable** by isolating OpenAI-specific code and providing mocks so tests pass without real API calls.
3. **Expose configuration & CLI knobs** so users can pick the model, temperature, and API credentials via config or environment variables.
4. **Document how to set up and run the new workflow** and provide a minimal end-to-end verification recipe.

---

## 1. Dependencies & Packaging

1. Edit `pyproject.toml` (`[tool.poetry.dependencies]`) to add the official OpenAI SDK:
   - `openai = { version = ">=1.40.0", optional = true }`
   - Keep it optional so users who only analyze text do not need the SDK.
2. Extend `[tool.poetry.extras]` with a new extra, e.g.:
   ```toml
   [tool.poetry.extras]
   lexile-v2 = ["tensorflow", "joblib"]
   llm-openai = ["openai"]
   ```
3. Regenerate `poetry.lock` with `poetry lock --no-update` so the new dependency is captured.
4. Confirm `README.md` mentions `pip install .[llm-openai]` (see Section 8).

---

## 2. Configuration Model Changes

1. In `src/lexile_corpus_tuner/config.py` introduce a nested settings object for LLMs:
   ```python
   @dataclass(slots=True)
   class OpenAISettings:
       enabled: bool = False
       model: str = "gpt-4.1-mini"
       api_key: str | None = None
       api_key_env: str = "OPENAI_API_KEY"
       base_url: str | None = None
       organization: str | None = None
       temperature: float = 0.3
       max_output_tokens: int = 400
       top_p: float = 0.95
       request_timeout: float = 60.0
       parallel_requests: int = 1
   ```
2. Add `openai: OpenAISettings = field(default_factory=OpenAISettings)` to `LexileTunerConfig` and keep `rewrite_enabled` / `rewrite_model` for backwards compatibility.
3. Update `_build_kwargs`, `config_from_dict`, and YAML loaders so nested `openai` keys are parsed (e.g., detect dict under `"openai"` and build `OpenAISettings`).
4. When serializing via `to_dict`, ensure nested dataclasses are converted to plain dicts (e.g., call `asdict` on `OpenAISettings`).
5. Make sure default config YAML printed by `lexile-tuner print-config` includes the `openai` block with inline comments (Section 8).

---

## 3. CLI & Environment Wiring

1. In `src/lexile_corpus_tuner/cli.py` add options to both `analyze` and `rewrite` commands:
   - `--rewrite-enabled / --no-rewrite-enabled`
   - `--rewrite-model`
   - OpenAI-specific overrides: `--openai-model`, `--openai-api-key`, `--openai-api-key-env`, `--openai-base-url`, `--openai-organization`, `--openai-temperature`, `--openai-max-output-tokens`, `--openai-top-p`, `--openai-request-timeout`.
2. Extend `_apply_estimator_overrides` or introduce `_apply_rewriter_overrides` to copy CLI values into `LexileTunerConfig.openai`.
3. Ensure `_build_rewriter` checks `config.openai.enabled` **and** `config.rewrite_enabled`. If both true, construct the OpenAI-backed rewriter; otherwise fall back to `NoOpRewriter`.
4. Pull API keys in this order:
   1. Explicit CLI `--openai-api-key`.
   2. Config `openai.api_key`.
   3. `os.environ[openai.api_key_env]`.
   - Raise a descriptive error if no key is available when rewriting is requested.

---

## 4. Prompt & Rewriting Strategy

1. Keep `PROMPT_TEMPLATE` in `src/lexile_corpus_tuner/rewriting.py` but refactor it into two pieces:
   - A `SYSTEM_PROMPT` describing the Lexile target, guardrails, and formatting rules.
   - A `USER_PROMPT_TEMPLATE` that receives `text`, `target`, and optionally metadata (document id, window id, stats).
2. Include explicit instructions:
   - Maintain factual content and names.
   - Keep roughly the same length (±10% tokens).
   - Output plain text only (no Markdown, headings, or quotes).
   - Avoid introducing new facts.
3. Provide optional context in the prompt (e.g., previous Lexile scores, constraint thresholds) to help the LLM.
4. Document these rules in code comments and README so future edits are consistent.

---

## 5. OpenAI Client & Rewriter Implementation

1. Create a new module `src/lexile_corpus_tuner/llm/openai_client.py`:
   - Define `class OpenAIRewriteClient:` that wraps `openai.OpenAI`.
   - Accept settings + logger; lazily instantiate `OpenAI` with `api_key`, `base_url`, `organization`.
   - Provide a method `rewrite(text: str, target: float, metadata: RewriteMetadata) -> str`.
   - Build the request using the Responses API:
     ```python
     response = self._client.responses.create(
         model=settings.model,
         input=[
             {"role": "system", "content": SYSTEM_PROMPT.format(target=int(target))},
             {"role": "user", "content": USER_PROMPT_TEMPLATE.format(...)}
         ],
         temperature=settings.temperature,
         max_output_tokens=settings.max_output_tokens,
         top_p=settings.top_p,
         timeout=settings.request_timeout,
     )
     ```
   - Extract the first text segment via `response.output[0].content[0].text`.
2. Replace the naive `LLMRewriter` in `src/lexile_corpus_tuner/rewriting.py` with an implementation that consumes any callable following the `RewriteClient` protocol. Then add `class OpenAIRewriter(Rewriter)` that composes the client + templates.
3. Include retries with exponential backoff (e.g., use `tenacity` if already in deps, otherwise implement simple loop) to handle 429/500 errors. Limit to 3 attempts and surface a clear error after failures.
4. Ensure rewrites preserve paragraph boundaries by splitting/joining on double newlines before sending; instruct the model to keep newline structure.
5. Return the new text trimmed of leading/trailing whitespace but keep inner whitespace as provided.

---

## 6. Pipeline Integration

1. In `cli._build_rewriter` detect `config.openai.enabled`. When true:
   - Resolve API key.
   - Instantiate `OpenAIRewriteClient` with `config.openai`.
   - Return `OpenAIRewriter(client=client, system_prompt=..., user_prompt_template=...)`.
2. Pass document/window metadata when calling `rewriter.rewrite`:
   - Extend `Rewriter.rewrite` signature to accept `doc_id`, `window_id`, and `ConstraintViolation` context or create a lightweight `RewriteRequest` dataclass in `models.py`.
   - Update `process_document` to supply this metadata.
3. Log each rewrite attempt (doc id, window id, source Lexile, target) for diagnostics; use the existing logging approach or add a new logger in `rewriting.py`.
4. Ensure rewritten text is reinserted via `replace_window_span` exactly once per pass. After a successful rewrite, continue loop (existing behavior already reruns scoring).

---

## 7. Testing Strategy

1. Add `tests/test_openai_rewriter.py` with the following cases (mock the SDK to avoid network access):
   - `test_openai_rewriter_builds_prompts`: monkeypatch `openai.OpenAI` so `responses.create` returns a stub object. Assert the prompt contains target Lexile, constraint text, and raw passage.
   - `test_openai_rewriter_handles_missing_key`: verify a `RuntimeError` is raised when no API key is available.
   - `test_retry_logic_on_transient_error`: have the mock client throw once, succeed next, and ensure the output is returned plus the client was called twice.
2. Update `tests/test_pipeline.py` (or add a new test) to ensure `process_document` uses the rewriter when enabled: inject a fake rewriter that returns deterministic text and assert `rewrite_enabled` triggers it.
3. Cover CLI wiring with a smoke test in `tests/test_cli.py`:
   - Use `CliRunner` to invoke `rewrite` with `--rewrite-enabled --openai-model dummy` plus env var `OPENAI_API_KEY=dummy`.
   - Patch `OpenAIRewriteClient` to bypass external calls.
4. Run `pytest` and ensure coverage still works without network access.

---

## 8. Documentation & Examples

1. Update `README.md`:
   - Add a “Rewriting with OpenAI” section describing prerequisites, installation command, environment variables, and sample CLI invocation.
   - Provide a warning about API costs and data privacy.
2. Update `examples/example_config.yaml` to include an `openai` block with sensible defaults:
   ```yaml
   rewrite_enabled: true
   rewrite_model: gpt-4.1-mini
   openai:
     enabled: true
     api_key_env: OPENAI_API_KEY
     temperature: 0.35
     max_output_tokens: 450
   ```
3. Add a short example script `examples/run_openai_rewrite.py` demonstrating how to load config, build the rewriter, and rewrite a string (similar to the lexile_v2 example).
4. Note in `.github/copilot-instructions.md` (or a new section) that Codex should never commit actual API keys and should use environment variables/local `.env`.

---

## 9. Operational Guidance

1. Provide logging hooks (e.g., `logging.getLogger(__name__)`) inside the OpenAI client to emit concise info + warning level messages for retries.
2. Add rate-limit friendly behavior: respect `OpenAISettings.parallel_requests` by guarding the client with an `asyncio.Semaphore` or a `threading.BoundedSemaphore` (simple approach: wrap calls in a context manager that enforces sequential calls when `parallel_requests == 1`).
3. Document how to rotate API keys and how to override the base URL for Azure/OpenAI-compatible gateways.
4. Mention privacy considerations—by default, OpenAI may retain prompts; instruct users to configure policy-compliant deployments (e.g., Azure) if needed.

---

## 10. Verification Checklist

1. `poetry install -E llm-openai` followed by `pytest` succeeds with mocks.
2. `OPENAI_API_KEY=dummy poetry run lexile-tuner rewrite --rewrite-enabled --openai-model gpt-4.1-mini --input-path examples/example_corpus --output-path artifacts/tuned --config examples/example_config.yaml` runs end-to-end (CLI uses stubbed OpenAI client during local verification).
3. README instructions render correctly and mention the new `llm-openai` extra.
4. A final sanity test uses a single short document, prints before/after Lexile stats, and confirms rewritten text differs while staying within constraints.

Following this plan will yield a robust OpenAI-backed rewriting flow that slots cleanly into the current architecture while remaining optional, testable, and well-documented.
