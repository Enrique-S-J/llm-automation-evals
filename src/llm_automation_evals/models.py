"""Data models: the shape of a task, a case, a model output, and a report.

Everything read from YAML forbids unknown keys. A typo such as
`expect_abstian: true` must stop the run, not quietly turn a safe-failure case
into an ordinary one that nothing checks.
"""

from __future__ import annotations

from typing import Annotated, Any, Literal, Optional, Union

from pydantic import BaseModel, ConfigDict, Field, field_validator, model_validator

# Ids and names end up in markdown and in the default output path, so they are
# restricted to characters that are inert in both.
SAFE_NAME = r"^[A-Za-z0-9][A-Za-z0-9_.-]*$"


class ConfigError(Exception):
    """A problem in the task files or the environment that the user can fix."""


class _Strict(BaseModel):
    model_config = ConfigDict(extra="forbid")


# ---------------------------------------------------------------------------
# Suite definition (task.yaml, cases.yaml)
# ---------------------------------------------------------------------------


class ScorerSpec(BaseModel):
    """One entry in a `scorers:` list.

    Written either as a bare type name (`- json_schema`) or as a mapping with
    options (`- {type: exact_field, name: category_match, fields: [category]}`).
    Options are scorer-specific, so unknown keys are kept here and checked by
    the scorer that receives them.
    """

    model_config = ConfigDict(extra="allow")

    type: Literal["json_schema", "exact_field", "keyword", "safe_failure", "llm_judge"]
    name: str = Field(default="", pattern=r"^([A-Za-z0-9][A-Za-z0-9_.-]*)?$")
    # None means "the scorer's own default": required for the deterministic
    # scorers, advisory for the LLM judge.
    required: Optional[bool] = None

    @model_validator(mode="before")
    @classmethod
    def _accept_bare_name(cls, value: Any) -> Any:
        return {"type": value} if isinstance(value, str) else value

    @property
    def label(self) -> str:
        return self.name or self.type

    @property
    def options(self) -> dict[str, Any]:
        return dict(self.model_extra or {})


# A keyword has to contain something visible. "" is a substring of every
# text: required, it always passes; forbidden, it always fails.
Term = Annotated[str, Field(pattern=r"\S")]


class Keywords(_Strict):
    # Each entry is a term that must appear, or a list of alternatives of
    # which at least one must appear ("voicemail" or "voice mail").
    must_contain: list[Union[Term, Annotated[list[Term], Field(min_length=1)]]] = []
    must_not_contain: list[Term] = []


class Case(_Strict):
    id: str = Field(pattern=SAFE_NAME)
    # Values substituted into the task's prompt template.
    input: dict[str, str]
    # Only the fields listed here are compared. A case can leave a field out
    # when no single answer is defensible.
    expected: dict[str, Any] = {}
    # The right answer is to decline: garbage, an empty ticket, an injection.
    expect_abstain: bool = False
    # Reasonable people could label this differently. Reported, not excused.
    ambiguous: bool = False
    keywords: Optional[Keywords] = None
    # Replaces the task's scorer list for this case when present.
    scorers: Optional[list[ScorerSpec]] = None
    notes: str = ""


class EndpointEnv(_Strict):
    """Names of the environment variables the endpoint is read from."""

    # pydantic before 2.10 reserves every field name starting with "model_"
    # and warns about `model_env` on import, which would put a warning on
    # stderr at the top of every command. Nothing here shadows a pydantic
    # attribute, so the reservation is lifted for this class.
    model_config = ConfigDict(extra="forbid", protected_namespaces=())

    base_url_env: str = "EVAL_BASE_URL"
    model_env: str = "EVAL_MODEL"
    api_key_env: str = "EVAL_API_KEY"
    judge_model_env: str = "EVAL_JUDGE_MODEL"


class RequestSettings(_Strict):
    temperature: Union[int, float] = 0
    max_tokens: Optional[int] = Field(default=512, ge=1)
    timeout_seconds: float = Field(default=60.0, gt=0)
    concurrency: int = Field(default=4, ge=1)
    retry_delay_seconds: float = Field(default=1.0, ge=0)
    # Merged into the request body last, for server-specific parameters such
    # as `seed` or `response_format`.
    extra_body: dict[str, Any] = {}


class JudgeSettings(_Strict):
    rubric: str
    # Which field of the model's output the judge grades.
    field: str = "summary"
    min_score: int = Field(default=3, ge=1, le=5)


class Task(_Strict):
    name: str = Field(pattern=SAFE_NAME)
    description: str = ""
    endpoint: EndpointEnv = EndpointEnv()
    request: RequestSettings = RequestSettings()
    system_prompt: str = ""
    # `{{name}}` placeholders are filled from each case's `input`.
    prompt_template: str
    output_schema: dict[str, Any]
    # Field values that mean "I decline to classify this".
    abstain: dict[str, Any] = {}
    # Output field to build the confusion table over; needs an `enum`.
    confusion_field: Optional[str] = None
    scorers: list[ScorerSpec] = Field(min_length=1)
    judge: Optional[JudgeSettings] = None


# ---------------------------------------------------------------------------
# Run results
# ---------------------------------------------------------------------------


class RawOutput(BaseModel):
    """What came back for one case, before any scoring.

    Scoring is a pure function of the suite and a list of these, which is
    what lets `report` re-score a saved run without calling the model.
    """

    case_id: str
    # The model's reply verbatim. None when the request itself failed.
    text: Optional[str] = None
    error: Optional[str] = None
    # Only populated when the LLM judge ran for this case.
    judge_text: Optional[str] = None
    judge_error: Optional[str] = None


class ScoreResult(BaseModel):
    scorer: str
    passed: bool
    required: bool = True
    # Numeric grade where the scorer has one (the judge's 1-5).
    score: Optional[float] = None
    detail: str = ""
    data: dict[str, Any] = {}


class CaseResult(BaseModel):
    id: str
    # "error" means no model output was obtained. It is kept apart from
    # "failed" so an endpoint outage does not read as a model regression.
    status: Literal["passed", "failed", "error"]
    ambiguous: bool = False
    expect_abstain: bool = False
    expected: dict[str, Any] = {}
    # The parsed output object, or None when it did not parse to one.
    actual: Optional[dict[str, Any]] = None
    raw_output: Optional[str] = None
    error: Optional[str] = None
    judge_raw_output: Optional[str] = None
    judge_error: Optional[str] = None
    scores: list[ScoreResult] = []


class Totals(BaseModel):
    cases: int
    passed: int
    failed: int
    errors: int
    # passed / cases, as a fraction. None for an empty suite.
    pass_rate: Optional[float]
    # How many replies were wrapped in a markdown code fence, out of how many
    # replies there were. Fences are stripped before scoring; the count is a
    # prompt-quality signal.
    fenced_outputs: int
    outputs: int


class ScorerSummary(BaseModel):
    scorer: str
    required: bool
    applicable: int
    passed: int
    pass_rate: Optional[float]


class Confusion(BaseModel):
    field: str
    # Row order, and column order up to the final catch-all column.
    labels: list[str]
    columns: list[str]
    # matrix[i][j]: cases expecting labels[i] where the model said columns[j].
    matrix: list[list[int]]


class Report(BaseModel):
    schema_version: int
    task: str
    model: str
    # Hostname only. The base URL can carry a path, a port, or credentials
    # that have no business in a file attached to a pull request.
    endpoint_host: str
    timestamp: str
    temperature: Union[int, float]
    judge_enabled: bool = False
    judge_model: Optional[str] = None
    totals: Totals
    by_scorer: list[ScorerSummary]
    confusion: Optional[Confusion] = None
    cases: list[CaseResult]

    @field_validator("schema_version")
    @classmethod
    def _known_version(cls, value: int) -> int:
        if value != 1:
            raise ValueError(f"unsupported report schema_version {value}; this version reads 1")
        return value
