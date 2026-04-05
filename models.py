from __future__ import annotations

from typing import Any, Literal

from openenv.core.env_server.types import Action, Observation, State
from pydantic import BaseModel, Field, field_validator, model_validator

Decision = Literal[
    "allow",
    "block",
    "sanitize",
    "ask_clarification",
    "escalate",
]
RiskType = Literal[
    "benign",
    "direct_injection",
    "indirect_injection",
    "data_exfiltration",
    "jailbreak",
]
Severity = Literal["low", "medium", "high"]
Difficulty = Literal["easy", "medium", "hard"]
TaskId = Literal["task_easy", "task_medium", "task_hard"]


class ConversationTurn(BaseModel):
    role: Literal["system", "user", "assistant", "tool"]
    content: str = Field(min_length=1)


class RetrievedContextItem(BaseModel):
    source: str = Field(min_length=1)
    title: str = Field(min_length=1)
    content: str = Field(min_length=1)
    content_type: str = Field(min_length=1)
    trust_level: Literal["trusted", "mixed", "untrusted"] = "mixed"


class ToolSpec(BaseModel):
    name: str = Field(min_length=1)
    description: str = Field(min_length=1)
    access_level: Literal["low", "medium", "high"] = "low"


class AgentAction(Action):
    decision: Decision
    risk_type: RiskType
    severity: Severity
    justification: str = Field(min_length=8, max_length=320)
    safe_rewrite: str | None = Field(default=None, max_length=800)

    @field_validator("justification")
    @classmethod
    def normalize_justification(cls, value: str) -> str:
        return value.strip()

    @field_validator("safe_rewrite")
    @classmethod
    def normalize_rewrite(cls, value: str | None) -> str | None:
        if value is None:
            return None
        cleaned = value.strip()
        return cleaned or None


class ActionTrace(BaseModel):
    step_index: int = Field(ge=1)
    decision: Decision
    risk_type: RiskType
    severity: Severity
    reward: float
    terminal: bool
    feedback: str


class PromptShieldObservation(Observation):
    case_id: str = Field(min_length=1)
    task_id: TaskId
    task_difficulty: Difficulty
    task_title: str = Field(min_length=1)
    system_policy: str = Field(min_length=1)
    conversation_history: list[ConversationTurn] = Field(default_factory=list)
    user_request: str = Field(min_length=1)
    retrieved_context: list[RetrievedContextItem] = Field(default_factory=list)
    available_tools: list[ToolSpec] = Field(default_factory=list)
    sensitivity_tags: list[str] = Field(default_factory=list)
    remaining_steps: int = Field(ge=0)
    last_action_feedback: str | None = None


class PromptShieldState(State):
    current_case_id: str | None = None
    task_id: TaskId | None = None
    task_difficulty: Difficulty | None = None
    episode_complete: bool = False
    cumulative_reward: float = 0.0
    actions_taken: list[ActionTrace] = Field(default_factory=list)
    current_observation: PromptShieldObservation | None = None
    final_score: float | None = None
    final_grade_breakdown: dict[str, float] = Field(default_factory=dict)


class PromptShieldCase(BaseModel):
    case_id: str = Field(min_length=1)
    task_id: TaskId
    difficulty: Difficulty
    title: str = Field(min_length=1)
    system_policy: str = Field(min_length=1)
    conversation_history: list[ConversationTurn] = Field(default_factory=list)
    user_request: str = Field(min_length=1)
    retrieved_context: list[RetrievedContextItem] = Field(default_factory=list)
    available_tools: list[ToolSpec] = Field(default_factory=list)
    sensitivity_tags: list[str] = Field(default_factory=list)
    max_steps: int = Field(default=2, ge=1, le=4)
    clarification_response: ConversationTurn | None = None
    ground_truth_risk_type: RiskType
    ground_truth_decision: Decision
    ground_truth_severity: Severity
    acceptable_decisions: list[Decision] = Field(default_factory=list)
    acceptable_risk_types: list[RiskType] = Field(default_factory=list)
    allowed_safe_rewrite_patterns: list[str] = Field(default_factory=list)
    rewrite_forbidden_patterns: list[str] = Field(default_factory=list)
    required_justification_keywords: list[str] = Field(default_factory=list)
    utility_keywords: list[str] = Field(default_factory=list)
    explanation_for_readme: str | None = None

    @model_validator(mode="after")
    def populate_defaults(self) -> "PromptShieldCase":
        if not self.acceptable_decisions:
            self.acceptable_decisions = [self.ground_truth_decision]
        if not self.acceptable_risk_types:
            self.acceptable_risk_types = [self.ground_truth_risk_type]
        return self


class TaskDefinition(BaseModel):
    task_id: TaskId
    difficulty: Difficulty
    name: str = Field(min_length=1)
    description: str = Field(min_length=1)
    scoring_summary: str = Field(min_length=1)
    max_steps: int = Field(ge=1)
    case_count: int = Field(default=0, ge=0)


class EpisodeGrade(BaseModel):
    case_id: str
    task_id: TaskId
    score: float = Field(ge=0.0, le=1.0)
    breakdown: dict[str, float]
    penalties: dict[str, float]
    final_decision: Decision


class ResetPayload(BaseModel):
    seed: int | None = Field(default=None, ge=0)
    episode_id: str | None = None
    task_id: TaskId | None = None
    case_id: str | None = None


class StepPayload(BaseModel):
    action: AgentAction
    timeout_s: float | None = Field(default=None, gt=0)


class GraderRequest(BaseModel):
    use_current_episode: bool = True
    case_id: str | None = None
    actions: list[AgentAction] = Field(default_factory=list)


class BaselineRequest(BaseModel):
    mode: Literal["heuristic", "openai"] = "heuristic"
    model: str | None = None
    base_url: str | None = None
    limit_per_task: int | None = Field(default=None, ge=1)


class BaselineCaseScore(BaseModel):
    case_id: str
    task_id: TaskId
    score: float = Field(ge=0.0, le=1.0)
    steps: int = Field(ge=1)


class BaselineTaskSummary(BaseModel):
    task_id: TaskId
    average_score: float = Field(ge=0.0, le=1.0)
    cases_run: int = Field(ge=0)
    case_scores: list[BaselineCaseScore] = Field(default_factory=list)


class BaselineRunSummary(BaseModel):
    mode: Literal["heuristic", "openai"]
    overall_average: float = Field(ge=0.0, le=1.0)
    task_summaries: list[BaselineTaskSummary] = Field(default_factory=list)
    metadata: dict[str, Any] = Field(default_factory=dict)


PromptShieldState.model_rebuild()
