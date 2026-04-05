from __future__ import annotations

from typing import Any

from openenv.core.client_types import StepResult
from openenv.core.env_client import EnvClient

from models import AgentAction, PromptShieldObservation, PromptShieldState


class PromptShieldEnv(EnvClient[AgentAction, PromptShieldObservation, PromptShieldState]):
    def _step_payload(self, action: AgentAction) -> dict[str, Any]:
        return action.model_dump(exclude_none=True)

    def _parse_result(self, payload: dict[str, Any]) -> StepResult[PromptShieldObservation]:
        observation_payload = payload.get("observation", {})
        observation = PromptShieldObservation.model_validate(observation_payload)
        return StepResult(
            observation=observation,
            reward=payload.get("reward"),
            done=payload.get("done", observation.done),
        )

    def _parse_state(self, payload: dict[str, Any]) -> PromptShieldState:
        return PromptShieldState.model_validate(payload)
