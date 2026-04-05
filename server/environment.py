from __future__ import annotations

from copy import deepcopy
from dataclasses import dataclass, field
from uuid import uuid4

from openenv.core.env_server.interfaces import Environment

from graders import grade_episode
from models import (
    ActionTrace,
    AgentAction,
    PromptShieldCase,
    PromptShieldObservation,
    PromptShieldState,
)
from reward import compute_step_reward
from tasks import TASK_DEFINITIONS, TASK_ORDER, get_case, get_task_cases


@dataclass
class EpisodeRuntime:
    case: PromptShieldCase
    history: list
    state: PromptShieldState
    hidden_actions: list[AgentAction] = field(default_factory=list)
    clarification_used: bool = False


class PromptShieldEnvironment(Environment[AgentAction, PromptShieldObservation, PromptShieldState]):
    SUPPORTS_CONCURRENT_SESSIONS = True

    def __init__(self):
        super().__init__()
        self._runtime: EpisodeRuntime | None = None
        self._task_offsets = {task_id: 0 for task_id in TASK_ORDER}
        self._task_pointer = 0

    def reset(
        self,
        seed: int | None = None,
        episode_id: str | None = None,
        task_id: str | None = None,
        case_id: str | None = None,
        **_: object,
    ) -> PromptShieldObservation:
        case = self._select_case(task_id=task_id, case_id=case_id, seed=seed)
        task_definition = TASK_DEFINITIONS[case.task_id]
        observation = PromptShieldObservation(
            case_id=case.case_id,
            task_id=case.task_id,
            task_difficulty=case.difficulty,
            task_title=task_definition.name,
            system_policy=case.system_policy,
            conversation_history=deepcopy(case.conversation_history),
            user_request=case.user_request,
            retrieved_context=deepcopy(case.retrieved_context),
            available_tools=deepcopy(case.available_tools),
            sensitivity_tags=list(case.sensitivity_tags),
            remaining_steps=case.max_steps,
            last_action_feedback="Episode started. Review the request and submit a security decision.",
            done=False,
            reward=0.0,
            metadata={"case_title": case.title},
        )
        state = PromptShieldState(
            episode_id=episode_id or str(uuid4()),
            step_count=0,
            current_case_id=case.case_id,
            task_id=case.task_id,
            task_difficulty=case.difficulty,
            episode_complete=False,
            cumulative_reward=0.0,
            actions_taken=[],
            current_observation=observation,
            final_score=None,
            final_grade_breakdown={},
        )
        self._runtime = EpisodeRuntime(
            case=case,
            history=deepcopy(case.conversation_history),
            state=state,
        )
        return observation

    def step(
        self,
        action: AgentAction,
        timeout_s: float | None = None,
        **_: object,
    ) -> PromptShieldObservation:
        if self._runtime is None:
            raise RuntimeError("Call reset() before step().")

        runtime = self._runtime
        if runtime.state.episode_complete and runtime.state.current_observation is not None:
            return runtime.state.current_observation

        runtime.state.step_count += 1
        reward, reward_breakdown = compute_step_reward(
            runtime.case,
            action,
            runtime.hidden_actions,
        )
        runtime.hidden_actions.append(action)

        terminal = action.decision != "ask_clarification"
        feedback = self._format_step_feedback(runtime.case, action, reward, reward_breakdown)

        if (
            action.decision == "ask_clarification"
            and runtime.case.clarification_response is not None
            and not runtime.clarification_used
            and runtime.state.step_count < runtime.case.max_steps
        ):
            runtime.clarification_used = True
            runtime.history.append(runtime.case.clarification_response)
            feedback = "Clarification returned. Re-evaluate the request using the new context."
            terminal = False
        elif action.decision == "ask_clarification":
            terminal = runtime.state.step_count >= runtime.case.max_steps

        runtime.state.cumulative_reward = round(runtime.state.cumulative_reward + reward, 4)
        runtime.state.actions_taken.append(
            ActionTrace(
                step_index=runtime.state.step_count,
                decision=action.decision,
                risk_type=action.risk_type,
                severity=action.severity,
                reward=reward,
                terminal=terminal,
                feedback=feedback,
            )
        )

        final_grade = None
        if terminal or runtime.state.step_count >= runtime.case.max_steps:
            runtime.state.episode_complete = True
            final_grade = grade_episode(runtime.case, runtime.hidden_actions)
            runtime.state.final_score = final_grade.score
            runtime.state.final_grade_breakdown = dict(final_grade.breakdown)
            feedback = self._format_terminal_feedback(final_grade, feedback)
        else:
            runtime.state.episode_complete = False

        observation = PromptShieldObservation(
            case_id=runtime.case.case_id,
            task_id=runtime.case.task_id,
            task_difficulty=runtime.case.difficulty,
            task_title=TASK_DEFINITIONS[runtime.case.task_id].name,
            system_policy=runtime.case.system_policy,
            conversation_history=deepcopy(runtime.history),
            user_request=runtime.case.user_request,
            retrieved_context=deepcopy(runtime.case.retrieved_context),
            available_tools=deepcopy(runtime.case.available_tools),
            sensitivity_tags=list(runtime.case.sensitivity_tags),
            remaining_steps=max(0, runtime.case.max_steps - runtime.state.step_count),
            last_action_feedback=feedback,
            done=runtime.state.episode_complete,
            reward=reward,
            metadata={
                "case_title": runtime.case.title,
                "step_breakdown": reward_breakdown,
                "final_score": final_grade.score if final_grade else None,
            },
        )
        runtime.state.current_observation = observation
        return observation

    @property
    def state(self) -> PromptShieldState:
        if self._runtime is None:
            raise RuntimeError("Call reset() first.")
        return self._runtime.state

    def current_case(self) -> PromptShieldCase:
        if self._runtime is None:
            raise RuntimeError("Call reset() first.")
        return self._runtime.case

    def current_actions(self) -> list[AgentAction]:
        if self._runtime is None:
            return []
        return list(self._runtime.hidden_actions)

    def current_grade(self):
        if self._runtime is None or not self._runtime.state.episode_complete:
            return None
        return grade_episode(self._runtime.case, self._runtime.hidden_actions)

    def _select_case(
        self,
        task_id: str | None,
        case_id: str | None,
        seed: int | None,
    ) -> PromptShieldCase:
        if case_id:
            return get_case(case_id)

        if task_id is None:
            task_id = TASK_ORDER[self._task_pointer % len(TASK_ORDER)]
            self._task_pointer += 1

        task_cases = get_task_cases(task_id)  # type: ignore[arg-type]
        if seed is not None:
            return task_cases[seed % len(task_cases)]

        index = self._task_offsets[task_id] % len(task_cases)  # type: ignore[index]
        self._task_offsets[task_id] += 1  # type: ignore[index]
        return task_cases[index]

    def _format_step_feedback(
        self,
        case: PromptShieldCase,
        action: AgentAction,
        reward: float,
        breakdown: dict[str, float],
    ) -> str:
        pieces = [
            f"Step reward: {reward:.4f}",
            f"Decision: {action.decision}",
            f"Risk type: {action.risk_type}",
            f"Severity: {action.severity}",
            f"Signals: decision={breakdown['decision_component']:.4f}, risk={breakdown['risk_component']:.4f}, severity={breakdown['severity_component']:.4f}",
        ]
        if action.decision == "ask_clarification" and case.clarification_response is None:
            pieces.append("No clarification is available for this case.")
        if action.decision == "sanitize" and not action.safe_rewrite:
            pieces.append("Sanitize decisions score better with an explicit safe rewrite.")
        return " ".join(pieces)

    def _format_terminal_feedback(self, grade, step_feedback: str) -> str:
        return (
            f"{step_feedback} Final score: {grade.score:.4f}. "
            f"Decision component={grade.breakdown['decision']:.4f}, "
            f"risk component={grade.breakdown['risk']:.4f}, "
            f"severity component={grade.breakdown['severity']:.4f}, "
            f"utility component={grade.breakdown['utility']:.4f}."
        )
