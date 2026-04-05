from env import PromptShieldEnvironment
from graders import grade_episode
from models import AgentAction


def test_reset_works() -> None:
    env = PromptShieldEnvironment()
    observation = env.reset(case_id="easy_003")
    assert observation.case_id == "easy_003"
    assert observation.task_id == "task_easy"
    assert observation.done is False
    assert env.state.current_case_id == "easy_003"


def test_step_works() -> None:
    env = PromptShieldEnvironment()
    env.reset(case_id="easy_001")
    observation = env.step(
        AgentAction(
            decision="block",
            risk_type="direct_injection",
            severity="high",
            justification="This request tries to override policy and reveal hidden instructions.",
        )
    )
    assert observation.done is True
    assert env.state.episode_complete is True
    assert env.state.final_score is not None
    assert 0.0 <= env.state.final_score <= 1.0


def test_grader_score_range_is_zero_to_one() -> None:
    env = PromptShieldEnvironment()
    env.reset(case_id="hard_005")
    action = AgentAction(
        decision="sanitize",
        risk_type="data_exfiltration",
        severity="medium",
        justification="The request would broadcast individual ratings, so it should be rewritten into an approved team-level summary.",
        safe_rewrite="Draft an aggregate summary for the team that notes the review cycle is complete and excludes any individual ratings.",
    )
    grade = grade_episode(env.current_case(), [action])
    assert 0.0 <= grade.score <= 1.0
