"""Mastery Path tools — the seam between the chat-loop tutor and the pure
mastery engine (:mod:`deeptutor.learning`).

These five tools are auto-mounted only when a mastery path is active on the
turn (via the chat loop mastery capability). The chat agent loop IS the tutor;
these tools let it read the gate and record outcomes, while the pedagogy —
what to teach, how to question, when to explain — stays the model's job. The
arithmetic (mastery, gate, spaced repetition) stays in the engine.

The active path id is injected server-side by the pipeline as
``_mastery_path_id``; the model never supplies it. Each call constructs a
fresh store + service (matching the REST router) so concurrent turns can't
race on a shared object.
"""

from __future__ import annotations

import json
import logging
import re
from typing import TYPE_CHECKING, Any
import uuid

from deeptutor.core.tool_protocol import BaseTool, ToolDefinition, ToolParameter, ToolResult

# ``learning.models`` and ``learning.policy`` only depend on pydantic — safe to
# import at module load. ``learning.service`` / ``storage`` / ``scheduler``
# reach the path service (and so the runtime + tool registry), so importing
# them here would close an import cycle through the built-in registry. They
# are imported lazily inside the call paths instead (same pattern as the other
# builtin tools).
from deeptutor.learning.models import (
    KnowledgePoint,
    KnowledgeType,
    LearningModule,
    PendingQuestion,
)
from deeptutor.learning.policy import (
    QUALITATIVE_TYPES,
    display_mastery,
    find_knowledge_point,
    gate_threshold,
    is_mastered,
    map_summary,
    next_objective,
)

if TYPE_CHECKING:
    from deeptutor.learning.service import LearningService

# Tool names the pipeline mounts together when a mastery path is active. Kept
# here so the mount policy and the registration list can't disagree.
MASTERY_TOOL_NAMES: tuple[str, ...] = (
    "mastery_status",
    "mastery_quiz",
    "mastery_grade",
    "mastery_assess",
    "mastery_build",
)

_QUESTION_TYPES = ("choice", "short", "open")
_ALLOWED_KP_TYPES = {t.value for t in KnowledgeType}
logger = logging.getLogger(__name__)

_OPTION_PREFIX_RE = re.compile(r"^\s*([A-Z])\s*[.:：、)）-]\s*(.+)$", re.IGNORECASE)


def _new_service() -> LearningService:
    from deeptutor.learning.service import LearningService
    from deeptutor.learning.storage import LearningStore

    return LearningService(LearningStore())


def _resolve_path_id(kwargs: dict[str, Any]) -> str:
    return str(kwargs.get("_mastery_path_id") or "").strip()


def _resolve_session_id(kwargs: dict[str, Any]) -> str:
    return str(kwargs.get("_session_id") or "").strip()


def _resolve_turn_id(kwargs: dict[str, Any]) -> str:
    return str(kwargs.get("_turn_id") or "").strip()


def _question_bank_type(question_type: str) -> str:
    qtype = str(question_type or "").strip().lower()
    if qtype == "choice":
        return "choice"
    if qtype == "open":
        return "written"
    return "short_answer"


def _question_bank_options(options: list[str]) -> dict[str, str]:
    result: dict[str, str] = {}
    for idx, raw in enumerate(options):
        text = str(raw or "").strip()
        if not text:
            continue
        match = _OPTION_PREFIX_RE.match(text)
        if match:
            key = match.group(1).upper()
            result[key] = match.group(2).strip()
        elif len(text) == 1 and text.isalnum():
            key = text.upper()
            result[key] = text
        else:
            key = chr(ord("A") + idx) if idx < 26 else str(idx + 1)
            result[key] = text
    return result


def _has_choice_option_bodies(options: dict[str, str]) -> bool:
    """Whether a choice map contains real answer text, not only A/B/C labels."""
    return len(options) >= 2 and all(
        value.strip() and value.strip().upper() != key.upper() for key, value in options.items()
    )


def _normalized_choice_prompt(value: str) -> str:
    return "".join(char.casefold() for char in str(value or "") if char.isalnum())


def _resolve_choice_answer(expected_answer: str, options: dict[str, str]) -> str:
    """Resolve a model-supplied choice answer to its stable option label.

    Models occasionally send ``"Step 6"`` or the full option text even though
    the interactive card returns ``"C"``.  Resolve a unique textual match at
    registration time so deterministic grading compares like with like.
    """
    expected = str(expected_answer or "").strip()
    key = expected.upper()
    if key in options:
        return key
    if not expected:
        return ""

    prefix_match = _OPTION_PREFIX_RE.match(expected)
    if prefix_match and prefix_match.group(1).upper() in options:
        return prefix_match.group(1).upper()

    needle = expected.casefold()
    exact = [label for label, text in options.items() if text.casefold() == needle]
    if len(exact) == 1:
        return exact[0]
    contained = [label for label, text in options.items() if needle in text.casefold()]
    return contained[0] if len(contained) == 1 else ""


async def _recover_choice_options_from_turn(
    store: Any,
    turn_id: str,
    question: str,
) -> dict[str, str]:
    """Recover choice descriptions from the most recent matching ask_user card.

    This is a compatibility fallback for questions registered by older
    versions, where ``mastery_quiz`` persisted only ``["A", "B", ...]`` even
    though the full descriptions were present in the turn's ``ask_user``
    event.
    """
    if not turn_id or not hasattr(store, "get_turn_events"):
        return {}
    try:
        events = await store.get_turn_events(turn_id)
    except Exception:
        logger.warning("Failed to load turn events for mastery option recovery", exc_info=True)
        return {}

    target = _normalized_choice_prompt(question)
    for event in reversed(events):
        if event.get("type") != "tool_call":
            continue
        metadata = event.get("metadata") or {}
        if metadata.get("tool_name") != "ask_user":
            continue
        questions = (metadata.get("args") or {}).get("questions") or []
        for item in reversed(questions):
            if not isinstance(item, dict):
                continue
            recovered = {
                str(option.get("label") or "").strip().upper(): str(
                    option.get("description") or ""
                ).strip()
                for option in (item.get("options") or [])
                if isinstance(option, dict)
                and str(option.get("label") or "").strip()
                and str(option.get("description") or "").strip()
            }
            if not _has_choice_option_bodies(recovered):
                continue
            prompt = _normalized_choice_prompt(str(item.get("prompt") or ""))
            if prompt == target or prompt.startswith(target) or target.startswith(prompt):
                return recovered
    return {}


async def _sync_mastery_attempt_to_question_bank(
    *,
    session_id: str,
    turn_id: str,
    pending: PendingQuestion,
    user_answer: str,
    is_correct: bool,
    choice_options: dict[str, str] | None = None,
    correct_answer: str | None = None,
) -> None:
    if not session_id:
        return
    item = {
        "turn_id": turn_id,
        "question_id": pending.question_id,
        "question": pending.prompt,
        "question_type": _question_bank_type(pending.question_type),
        "options": choice_options or _question_bank_options(list(pending.options or [])),
        "correct_answer": correct_answer or pending.expected_answer,
        "explanation": "",
        "difficulty": "",
        "user_answer": user_answer,
        "is_correct": is_correct,
    }
    try:
        from deeptutor.services.session import get_sqlite_session_store

        await get_sqlite_session_store().upsert_notebook_entries(session_id, [item])
    except Exception:
        logger.warning(
            "Failed to sync mastery question %s to question bank for session %s",
            pending.question_id,
            session_id,
            exc_info=True,
        )


def _json_result(payload: dict[str, Any], *, meta_key: str, success: bool = True) -> ToolResult:
    return ToolResult(
        content=json.dumps(payload, ensure_ascii=False),
        success=success,
        metadata={meta_key: payload},
    )


def _no_path_result() -> ToolResult:
    return ToolResult(
        content="No mastery path is active on this turn; mastery tools are unavailable.",
        success=False,
    )


class MasteryStatusTool(BaseTool):
    """Read the current objective + map snapshot. Call FIRST every turn."""

    def get_definition(self) -> ToolDefinition:
        return ToolDefinition(
            name="mastery_status",
            description=(
                "Read the learner's mastery path: the next objective to work on "
                "(decided by a hard mastery gate), any question awaiting an "
                "answer, due reviews, and a map of every objective's status "
                "(new / learning / mastered). Call this FIRST on every mastery "
                "turn — it tells you what to do; never guess the next objective."
            ),
            parameters=[],
        )

    async def execute(self, **kwargs: Any) -> ToolResult:
        path_id = _resolve_path_id(kwargs)
        if not path_id:
            return _no_path_result()
        service = _new_service()
        progress = service.get_or_create(path_id)
        if not any(module.knowledge_points for module in progress.modules):
            return _json_result(
                {
                    "status": "empty",
                    "message": (
                        "No mastery path has been built yet. Design one from the "
                        "learner's materials and call mastery_build."
                    ),
                },
                meta_key="mastery_status",
            )
        payload = {
            "status": "active",
            "next": next_objective(progress).to_dict(),
            "map": map_summary(progress),
        }
        return _json_result(payload, meta_key="mastery_status")


class MasteryQuizTool(BaseTool):
    """Register an objective-type question; the engine holds the answer."""

    def get_definition(self) -> ToolDefinition:
        return ToolDefinition(
            name="mastery_quiz",
            description=(
                "Pose a question for a MEMORY or PROCEDURE objective and register "
                "its expected answer with the engine (so grading is deterministic "
                "and you never re-state the answer later). After calling this, "
                "present the question with the ask_user tool so the learner answers "
                "on an interactive card (for choices, give ask_user options short "
                "labels like A/B/C, pass every full option body here, and set the "
                "correct label as expected_answer); "
                "then call mastery_grade with their answer. For CONCEPT / DESIGN "
                "objectives use mastery_assess instead."
            ),
            parameters=[
                ToolParameter(
                    name="knowledge_point_id",
                    type="string",
                    description="Objective id from mastery_status (verbatim).",
                ),
                ToolParameter(
                    name="question",
                    type="string",
                    description="The question text shown to the learner.",
                ),
                ToolParameter(
                    name="expected_answer",
                    type="string",
                    description="The correct answer, used only server-side for grading.",
                ),
                ToolParameter(
                    name="question_type",
                    type="string",
                    description=(
                        "'choice' (exact match), 'short' (exact / fuzzy for ≤30 "
                        "chars), or 'open' (keyword overlap). Default 'short'."
                    ),
                    required=False,
                    default="short",
                    enum=list(_QUESTION_TYPES),
                ),
                ToolParameter(
                    name="options",
                    type="array",
                    description=(
                        "For question_type='choice', every full option in label order, "
                        "for example ['A: first answer', 'B: second answer']. Never "
                        "pass bare labels such as ['A', 'B', 'C', 'D']. Use the same "
                        "bodies as the ask_user option descriptions."
                    ),
                    required=False,
                    items={"type": "string"},
                ),
            ],
        )

    async def execute(self, **kwargs: Any) -> ToolResult:
        path_id = _resolve_path_id(kwargs)
        if not path_id:
            return _no_path_result()
        kp_id = str(kwargs.get("knowledge_point_id") or "").strip()
        question = str(kwargs.get("question") or "").strip()
        expected = str(kwargs.get("expected_answer") or "").strip()
        if not kp_id or not question or not expected:
            return ToolResult(
                content="mastery_quiz needs knowledge_point_id, question, and expected_answer.",
                success=False,
            )
        q_type = str(kwargs.get("question_type") or "short").strip().lower()
        if q_type not in _QUESTION_TYPES:
            q_type = "short"
        options = [str(o) for o in (kwargs.get("options") or []) if str(o).strip()]
        if q_type == "choice":
            choice_options = _question_bank_options(options)
            if not _has_choice_option_bodies(choice_options):
                return ToolResult(
                    content=(
                        "Choice questions need full option bodies in mastery_quiz.options "
                        "(for example ['A: first answer', 'B: second answer']), not only "
                        "the labels A/B/C/D. Retry mastery_quiz with the exact option "
                        "descriptions you will show through ask_user."
                    ),
                    success=False,
                )
            resolved_expected = _resolve_choice_answer(expected, choice_options)
            if not resolved_expected:
                return ToolResult(
                    content=(
                        "Choice expected_answer must be an option label such as A/B/C/D, "
                        "or uniquely match one full option body. Retry mastery_quiz with "
                        "the correct label."
                    ),
                    success=False,
                )
            expected = resolved_expected
            options = [f"{key}: {text}" for key, text in choice_options.items()]

        service = _new_service()
        progress = service.get_or_create(path_id)
        kp, module_id, _ = find_knowledge_point(progress, kp_id)
        if kp is None:
            return ToolResult(
                content=f"Unknown objective {kp_id!r}; call mastery_status for valid ids.",
                success=False,
            )
        pending = PendingQuestion(
            question_id=uuid.uuid4().hex,
            knowledge_point_id=kp_id,
            module_id=module_id,
            prompt=question,
            question_type=q_type,
            expected_answer=expected,
            options=options,
        )
        service.set_pending_question(progress, pending)
        return _json_result(
            {
                "status": "registered",
                "knowledge_point_id": kp_id,
                "question": question,
                "options": options,
                "instruction": (
                    "Present this question with the ask_user tool (use its options "
                    "for multiple choice; the option labels must match the "
                    "expected_answer you registered), then call mastery_grade with "
                    "the learner's answer."
                ),
            },
            meta_key="mastery_quiz",
        )


class MasteryGradeTool(BaseTool):
    """Grade the learner's answer to the pending question (deterministic)."""

    def get_definition(self) -> ToolDefinition:
        return ToolDefinition(
            name="mastery_grade",
            description=(
                "Grade the learner's answer to the question you registered with "
                "mastery_quiz. Grading is deterministic against the stored "
                "expected answer; this updates mastery, advances spaced "
                "repetition, and tells you whether the objective's gate is now "
                "cleared. Then give the learner feedback."
            ),
            parameters=[
                ToolParameter(
                    name="answer",
                    type="string",
                    description="The learner's answer, verbatim.",
                ),
            ],
        )

    async def execute(self, **kwargs: Any) -> ToolResult:
        path_id = _resolve_path_id(kwargs)
        if not path_id:
            return _no_path_result()
        from deeptutor.learning.scheduler import SpacedRepetitionScheduler

        answer = str(kwargs.get("answer") or "")
        service = _new_service()
        scheduler = SpacedRepetitionScheduler()
        progress = service.get_or_create(path_id)
        pending = progress.pending_question
        if pending is None:
            return ToolResult(
                content="No question is awaiting an answer. Pose one with mastery_quiz first.",
                success=False,
            )
        choice_options: dict[str, str] = {}
        expected_answer = pending.expected_answer
        if pending.question_type == "choice":
            choice_options = _question_bank_options(list(pending.options or []))
            if not _has_choice_option_bodies(choice_options):
                try:
                    from deeptutor.services.session import get_sqlite_session_store

                    choice_options = await _recover_choice_options_from_turn(
                        get_sqlite_session_store(),
                        _resolve_turn_id(kwargs),
                        pending.prompt,
                    )
                except Exception:
                    logger.warning("Failed to recover legacy mastery choice options", exc_info=True)
            resolved_expected = _resolve_choice_answer(expected_answer, choice_options)
            if resolved_expected:
                expected_answer = resolved_expected

        is_correct = service.grade_and_record(
            progress,
            question_id=pending.question_id,
            knowledge_point_id=pending.knowledge_point_id,
            module_id=pending.module_id,
            user_answer=answer,
            expected_answer=expected_answer,
            question_type=pending.question_type,
            scheduler=scheduler,
        )
        await _sync_mastery_attempt_to_question_bank(
            session_id=_resolve_session_id(kwargs),
            turn_id=_resolve_turn_id(kwargs),
            pending=pending,
            user_answer=answer,
            is_correct=is_correct,
            choice_options=choice_options,
            correct_answer=expected_answer,
        )
        service.clear_pending_question(progress)
        kp, _, _ = find_knowledge_point(progress, pending.knowledge_point_id)
        mastered = bool(kp and is_mastered(progress, kp))
        payload = {
            "is_correct": is_correct,
            "knowledge_point_id": pending.knowledge_point_id,
            "mastery": round(display_mastery(progress, kp), 3) if kp else 0.0,
            "threshold": round(gate_threshold(kp.type), 3) if kp else 0.0,
            "mastered": mastered,
            "next": next_objective(progress).to_dict(),
        }
        return _json_result(payload, meta_key="mastery_grade")


class MasteryAssessTool(BaseTool):
    """Record the qualitative (CONCEPT / DESIGN) gate from a Feynman check."""

    def get_definition(self) -> ToolDefinition:
        return ToolDefinition(
            name="mastery_assess",
            description=(
                "Record your judgement of a CONCEPT or DESIGN objective after the "
                "learner explains it in their own words (a Feynman-style check). "
                "Pass passed=true only when the explanation is correct and "
                "complete enough to count as mastery — this is the gate for these "
                "objective types. For MEMORY / PROCEDURE objectives use "
                "mastery_quiz + mastery_grade instead."
            ),
            parameters=[
                ToolParameter(
                    name="knowledge_point_id",
                    type="string",
                    description="Objective id from mastery_status (verbatim).",
                ),
                ToolParameter(
                    name="passed",
                    type="boolean",
                    description="True if the explanation demonstrates mastery.",
                ),
                ToolParameter(
                    name="feedback",
                    type="string",
                    description="Short note on what was strong or missing (stored as evidence).",
                    required=False,
                ),
            ],
        )

    async def execute(self, **kwargs: Any) -> ToolResult:
        path_id = _resolve_path_id(kwargs)
        if not path_id:
            return _no_path_result()
        kp_id = str(kwargs.get("knowledge_point_id") or "").strip()
        if not kp_id:
            return ToolResult(content="mastery_assess needs a knowledge_point_id.", success=False)
        passed = bool(kwargs.get("passed"))
        feedback = str(kwargs.get("feedback") or "").strip()

        service = _new_service()
        progress = service.get_or_create(path_id)
        kp, _, _ = find_knowledge_point(progress, kp_id)
        if kp is None:
            return ToolResult(
                content=f"Unknown objective {kp_id!r}; call mastery_status for valid ids.",
                success=False,
            )
        if kp.type not in QUALITATIVE_TYPES:
            return ToolResult(
                content=(
                    f"Objective {kp.name!r} is a {kp.type.value} type — gate it with "
                    "mastery_quiz + mastery_grade, not mastery_assess."
                ),
                success=False,
            )
        service.record_qualitative(progress, kp_id, passed=passed, evidence=feedback)
        payload = {
            "knowledge_point_id": kp_id,
            "passed": passed,
            "mastered": is_mastered(progress, kp),
            "mastery": round(display_mastery(progress, kp), 3),
            "next": next_objective(progress).to_dict(),
        }
        return _json_result(payload, meta_key="mastery_assess")


class MasteryBuildTool(BaseTool):
    """Create / extend the skill map from objectives the tutor designed."""

    def get_definition(self) -> ToolDefinition:
        return ToolDefinition(
            name="mastery_build",
            description=(
                "Create or extend the learner's mastery path. Design modules and "
                "their knowledge points from the learner's materials (use rag / "
                "read_source first when materials are attached) and pass them "
                "here. Each knowledge point needs a 'type': memory (facts), "
                "procedure (step-by-step skills), concept (ideas to understand), "
                "or design (open-ended judgement). Use mode='replace' to start "
                "fresh or 'append' to add to an existing path."
            ),
            parameters=[
                ToolParameter(
                    name="modules",
                    type="array",
                    description=(
                        "Ordered modules: each {name, knowledge_points: [{name, "
                        "type}]}. type is one of memory/procedure/concept/design."
                    ),
                    items={
                        "type": "object",
                        "properties": {
                            "name": {"type": "string"},
                            "knowledge_points": {
                                "type": "array",
                                "items": {
                                    "type": "object",
                                    "properties": {
                                        "name": {"type": "string"},
                                        "type": {
                                            "type": "string",
                                            "enum": sorted(_ALLOWED_KP_TYPES),
                                        },
                                    },
                                    "required": ["name"],
                                },
                            },
                        },
                        "required": ["name", "knowledge_points"],
                    },
                ),
                ToolParameter(
                    name="mode",
                    type="string",
                    description="'replace' (default) starts fresh; 'append' adds modules.",
                    required=False,
                    default="replace",
                    enum=["replace", "append"],
                ),
            ],
        )

    async def execute(self, **kwargs: Any) -> ToolResult:
        path_id = _resolve_path_id(kwargs)
        if not path_id:
            return _no_path_result()
        mode = str(kwargs.get("mode") or "replace").strip().lower()
        if mode not in {"replace", "append"}:
            mode = "replace"

        service = _new_service()
        progress = service.get_or_create(path_id)
        offset = len(progress.modules) if mode == "append" else 0
        new_modules, error = _parse_modules(kwargs.get("modules"), path_id, offset)
        if error:
            return ToolResult(content=error, success=False)

        combined = (list(progress.modules) + new_modules) if mode == "append" else new_modules
        service.replace_modules(progress, combined)
        progress.pending_question = None  # a rebuilt map invalidates any open question
        if combined:
            progress.current_module_id = combined[0].id
            progress.current_kp_index = 0
        service.save(progress)
        kp_count = sum(len(m.knowledge_points) for m in new_modules)
        return _json_result(
            {
                "status": "built",
                "mode": mode,
                "modules_added": len(new_modules),
                "knowledge_points_added": kp_count,
                "map": map_summary(progress),
            },
            meta_key="mastery_build",
        )


def _parse_modules(
    raw_modules: Any, path_id: str, offset: int
) -> tuple[list[LearningModule], str | None]:
    """Validate the model-designed module tree into engine models.

    Ids are generated server-side (``<path>_m<i>_kp<j>``) so the model never
    controls storage keys; unknown knowledge types fall back to 'concept'.
    """
    if not isinstance(raw_modules, list) or not raw_modules:
        return [], "mastery_build needs a non-empty 'modules' array."
    modules: list[LearningModule] = []
    for i, raw in enumerate(raw_modules):
        if not isinstance(raw, dict):
            continue
        index = offset + i
        name = str(raw.get("name") or "").strip()[:200]
        if not name:
            continue
        module_id = f"{path_id}_m{index}"
        kps: list[KnowledgePoint] = []
        for j, raw_kp in enumerate(raw.get("knowledge_points") or []):
            if not isinstance(raw_kp, dict):
                continue
            kp_name = str(raw_kp.get("name") or "").strip()[:200]
            if len(kp_name) < 2:
                continue
            kp_type = str(raw_kp.get("type") or "concept").strip().lower()
            if kp_type not in _ALLOWED_KP_TYPES:
                kp_type = "concept"
            kps.append(
                KnowledgePoint(
                    id=f"{module_id}_kp{j}",
                    name=kp_name,
                    type=KnowledgeType(kp_type),
                    module_id=module_id,
                )
            )
        if not kps:
            continue
        modules.append(LearningModule(id=module_id, name=name, order=index, knowledge_points=kps))
    if not modules:
        return [], "No valid modules: each module needs a name and at least one knowledge point."
    return modules, None


MASTERY_TOOL_TYPES: tuple[type[BaseTool], ...] = (
    MasteryStatusTool,
    MasteryQuizTool,
    MasteryGradeTool,
    MasteryAssessTool,
    MasteryBuildTool,
)


__all__ = [
    "MASTERY_TOOL_NAMES",
    "MASTERY_TOOL_TYPES",
    "MasteryStatusTool",
    "MasteryQuizTool",
    "MasteryGradeTool",
    "MasteryAssessTool",
    "MasteryBuildTool",
]
