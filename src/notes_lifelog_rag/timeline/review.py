from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
from typing import Any

from notes_lifelog_rag.db.schema import connect, init_db


ACTIVE_STATUS = "active"
REVERTED_STATUS = "reverted"
ITEM_ACTIONS = {"verify", "hide", "dismiss_warning", "mark_needs_fix", "exclude_from_timeline", "pin", "unpin", "request_reanalysis"}
NOTE_ACTIONS = {"exclude_source_note"}
REANALYSIS_ACTIONS = {"mark_needs_fix", "request_reanalysis"}


@dataclass(frozen=True)
class TimelineReviewAction:
    id: str
    month: str
    item_id: str
    source_note_id: str
    action_type: str
    status: str
    reason: str
    comment: str
    quality_flags: list[str]
    item_title: str
    created_at: str
    updated_at: str

    def to_dict(self) -> dict[str, Any]:
        return {
            "id": self.id,
            "month": self.month,
            "item_id": self.item_id,
            "source_note_id": self.source_note_id,
            "action_type": self.action_type,
            "status": self.status,
            "reason": self.reason,
            "comment": self.comment,
            "quality_flags": self.quality_flags,
            "item_title": self.item_title,
            "created_at": self.created_at,
            "updated_at": self.updated_at,
        }


@dataclass(frozen=True)
class ReviewState:
    actions: list[TimelineReviewAction]
    hidden_item_ids: set[str]
    excluded_item_ids: set[str]
    verified_item_ids: set[str]
    pinned_item_ids: set[str]
    needs_fix_item_ids: set[str]
    request_reanalysis_item_ids: set[str]
    dismissed_warnings_by_item: dict[str, set[str]]
    excluded_source_note_ids: set[str]
    actions_by_item: dict[str, list[TimelineReviewAction]]
    actions_by_note: dict[str, list[TimelineReviewAction]]


def upsert_review_action(
    *,
    action_type: str,
    month: str | None = None,
    item_id: str | None = None,
    source_note_id: str | None = None,
    reason: str | None = None,
    comment: str | None = None,
    quality_flags: list[str] | None = None,
    item_title: str | None = None,
    db_path: str | Path | None = None,
) -> TimelineReviewAction:
    action_type = _normalize_action_type(action_type)
    if action_type in ITEM_ACTIONS and not item_id:
        raise ValueError(f"{action_type} requires item_id")
    if action_type in NOTE_ACTIONS and not source_note_id:
        raise ValueError(f"{action_type} requires source_note_id")
    init_db(db_path)
    now = datetime.now(tz=timezone.utc).isoformat()
    action_id = _review_action_id(action_type=action_type, item_id=item_id, source_note_id=source_note_id)
    with connect(db_path) as conn:
        existing = conn.execute(
            "SELECT created_at FROM timeline_review_actions WHERE id = ?",
            (action_id,),
        ).fetchone()
        created_at = existing["created_at"] if existing else now
        conn.execute(
            """
            INSERT INTO timeline_review_actions(
                id, month, item_id, source_note_id, action_type, status, reason,
                comment, quality_flags_json, item_title, created_at, updated_at
            )
            VALUES (?, ?, ?, ?, ?, 'active', ?, ?, ?, ?, ?, ?)
            ON CONFLICT(id) DO UPDATE SET
                month = excluded.month,
                source_note_id = excluded.source_note_id,
                status = 'active',
                reason = excluded.reason,
                comment = excluded.comment,
                quality_flags_json = excluded.quality_flags_json,
                item_title = excluded.item_title,
                updated_at = excluded.updated_at
            """,
            (
                action_id,
                month or "",
                item_id or "",
                source_note_id or "",
                action_type,
                reason or "",
                comment or "",
                json.dumps(quality_flags or [], ensure_ascii=False),
                item_title or "",
                created_at,
                now,
            ),
        )
    return get_review_action(action_id, db_path=db_path)


def revert_review_action(action_id: str, *, db_path: str | Path | None = None) -> TimelineReviewAction | None:
    init_db(db_path)
    now = datetime.now(tz=timezone.utc).isoformat()
    with connect(db_path) as conn:
        conn.execute(
            """
            UPDATE timeline_review_actions
            SET status = 'reverted', updated_at = ?
            WHERE id = ?
            """,
            (now, action_id),
        )
    return get_review_action(action_id, db_path=db_path)


def get_review_action(action_id: str, *, db_path: str | Path | None = None) -> TimelineReviewAction:
    init_db(db_path)
    with connect(db_path) as conn:
        row = conn.execute("SELECT * FROM timeline_review_actions WHERE id = ?", (action_id,)).fetchone()
    if not row:
        raise ValueError(f"timeline review action not found: {action_id}")
    return _action_from_row(row)


def list_review_actions(
    *,
    month: str | None = None,
    source_note_id: str | None = None,
    status: str | None = None,
    db_path: str | Path | None = None,
) -> list[TimelineReviewAction]:
    init_db(db_path)
    where: list[str] = []
    params: list[Any] = []
    if month:
        where.append("month = ?")
        params.append(month)
    if source_note_id:
        where.append("source_note_id = ?")
        params.append(source_note_id)
    if status:
        where.append("status = ?")
        params.append(status)
    where_sql = f"WHERE {' AND '.join(where)}" if where else ""
    with connect(db_path) as conn:
        rows = conn.execute(
            f"""
            SELECT *
            FROM timeline_review_actions
            {where_sql}
            ORDER BY updated_at DESC, created_at DESC
            """,
            params,
        ).fetchall()
    return [_action_from_row(row) for row in rows]


def active_review_state(*, month: str | None = None, db_path: str | Path | None = None) -> ReviewState:
    actions = [
        action
        for action in list_review_actions(db_path=db_path)
        if action.status == ACTIVE_STATUS and (not month or not action.month or action.month == month or action.action_type == "exclude_source_note")
    ]
    actions_by_item: dict[str, list[TimelineReviewAction]] = {}
    actions_by_note: dict[str, list[TimelineReviewAction]] = {}
    dismissed: dict[str, set[str]] = {}
    for action in actions:
        if action.item_id:
            actions_by_item.setdefault(action.item_id, []).append(action)
        if action.source_note_id:
            actions_by_note.setdefault(action.source_note_id, []).append(action)
        if action.action_type == "dismiss_warning" and action.item_id and action.reason:
            dismissed.setdefault(action.item_id, set()).add(action.reason)
    return ReviewState(
        actions=actions,
        hidden_item_ids={action.item_id for action in actions if action.action_type == "hide" and action.item_id},
        excluded_item_ids={action.item_id for action in actions if action.action_type == "exclude_from_timeline" and action.item_id},
        verified_item_ids={action.item_id for action in actions if action.action_type == "verify" and action.item_id},
        pinned_item_ids={action.item_id for action in actions if action.action_type == "pin" and action.item_id},
        needs_fix_item_ids={action.item_id for action in actions if action.action_type == "mark_needs_fix" and action.item_id},
        request_reanalysis_item_ids={action.item_id for action in actions if action.action_type == "request_reanalysis" and action.item_id},
        dismissed_warnings_by_item=dismissed,
        excluded_source_note_ids={action.source_note_id for action in actions if action.action_type == "exclude_source_note" and action.source_note_id},
        actions_by_item=actions_by_item,
        actions_by_note=actions_by_note,
    )


def review_statuses_for_item(item_id: str, source_note_id: str, state: ReviewState) -> list[str]:
    statuses: list[str] = []
    item_actions = state.actions_by_item.get(item_id, [])
    note_actions = state.actions_by_note.get(source_note_id, [])
    if any(action.action_type == "verify" for action in item_actions):
        statuses.append("verified")
    if any(action.action_type in {"hide", "exclude_from_timeline"} for action in item_actions):
        statuses.append("hidden")
    if any(action.action_type == "pin" for action in item_actions):
        statuses.append("pinned")
    if any(action.action_type in REANALYSIS_ACTIONS for action in item_actions):
        statuses.append("needs_fix")
    if any(action.action_type == "exclude_source_note" for action in note_actions):
        statuses.append("excluded_source")
    if any(action.action_type == "dismiss_warning" for action in item_actions):
        statuses.append("warning_dismissed")
    return sorted(set(statuses))


def action_counts_for_items(items: list[Any], state: ReviewState) -> dict[str, int]:
    item_ids = {str(getattr(item, "id", "") or "") for item in items}
    note_ids = {str(getattr(item, "source_note_id", "") or "") for item in items}
    return {
        "reviewed_items_count": sum(1 for item_id in item_ids if state.actions_by_item.get(item_id)),
        "hidden_items_count": len(item_ids & (state.hidden_item_ids | state.excluded_item_ids)),
        "verified_items_count": len(item_ids & state.verified_item_ids),
        "needs_fix_count": len(item_ids & (state.needs_fix_item_ids | state.request_reanalysis_item_ids)),
        "excluded_source_notes_count": len(note_ids & state.excluded_source_note_ids),
    }


def list_excluded_notes(*, db_path: str | Path | None = None) -> list[TimelineReviewAction]:
    return [
        action
        for action in list_review_actions(status=ACTIVE_STATUS, db_path=db_path)
        if action.action_type == "exclude_source_note"
    ]


def reanalysis_plan_rows(*, month: str | None = None, db_path: str | Path | None = None) -> list[dict[str, Any]]:
    init_db(db_path)
    actions = [
        action
        for action in list_review_actions(month=month, status=ACTIVE_STATUS, db_path=db_path)
        if action.action_type in REANALYSIS_ACTIONS
    ]
    output: list[dict[str, Any]] = []
    with connect(db_path) as conn:
        for action in actions:
            note = None
            if action.source_note_id:
                note = conn.execute(
                    "SELECT id, title, source_relative_path FROM notes WHERE id = ?",
                    (action.source_note_id,),
                ).fetchone()
            flags = set(action.quality_flags)
            if "weak_evidence" in flags or "title_only_evidence" in flags:
                task = "evidence-enrichment"
                command = f"python -m notes_lifelog_rag.cli analyze-sample --note-id {action.source_note_id} --task summary --show-raw-output"
            elif "low_confidence" in flags:
                task = "summarize"
                command = f"python -m notes_lifelog_rag.cli summarize-notes --note-id {action.source_note_id} --force"
            elif "date_uncertain" in flags:
                task = "extract-events"
                command = f"python -m notes_lifelog_rag.cli extract-events --note-id {action.source_note_id} --force"
            else:
                task = "extract-thoughts"
                command = f"python -m notes_lifelog_rag.cli extract-thoughts --note-id {action.source_note_id} --force"
            output.append(
                {
                    "action_id": action.id,
                    "month": action.month,
                    "item_id": action.item_id,
                    "note_id": action.source_note_id,
                    "title": (note["title"] if note else action.item_title) or action.item_title,
                    "source_path": note["source_relative_path"] if note else "",
                    "reason": action.reason or action.comment,
                    "recommended_task": task,
                    "suggested_command": command,
                }
            )
    return output


def format_reanalysis_plan(rows: list[dict[str, Any]]) -> str:
    lines = ["# Timeline Reanalysis Plan", ""]
    if not rows:
        lines.append("No active needs-fix or request-reanalysis timeline review actions.")
        return "\n".join(lines).rstrip() + "\n"
    for row in rows:
        lines.extend(
            [
                f"## {row.get('title') or row.get('note_id') or 'Untitled'}",
                f"- month: {row.get('month') or ''}",
                f"- note_id: {row.get('note_id') or ''}",
                f"- item_id: {row.get('item_id') or ''}",
                f"- reason: {row.get('reason') or ''}",
                f"- recommended task: {row.get('recommended_task') or ''}",
                "",
                "```bash",
                str(row.get("suggested_command") or ""),
                "```",
                "",
            ]
        )
    return "\n".join(lines).rstrip() + "\n"


def _action_from_row(row) -> TimelineReviewAction:
    return TimelineReviewAction(
        id=str(row["id"] or ""),
        month=str(row["month"] or ""),
        item_id=str(row["item_id"] or ""),
        source_note_id=str(row["source_note_id"] or ""),
        action_type=str(row["action_type"] or ""),
        status=str(row["status"] or ""),
        reason=str(row["reason"] or ""),
        comment=str(row["comment"] or ""),
        quality_flags=_json_list(row["quality_flags_json"]),
        item_title=str(row["item_title"] or ""),
        created_at=str(row["created_at"] or ""),
        updated_at=str(row["updated_at"] or ""),
    )


def _json_list(payload: str | None) -> list[str]:
    try:
        value = json.loads(payload or "[]")
    except (TypeError, json.JSONDecodeError):
        return []
    if isinstance(value, list):
        return [str(item) for item in value]
    return []


def _normalize_action_type(action_type: str) -> str:
    value = str(action_type or "").strip()
    aliases = {
        "needs-fix": "mark_needs_fix",
        "needs_fix": "mark_needs_fix",
        "exclude-note": "exclude_source_note",
        "restore-note": "exclude_source_note",
    }
    value = aliases.get(value, value)
    allowed = ITEM_ACTIONS | NOTE_ACTIONS
    if value not in allowed:
        raise ValueError(f"unsupported timeline review action: {action_type}")
    return value


def _review_action_id(*, action_type: str, item_id: str | None, source_note_id: str | None) -> str:
    key = "|".join([action_type, item_id or "", source_note_id or ""])
    digest = hashlib.sha1(key.encode("utf-8")).hexdigest()[:20]
    return f"tra_{digest}"
