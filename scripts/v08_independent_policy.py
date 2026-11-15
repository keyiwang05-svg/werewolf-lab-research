"""Independent, offline Werewolf seat policy for the simplified eight-player Lab.

Important boundaries:
* Uses only a detached, seat-filtered Observation and legal numbered options.
* Does not import strategy_families.py or diagnostics.policy.
* Public statements are not treated as ground-truth role revelations.
* RNG is supplied by each seat; no model API or external network required.

The behavior is a NEW policy implementation, not an independent game simulator
or human-behavior distribution.
"""
from __future__ import annotations

import random
import re
from collections import defaultdict
from typing import Any, Mapping, Sequence

Choice = tuple[str, str]  # (number to return to engine, player_id)

_CHOICES_MARKER = "\nAvailable targets:\n"
_CHOICES_RE = re.compile(r"^\s*(\d+)\.\s+.*?\(Player ID:\s*(player_\d+)\)\s*$", re.M)

# All non-claim templates are shared by all roles, not keyed to the hidden label.
_SHARED_SPEECH = (
    "我先把{target}列入观察名单，目前证据仍不够充分。",
    "相比口头表态，我更关心{target}在发言与投票之间是否一致。",
    "请{target}解释一下之前的投票依据，暂时不下结论。",
    "仅凭一段发言很难判断身份，我希望听听{target}的后续解释。",
    "当前记录中{target}仍有疑点，最好对照前后两轮的信息。",
    "大家先看公开事件的顺序，再核对{target}的理由。",
)


def parse_choices(message: str) -> list[Choice]:
    """Extract only legal targets explicitly advertised by the engine.

    Invalid target formatting fails closed: a numbered decision is never guessed.
    Speech / team discussion prompts without a choices block return [].
    """
    if _CHOICES_MARKER not in message:
        return []
    block = message.rsplit(_CHOICES_MARKER, 1)[1].split("\n\n", 1)[0]
    matches = _CHOICES_RE.findall(block)
    if not matches:
        raise ValueError("Engine advertised target choices but none could be parsed")
    nums = [number for number, _ in matches]
    if len(nums) != len(set(nums)):
        raise ValueError("Duplicate numbered choices")
    return matches


def _public_events(view: Mapping[str, Any]) -> list[dict[str, Any]]:
    """Allow only PUBLIC event types for common belief scoring.

    Deliberately ignore any other event in the view, even if unexpected.
    A private seer check is consumed separately, and only by that seer.
    """
    allowed = {"player_speech", "vote_cast", "vote_result", "player_eliminated", "player_died"}
    return [e for e in view.get("events", ()) if e.get("type") in allowed]


def _living(view: Mapping[str, Any]) -> list[str]:
    me = view["self"]["id"]
    return [p["id"] for p in view["players"] if p.get("alive") and p["id"] != me]


def _display_name(view: Mapping[str, Any], player_id: str) -> str:
    return next((p.get("name") or player_id for p in view["players"] if p["id"] == player_id), player_id)


def _mentioned(speech: str, player: Mapping[str, Any]) -> bool:
    ident = player["id"]
    name = player.get("name") or ""
    # ID and a nonempty explicit display name are both allowed public references.
    return ident in speech or (bool(name) and name in speech)


def public_suspicion(view: Mapping[str, Any]) -> dict[str, float]:
    """A deterministic, transparent heuristic using PUBLIC evidence only.

    Votes for revealed villagers and vote-target switches add weak suspicion;
    public accusations add weak suspicion, public defenses reduce it. These are
    synthetic heuristics, not a validated Bayesian role posterior.
    """
    roster = {p["id"]: p for p in view.get("players", ())}
    scores = {pid: 0.0 for pid, p in roster.items() if p.get("alive")}
    past_votes: dict[str, list[tuple[int, str]]] = defaultdict(list)
    revealed = {pid: p.get("revealed_role") for pid, p in roster.items()}

    for ev in _public_events(view):
        kind, data = ev["type"], ev.get("data", {})
        if kind == "vote_cast":
            voter, target = data.get("voter_id"), data.get("target_id")
            if voter and target:
                past_votes[voter].append((int(ev.get("round", 0)), target))
                if target in scores:
                    scores[target] += 0.10  # being voted is weak, fallible evidence
                if revealed.get(target) == "Villager" and voter in scores:
                    scores[voter] += 0.15
        elif kind == "player_speech":
            speaker = data.get("player_id")
            speech = str(data.get("speech", ""))
            suspect = any(term in speech for term in ("怀疑", "疑点", "不一致", "可疑", "质疑", "查杀"))
            defense = any(term in speech for term in ("可信", "信任", "支持", "清白"))
            for pid, player in roster.items():
                if pid == speaker or pid not in scores or not _mentioned(speech, player):
                    continue
                if suspect:
                    scores[pid] += 0.30
                if defense and not suspect:
                    scores[pid] -= 0.15

    for voter, history in past_votes.items():
        # Target-switch evidence comes from distinct *past* voting rounds.
        by_round = {}
        for round_no, target in history:
            by_round[round_no] = target
        targets = [by_round[r] for r in sorted(by_round)]
        switches = sum(a != b for a, b in zip(targets, targets[1:]))
        if voter in scores:
            scores[voter] += 0.20 * switches
    return scores


def _private_checks(view: Mapping[str, Any]) -> dict[str, str]:
    if view["self"].get("role") != "Seer":
        return {}
    checks = {}
    for ev in view.get("events", ()):
        if ev.get("type") == "seer_checked":
            data = ev.get("data", {})
            if data.get("target_id") and data.get("result") in {"werewolf", "villager"}:
                checks[data["target_id"]] = data["result"]
    return checks


def _claimants(view: Mapping[str, Any]) -> set[str]:
    """Claimants are speakers using role/inspection words, never verified seers."""
    return {
        e.get("data", {}).get("player_id")
        for e in _public_events(view)
        if e["type"] == "player_speech"
        and any(w in str(e.get("data", {}).get("speech", "")) for w in ("预言", "核验", "查验", "身份反馈"))
    } - {None}


def _ranked_pick(candidates: Sequence[Choice], scores: Mapping[str, float], rng: random.Random,
                 *, high: bool = True) -> str:
    if not candidates:
        raise ValueError("No legal targets available")
    values = [scores.get(pid, 0.0) for _, pid in candidates]
    preferred = max(values) if high else min(values)
    ties = [number for (number, pid), value in zip(candidates, values) if value == preferred]
    return rng.choice(ties)


def decide(view: Mapping[str, Any], choices: Sequence[Choice], rng: random.Random,
           memory: dict[str, Any]) -> str:
    """Return a legal action number or a short speech for this player's turn."""
    role = view["self"]["role"]
    me = view["self"]["id"]
    if role not in {"Werewolf", "Seer", "Guard", "Villager"}:
        raise ValueError(f"Unsupported role: {role}")
    mates = set(view.get("wolf_teammates", ())) if role == "Werewolf" else set()
    if role != "Werewolf" and view.get("wolf_teammates"):
        raise ValueError("Unexpected private wolf-team information in non-wolf view")

    scores = public_suspicion(view)
    checks = _private_checks(view)
    living = _living(view)

    if not choices:
        if view.get("phase") == "night":
            return "夜里先观察公开指控与投票趋势，不要将未经证实的声明视为事实。"
        # A real seer occasionally publicly shares an actual private check.
        if role == "Seer" and checks and (rng.random() < 0.65 or int(view.get("round", 1)) >= 3):
            checked = sorted(checks, key=lambda p: (checks[p] != "werewolf", p))[0]
            kind = "狼" if checks[checked] == "werewolf" else "非狼"
            return f"我愿意公开一条自己的查验反馈：{_display_name(view, checked)}显示为{kind}，请结合其他证据核对。"
        # A wolf may make a FALSE public claim; this is game behavior, not leakage.
        if role == "Werewolf" and living and rng.random() < 0.12:
            non_mates = [p for p in living if p not in mates]
            if non_mates:
                alleged = rng.choice(non_mates)
                return f"我有一条尚待验证的身份线索，倾向怀疑{_display_name(view, alleged)}。"
        candidates = [pid for pid in living if pid not in mates] or living
        if not candidates:
            return "目前没有足够的公开信息，需要再核对发言与票型。"
        best = max(scores.get(p, 0.0) for p in candidates)
        ties = [p for p in candidates if scores.get(p, 0.0) == best]
        target = _display_name(view, rng.choice(ties))
        return rng.choice(_SHARED_SPEECH).format(target=target)

    legal = list(choices)
    allowed = {pid for _, pid in legal}
    if view.get("phase") == "night":
        if role == "Seer":
            unexplored = [(num, pid) for num, pid in legal if pid not in checks]
            return _ranked_pick(unexplored or legal, scores, rng)
        if role == "Guard":
            claimants = _claimants(view) & allowed
            supported = [(num, pid) for num, pid in legal if pid in claimants]
            if supported and rng.random() < .7:
                return _ranked_pick(supported, scores, rng, high=False)
            personal = [(num, pid) for num, pid in legal if pid == me]
            if personal and rng.random() < .3:
                return personal[0][0]
            return _ranked_pick(legal, scores, rng, high=False)
        if role == "Werewolf":
            targets = [(num, pid) for num, pid in legal if pid not in mates]
            if not targets:
                raise ValueError("No legal non-teammate target for wolf")
            claimants = _claimants(view)
            priority = [(num, pid) for num, pid in targets if pid in claimants]
            return _ranked_pick(priority or targets, scores, rng, high=False)
        # In Lab, villagers normally do not get a targeted night action.
        return _ranked_pick(legal, scores, rng)

    # Day voting. Prefer private seer knowledge; everyone else uses public data.
    if role == "Seer":
        known_wolves = [(num, pid) for num, pid in legal if checks.get(pid) == "werewolf"]
        if known_wolves:
            return _ranked_pick(known_wolves, scores, rng)
        legal = [(num, pid) for num, pid in legal if checks.get(pid) != "villager"] or legal
    if role == "Werewolf":
        legal = [(num, pid) for num, pid in legal if pid not in mates] or legal
    return _ranked_pick(legal, scores, rng)
