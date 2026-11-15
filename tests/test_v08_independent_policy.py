"""Run as pure Python: python -m unittest discover -s tests -p test_v08*.py"""
import random
import sys
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))
from v08_independent_policy import decide, parse_choices, public_suspicion


def view(role="Villager", events=None):
    return {
        "phase": "day_voting", "round": 2,
        "self": {"id": "player_1", "role": role},
        "wolf_teammates": ["player_2"] if role == "Werewolf" else [],
        "players": [
            {"id": f"player_{i}", "name": f"{i}号", "alive": True}
            for i in range(1, 9)
        ],
        "events": list(events or []),
    }


class IndependentPolicyTests(unittest.TestCase):
    def test_parser_accepts_only_advertised_targets(self):
        text = "Select an option.\nAvailable targets:\n1. Alice (Player ID: player_2)\n2. Bob (Player ID: player_3)\n\nDo not include any other text in your response."
        self.assertEqual(parse_choices(text), [("1", "player_2"), ("2", "player_3")])
        self.assertEqual(parse_choices("You may speak now"), [])
        with self.assertRaises(ValueError):
            parse_choices("\nAvailable targets:\ninvalid!\n\nDo not include")

    def test_no_private_seer_knowledge_for_villager(self):
        opts = [("1", "player_2"), ("2", "player_3")]
        a = view()
        b = view(events=[{"type": "seer_checked", "data": {"target_id": "player_3", "result": "werewolf"}}])
        self.assertEqual(public_suspicion(a), public_suspicion(b))
        self.assertEqual(decide(a, opts, random.Random(11), {}), decide(b, opts, random.Random(11), {}))

    def test_seer_uses_private_check_for_vote(self):
        obs = view("Seer", events=[{"type": "seer_checked", "data": {"target_id": "player_3", "result": "werewolf"}}])
        self.assertEqual(decide(obs, [("1", "player_2"), ("2", "player_3")], random.Random(2), {}), "2")

    def test_wolf_avoids_teammate(self):
        obs = view("Werewolf")
        choice = decide(obs, [("1", "player_2"), ("2", "player_3")], random.Random(2), {})
        self.assertEqual(choice, "2")

    def test_choices_always_legal(self):
        opts = [("1", "player_2"), ("2", "player_3"), ("3", "player_4")]
        for role in ("Werewolf", "Seer", "Guard", "Villager"):
            for phase in ("night", "day_voting"):
                obs = view(role)
                obs["phase"] = phase
                self.assertIn(decide(obs, opts, random.Random(55), {}), {n for n, _ in opts})

    def test_unknown_role_denied(self):
        with self.assertRaises(ValueError):
            decide(view("Unknown"), [("1", "player_2")], random.Random(0), {})

    def test_nonwolf_private_team_leak_denied(self):
        obs = view("Villager")
        obs["wolf_teammates"] = ["player_2"]
        with self.assertRaises(ValueError):
            decide(obs, [("1", "player_3")], random.Random(0), {})

    def test_speech_role_neutral_when_no_claim(self):
        obs = view("Villager")
        obs["phase"] = "day_discussion"
        text = decide(obs, [], random.Random(1), {})
        self.assertTrue(text)
        self.assertNotIn("我自称预言家，查验", text)


if __name__ == "__main__":
    unittest.main()
