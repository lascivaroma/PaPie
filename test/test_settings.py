import os
import unittest
from unittest import mock

from pie import settings as S


class TestSettings(unittest.TestCase):
    def test_defaults_load(self):
        s = S.load_default_settings()
        self.assertTrue(len(s.tasks) >= 1)
        self.assertEqual(s.tasks[0]["settings"]["target"], s.tasks[0]["name"])

    def test_attribute_access(self):
        s = S.Settings({"a": 1})
        s.b = 2
        self.assertEqual((s.a, s["b"], s.missing), (1, 2, None))
        del s.a
        self.assertNotIn("a", s)

    def test_parse_key(self):
        self.assertEqual(S.parse_key(["a", "b"], "1", {"a": {"b": 2}}),
                         {"a": {"b": 1}})
        with self.assertRaises(ValueError):
            S.parse_key(["zzz"], "1", {"a": 1})

    def test_env_settings(self):
        with mock.patch.dict(os.environ, {"PIE_BATCH_SIZE": "7"}):
            self.assertEqual(S.parse_env_settings({"batch_size": 1}),
                             {"batch_size": 7})

    def test_check_settings(self):
        s = S.load_default_settings()
        s.tasks = [{"name": "a", "decoder": "linear"}]
        self.assertTrue(S.check_settings(s).tasks[0]["target"])
        s.tasks = [{"name": "a", "decoder": "linear", "target": True},
                   {"name": "b", "decoder": "linear", "target": True}]
        with self.assertRaises(ValueError):
            S.check_settings(s)
        s.tasks = [{"name": "a", "decoder": "linear"},
                   {"name": "b", "decoder": "linear"}]
        with self.assertRaises(ValueError):  # no target
            S.check_settings(s)
        s.tasks = [{"name": "a", "decoder": "linear", "conditions": ["nope"]}]
        with self.assertRaises(ValueError):
            S.check_settings(s)
