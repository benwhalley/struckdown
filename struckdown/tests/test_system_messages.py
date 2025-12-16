"""
Tests for system message semantics using new XML-style syntax.
Tests the two-list model (globals + locals) for system prompts.
"""

import unittest

from struckdown.parsing import parse_syntax


class SystemMessageTestCase(unittest.TestCase):
    """Test <system> tags"""

    def test_system_message_basic(self):
        """Test basic <system> tag"""
        template = """<system>You are a helpful assistant.</system>

Tell a joke [[joke]]"""

        sections = parse_syntax(template)
        self.assertEqual(len(sections), 1)

        part = sections[0]["joke"]
        self.assertEqual(part.system_message, "You are a helpful assistant.")
        self.assertEqual(part.text, "Tell a joke")

    def test_system_message_append(self):
        """Test multiple <system> tags append to globals by default"""
        template = """<system>You are a helpful assistant.</system>

<system>Talk like a pirate.</system>

Tell a joke [[joke]]"""

        sections = parse_syntax(template)
        part = sections[0]["joke"]

        expected = "You are a helpful assistant.\n\nTalk like a pirate."
        self.assertEqual(part.system_message, expected)

    def test_system_message_replace(self):
        """Test <system replace> replaces previous system message"""
        template = """<system>First message.</system>

<system replace>Second message.</system>

Tell a joke [[joke]]"""

        sections = parse_syntax(template)
        part = sections[0]["joke"]

        self.assertEqual(part.system_message, "Second message.")

    def test_system_persists_across_checkpoints(self):
        """Test global system message persists across <checkpoint>"""
        template = """<system>You are helpful.</system>

First question? [[q1]]

<checkpoint>

Second question? [[q2]]"""

        sections = parse_syntax(template)
        self.assertEqual(len(sections), 2)

        self.assertEqual(sections[0]["q1"].system_message, "You are helpful.")
        self.assertEqual(sections[1]["q2"].system_message, "You are helpful.")


class LocalSystemTestCase(unittest.TestCase):
    """Test <system local> tags"""

    def test_local_system_basic(self):
        """Test <system local> adds to locals list"""
        template = """<system>Global prompt.</system>

<system local>Local addition.</system>

Question? [[q]]"""

        sections = parse_syntax(template)
        part = sections[0]["q"]

        expected = "Global prompt.\n\nLocal addition."
        self.assertEqual(part.system_message, expected)

    def test_local_system_cleared_after_checkpoint(self):
        """Test locals are cleared after <checkpoint>"""
        template = """<system>Global.</system>

<system local>Local1.</system>
Q1? [[q1]]

<checkpoint>

Q2? [[q2]]"""

        sections = parse_syntax(template)

        # Section 1: has both global and local
        self.assertEqual(sections[0]["q1"].system_message, "Global.\n\nLocal1.")
        # Section 2: only global (local was cleared)
        self.assertEqual(sections[1]["q2"].system_message, "Global.")

    def test_local_system_replace(self):
        """Test <system local replace> replaces local list"""
        template = """<system>Global.</system>

<system local>Local1.</system>
<system local>Local2.</system>
<system local replace>LocalReplaced.</system>

Q? [[q]]"""

        sections = parse_syntax(template)
        part = sections[0]["q"]

        expected = "Global.\n\nLocalReplaced."
        self.assertEqual(part.system_message, expected)


class CheckpointTestCase(unittest.TestCase):
    """Test <checkpoint> and <obliviate> tags"""

    def test_checkpoint_creates_segment(self):
        """Test <checkpoint> creates a new segment"""
        template = """First question? [[q1]]

<checkpoint>

Second question? [[q2]]"""

        sections = parse_syntax(template)
        self.assertEqual(len(sections), 2)

        self.assertEqual(sections[0]["q1"].text, "First question?")
        self.assertEqual(sections[1]["q2"].text, "Second question?")

    def test_obliviate_synonym(self):
        """Test <obliviate> is synonym for <checkpoint>"""
        template = """First question? [[q1]]

<obliviate>

Second question? [[q2]]"""

        sections = parse_syntax(template)
        self.assertEqual(len(sections), 2)

    def test_named_checkpoint(self):
        """Test named checkpoint with closed tag"""
        template = """<checkpoint>Introduction</checkpoint>
First? [[q1]]

<checkpoint>Analysis Phase</checkpoint>
Second? [[q2]]"""

        sections = parse_syntax(template)
        self.assertEqual(len(sections), 2)

        self.assertEqual(sections[0].segment_name, "Introduction")
        self.assertEqual(sections[1].segment_name, "Analysis Phase")

    def test_auto_named_checkpoint(self):
        """Test auto-named checkpoint (no closing tag)"""
        template = """First? [[q1]]

<checkpoint>

Second? [[q2]]

<checkpoint>

Third? [[q3]]"""

        sections = parse_syntax(template)
        self.assertEqual(len(sections), 3)

        self.assertIsNone(sections[0].segment_name)
        self.assertEqual(sections[1].segment_name, "Checkpoint 1")
        self.assertEqual(sections[2].segment_name, "Checkpoint 2")

    def test_multiple_completions_in_segment(self):
        """Test multiple completions in same segment"""
        template = """<system>You are helpful.</system>

First question? [[q1]]

Follow up? [[q2]]

Final question? [[q3]]"""

        sections = parse_syntax(template)
        self.assertEqual(len(sections), 1)

        # All three completions in same segment
        self.assertIn("q1", sections[0])
        self.assertIn("q2", sections[0])
        self.assertIn("q3", sections[0])

        # All share same system message
        for key in ["q1", "q2", "q3"]:
            self.assertEqual(sections[0][key].system_message, "You are helpful.")


class IntegrationTestCase(unittest.TestCase):
    """Integration tests for combined features"""

    def test_full_example(self):
        """Test a comprehensive example with all features"""
        template = """<system>You are a kind and helpful therapist.</system>

What is the meaning of life? [[meaning_of_life]]

Are you sure? [[are_you_sure]]

<checkpoint>Analysis</checkpoint>

What is the pirate code? [[pirate_code]]

<checkpoint>Pirate Mode</checkpoint>

<system>Talk like a pirate.</system>

What should we do with a drunken sailor? [[drunken_sailor]]

<checkpoint>Final</checkpoint>

Make up a cool name? [[name]]"""

        sections = parse_syntax(template)
        self.assertEqual(len(sections), 4)

        # Section 0: meaning_of_life, are_you_sure
        self.assertIn("meaning_of_life", sections[0])
        self.assertIn("are_you_sure", sections[0])
        self.assertEqual(
            sections[0]["meaning_of_life"].system_message,
            "You are a kind and helpful therapist.",
        )

        # Section 1: pirate_code (named "Analysis")
        self.assertIn("pirate_code", sections[1])
        self.assertEqual(sections[1].segment_name, "Analysis")
        self.assertEqual(
            sections[1]["pirate_code"].system_message,
            "You are a kind and helpful therapist.",
        )

        # Section 2: drunken_sailor (with appended system)
        self.assertIn("drunken_sailor", sections[2])
        self.assertEqual(sections[2].segment_name, "Pirate Mode")
        expected_system = "You are a kind and helpful therapist.\n\nTalk like a pirate."
        self.assertEqual(sections[2]["drunken_sailor"].system_message, expected_system)

        # Section 3: name (system persists from previous)
        self.assertIn("name", sections[3])
        self.assertEqual(sections[3].segment_name, "Final")
        self.assertEqual(sections[3]["name"].system_message, expected_system)


if __name__ == "__main__":
    unittest.main()
