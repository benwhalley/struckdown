"""
Tests for new system message and header semantics.
"""

import unittest
from struckdown.parsing import parse_syntax


class SystemMessageTestCase(unittest.TestCase):
    """Test ¡SYSTEM and ¡IMPORTANT tags"""

    def test_system_message_basic(self):
        """Test basic ¡SYSTEM tag"""
        template = """¡SYSTEM
You are a helpful assistant.
/END

Tell a joke [[joke]]"""

        sections = parse_syntax(template)
        self.assertEqual(len(sections), 1)

        part = sections[0]["joke"]
        self.assertEqual(part.system_message, "You are a helpful assistant.")
        self.assertEqual(part.header_content, "")
        self.assertEqual(part.text, "Tell a joke")

    def test_system_message_append(self):
        """Test ¡SYSTEM+ appends to system message"""
        template = """¡SYSTEM
You are a helpful assistant.
/END

¡SYSTEM+
Talk like a pirate.
/END

Tell a joke [[joke]]"""

        sections = parse_syntax(template)
        part = sections[0]["joke"]

        expected = "You are a helpful assistant.\n\nTalk like a pirate."
        self.assertEqual(part.system_message, expected)

    def test_system_message_replace(self):
        """Test ¡SYSTEM replaces previous system message"""
        template = """¡SYSTEM
First message.
/END

¡SYSTEM
Second message.
/END

Tell a joke [[joke]]"""

        sections = parse_syntax(template)
        part = sections[0]["joke"]

        self.assertEqual(part.system_message, "Second message.")

    def test_important_synonym(self):
        """Test ¡IMPORTANT is synonym for ¡SYSTEM"""
        template = """¡IMPORTANT
You are helpful.
/END

¡IMPORTANT+
Be concise.
/END

Answer [[answer]]"""

        sections = parse_syntax(template)
        part = sections[0]["answer"]

        expected = "You are helpful.\n\nBe concise."
        self.assertEqual(part.system_message, expected)

    def test_system_persists_across_segments(self):
        """Test system message persists across ¡OBLIVIATE"""
        template = """¡SYSTEM
You are helpful.
/END

First question? [[q1]]

¡OBLIVIATE

Second question? [[q2]]"""

        sections = parse_syntax(template)
        self.assertEqual(len(sections), 2)

        self.assertEqual(sections[0]["q1"].system_message, "You are helpful.")
        self.assertEqual(sections[1]["q2"].system_message, "You are helpful.")


class HeaderTestCase(unittest.TestCase):
    """Test ¡HEADER tags"""

    def test_header_basic(self):
        """Test basic ¡HEADER tag"""
        template = """¡HEADER
Answer all questions carefully.
/END

What is 2+2? [[answer]]"""

        sections = parse_syntax(template)
        part = sections[0]["answer"]

        self.assertEqual(part.header_content, "Answer all questions carefully.")
        self.assertEqual(part.text, "What is 2+2?")

    def test_header_append(self):
        """Test ¡HEADER+ appends to header"""
        template = """¡HEADER
First instruction.
/END

¡HEADER+
Second instruction.
/END

Question? [[answer]]"""

        sections = parse_syntax(template)
        part = sections[0]["answer"]

        expected = "First instruction.\n\nSecond instruction."
        self.assertEqual(part.header_content, expected)

    def test_header_replace(self):
        """Test ¡HEADER replaces previous header"""
        template = """¡HEADER
First header.
/END

¡HEADER
Second header.
/END

Question? [[answer]]"""

        sections = parse_syntax(template)
        part = sections[0]["answer"]

        self.assertEqual(part.header_content, "Second header.")

    def test_empty_header_wipes(self):
        """Test empty ¡HEADER wipes current header"""
        template = """¡HEADER
Original header.
/END

¡HEADER

/END

Question? [[answer]]"""

        sections = parse_syntax(template)
        part = sections[0]["answer"]

        # Empty header wipes it
        self.assertEqual(part.header_content, "")

    def test_header_persists_across_segments(self):
        """Test header persists across ¡OBLIVIATE"""
        template = """¡HEADER
Important instructions.
/END

First question? [[q1]]

¡OBLIVIATE

Second question? [[q2]]"""

        sections = parse_syntax(template)
        self.assertEqual(len(sections), 2)

        self.assertEqual(
            sections[0]["q1"].header_content, "Important instructions."
        )
        self.assertEqual(
            sections[1]["q2"].header_content, "Important instructions."
        )


class SegmentTestCase(unittest.TestCase):
    """Test ¡OBLIVIATE and ¡SEGMENT tags"""

    def test_obliviate_creates_segment(self):
        """Test ¡OBLIVIATE creates a new segment"""
        template = """First question? [[q1]]

¡OBLIVIATE

Second question? [[q2]]"""

        sections = parse_syntax(template)
        self.assertEqual(len(sections), 2)

        self.assertEqual(sections[0]["q1"].text, "First question?")
        self.assertEqual(sections[1]["q2"].text, "Second question?")

    def test_segment_synonym(self):
        """Test ¡SEGMENT is synonym for ¡OBLIVIATE"""
        template = """First question? [[q1]]

¡SEGMENT

Second question? [[q2]]"""

        sections = parse_syntax(template)
        self.assertEqual(len(sections), 2)

    def test_multiple_completions_in_segment(self):
        """Test multiple completions in same segment"""
        template = """¡SYSTEM
You are helpful.
/END

¡HEADER
Context here.
/END

First question? [[q1]]

Follow up? [[q2]]

Final question? [[q3]]"""

        sections = parse_syntax(template)
        self.assertEqual(len(sections), 1)

        # All three completions in same segment
        self.assertIn("q1", sections[0])
        self.assertIn("q2", sections[0])
        self.assertIn("q3", sections[0])

        # All share same system and header
        for key in ["q1", "q2", "q3"]:
            self.assertEqual(sections[0][key].system_message, "You are helpful.")
            self.assertEqual(sections[0][key].header_content, "Context here.")


class IntegrationTestCase(unittest.TestCase):
    """Integration tests for combined features"""

    def test_full_example(self):
        """Test the full example from the specification"""
        template = """¡SYSTEM
You are a kind and helpful therapist.
/END

¡HEADER
Answer all questions carefully
/END

What is the meaning of life? [[meaning_of_life]]

Are you sure? [[are_you_sure]]

¡OBLIVIATE

What is the pirate code? [[pirate_code]]

¡OBLIVIATE

¡SYSTEM+
Talk like a pirate.
/END

¡HEADER+
The pirate code is: {{pirate_code}}. Bear this in mind.
/END

What should we do with a drunken sailor who steals our rum? [[drunken_sailor]]

¡OBLIVIATE

Make up a cool name? [[name]]"""

        sections = parse_syntax(template)
        self.assertEqual(len(sections), 4)

        # Segment 0: meaning_of_life, are_you_sure
        self.assertIn("meaning_of_life", sections[0])
        self.assertIn("are_you_sure", sections[0])
        self.assertEqual(
            sections[0]["meaning_of_life"].system_message,
            "You are a kind and helpful therapist.",
        )
        self.assertEqual(
            sections[0]["meaning_of_life"].header_content,
            "Answer all questions carefully",
        )

        # Segment 1: pirate_code
        self.assertIn("pirate_code", sections[1])
        self.assertEqual(
            sections[1]["pirate_code"].system_message,
            "You are a kind and helpful therapist.",
        )

        # Segment 2: drunken_sailor (with appended system and header)
        self.assertIn("drunken_sailor", sections[2])
        expected_system = (
            "You are a kind and helpful therapist.\n\nTalk like a pirate."
        )
        self.assertEqual(sections[2]["drunken_sailor"].system_message, expected_system)

        expected_header = (
            "Answer all questions carefully\n\n"
            "The pirate code is: {{pirate_code}}. Bear this in mind."
        )
        self.assertEqual(sections[2]["drunken_sailor"].header_content, expected_header)

        # Segment 3: name (system and header persist from previous segment)
        self.assertIn("name", sections[3])
        self.assertEqual(sections[3]["name"].system_message, expected_system)
        self.assertEqual(sections[3]["name"].header_content, expected_header)


if __name__ == "__main__":
    unittest.main()
