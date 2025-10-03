from typing import List, Optional

from django.test import TestCase
from pydantic import BaseModel, Field

from llmtools.llm_calling import chatter, structured_chat
from mindframe.models import LLM


class LLMToolsTestCase(TestCase):
    def setUp(self):
        self.mini = LLM.objects.create(model_name="gpt-4o-mini")
        raise Exception("Need to add credentials to LLM")
        self.credentials = None

    def test_chatter_joke(self):
        jk = chatter(
            "Think about mammals very briefly, in one or 2 lines: [[THOUGHTS]] Now tell me a joke. [[speak:JOKE]]",
            self.mini,
            self.credentials,
        )

        self.assertIsInstance(jk["JOKE"], str)
        self.assertEqual(len(jk.keys()), 3)
        self.assertEqual(set(jk.keys()), {"THOUGHTS", "JOKE"})

    def test_chatter_topic_choice(self):
        rr = chatter(
            "Which is more interesting for nerds: art, science or magic? [[pick:topic|art,science,magic]]",
            self.mini,
            self.credentials,
        )

        self.assertIn(rr["topic"], ["art", "science", "magic"])
        self.assertEqual(rr["topic"], "science")

    def test_structured_chat(self):
        class UserInfo(BaseModel):
            name: str
            age: Optional[int] = Field(
                description="The age in years of the user.", default=None
            )

        class UserList(BaseModel):
            peeps: List[UserInfo]

        newusers, completions = structured_chat(
            "Create a list of 3 imaginary users. Use consistent field names for each item. Use the tools [[users]]",
            self.mini,
            self.credentials,
            return_type=UserList,
        )

        self.assertEqual(len(newusers.peeps), 3)
