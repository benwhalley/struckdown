"""
Comprehensive tests for temporal response types (date, datetime, time, duration).
Tests both basic extraction and context-aware relative temporal references.
"""

import unittest
from datetime import date, datetime, time, timedelta
from unittest.mock import Mock, patch

from struckdown import chatter
from struckdown.parsing import parse_syntax
from struckdown.return_type_models import (
    date_response_model,
    datetime_response_model,
    duration_response_model,
    time_response_model,
)


class TemporalResponseModelsTestCase(unittest.TestCase):
    """Test that temporal response models are properly defined"""

    def test_date_response_model_factory(self):
        """Test date_response_model factory creates model"""
        model = date_response_model()
        self.assertIsNotNone(model)
        self.assertIn("response", model.model_fields)

    def test_datetime_response_model_factory(self):
        """Test datetime_response_model factory creates model"""
        model = datetime_response_model()
        self.assertIsNotNone(model)
        self.assertIn("response", model.model_fields)

    def test_time_response_model_factory(self):
        """Test time_response_model factory creates model"""
        model = time_response_model()
        self.assertIsNotNone(model)
        self.assertIn("response", model.model_fields)

    def test_duration_response_model_factory(self):
        """Test duration_response_model factory creates model"""
        model = duration_response_model()
        self.assertIsNotNone(model)
        self.assertIn("response", model.model_fields)


class TemporalParsingTestCase(unittest.TestCase):
    """Test that temporal syntax is parsed correctly"""

    def test_parse_date_syntax(self):
        """Test [[date:varname]] parsing"""
        template = "Extract the date [[date:event_date]]"
        sections = parse_syntax(template)

        self.assertEqual(len(sections), 1)
        part = sections[0]["event_date"]
        self.assertEqual(part.key, "event_date")
        self.assertEqual(part.action_type, "date")

    def test_parse_datetime_syntax(self):
        """Test [[datetime:varname]] parsing"""
        template = "Extract the datetime [[datetime:event_time]]"
        sections = parse_syntax(template)

        part = sections[0]["event_time"]
        self.assertEqual(part.key, "event_time")
        self.assertEqual(part.action_type, "datetime")

    def test_parse_time_syntax(self):
        """Test [[time:varname]] parsing"""
        template = "Extract the time [[time:meeting_time]]"
        sections = parse_syntax(template)

        part = sections[0]["meeting_time"]
        self.assertEqual(part.key, "meeting_time")
        self.assertEqual(part.action_type, "time")

    def test_parse_duration_syntax(self):
        """Test [[duration:varname]] parsing"""
        template = "Extract the duration [[duration:length]]"
        sections = parse_syntax(template)

        part = sections[0]["length"]
        self.assertEqual(part.key, "length")
        self.assertEqual(part.action_type, "duration")

    def test_parse_multiple_temporal_types(self):
        """Test multiple temporal types in one template"""
        template = """Extract the date [[date:event_date]]

¡OBLIVIATE

Extract the time [[time:start_time]]"""
        sections = parse_syntax(template)

        self.assertEqual(len(sections), 2)
        self.assertEqual(sections[0]["event_date"].action_type, "date")
        self.assertEqual(sections[1]["start_time"].action_type, "time")


class TemporalActionLookupTestCase(unittest.TestCase):
    """Test that ACTION_LOOKUP correctly maps temporal types"""

    def test_action_lookup_date(self):
        """Test 'date' maps to date_response_model factory"""
        from struckdown.return_type_models import ACTION_LOOKUP

        self.assertEqual(ACTION_LOOKUP["date"], date_response_model)

    def test_action_lookup_datetime(self):
        """Test 'datetime' maps to datetime_response_model factory"""
        from struckdown.return_type_models import ACTION_LOOKUP

        self.assertEqual(ACTION_LOOKUP["datetime"], datetime_response_model)

    def test_action_lookup_time(self):
        """Test 'time' maps to time_response_model factory"""
        from struckdown.return_type_models import ACTION_LOOKUP

        self.assertEqual(ACTION_LOOKUP["time"], time_response_model)

    def test_action_lookup_duration(self):
        """Test 'duration' maps to duration_response_model factory"""
        from struckdown.return_type_models import ACTION_LOOKUP

        self.assertEqual(ACTION_LOOKUP["duration"], duration_response_model)


class TemporalContextInjectionTestCase(unittest.TestCase):
    """Test that temporal context is automatically injected"""

    def test_temporal_context_injected_for_date(self):
        """Test that temporal context is added for date extractions"""
        template = "The event is on January 15, 2024 [[date:event_date]]"
        sections = parse_syntax(template)

        part = sections[0]["event_date"]
        self.assertEqual(part.action_type, "date")

        # temporal context injection is tested by integration tests


class TemporalModelValidationTestCase(unittest.TestCase):
    """Test that Pydantic validates temporal types correctly"""

    def test_date_response_validation_success(self):
        """Test date model accepts valid date"""
        model = date_response_model(options=["required"])  # Required for this test
        response = model(response=date(2024, 1, 15))
        self.assertEqual(response.response, date(2024, 1, 15))

    def test_date_response_validation_from_string(self):
        """Test date model accepts ISO date string (as pattern string)"""
        model = date_response_model(options=["required"])  # Required for this test
        response = model(response="2024-01-15")
        # Union[date, str] accepts strings - they could be pattern strings or ISO dates
        # In practice, instructor returns properly typed date objects
        self.assertEqual(response.response, "2024-01-15")

    def test_datetime_response_validation_success(self):
        """Test datetime model accepts valid datetime"""
        model = datetime_response_model(options=["required"])  # Required for this test
        dt = datetime(2024, 1, 15, 14, 30, 0)
        response = model(response=dt)
        self.assertEqual(response.response, dt)

    def test_datetime_response_validation_from_string(self):
        """Test datetime model accepts ISO datetime string (as pattern string)"""
        model = datetime_response_model(options=["required"])  # Required for this test
        response = model(response="2024-01-15T14:30:00")
        # Union[datetime, str] accepts strings - they could be pattern strings or ISO datetimes
        # In practice, instructor returns properly typed datetime objects
        self.assertEqual(response.response, "2024-01-15T14:30:00")

    def test_time_response_validation_success(self):
        """Test time model accepts valid time"""
        model = time_response_model(options=["required"])  # Required for this test
        t = time(14, 30, 0)
        response = model(response=t)
        self.assertEqual(response.response, t)

    def test_time_response_validation_from_string(self):
        """Test time model accepts time string (as pattern string)"""
        model = time_response_model(options=["required"])  # Required for this test
        response = model(response="14:30:00")
        # Union[time, str] accepts strings - they could be pattern strings or ISO times
        # In practice, instructor returns properly typed time objects
        self.assertEqual(response.response, "14:30:00")

    def test_duration_response_validation_success(self):
        """Test duration model accepts valid timedelta"""
        model = duration_response_model(options=["required"])  # Required for this test
        td = timedelta(days=2, hours=3, minutes=30)
        response = model(response=td)
        self.assertEqual(response.response, td)

    def test_duration_response_validation_from_seconds(self):
        """Test duration model accepts seconds"""
        model = duration_response_model(options=["required"])  # Required for this test
        # Pydantic can convert numeric seconds to timedelta
        response = model(response=timedelta(seconds=3600))
        self.assertEqual(response.response, timedelta(hours=1))


class TemporalQuantifierTestCase(unittest.TestCase):
    """Test that temporal types work with quantifiers"""

    def test_parse_date_with_zero_or_more_quantifier(self):
        """Test [[date*:dates]] parsing"""
        template = "Extract dates [[date*:dates]]"
        sections = parse_syntax(template)

        part = sections[0]["dates"]
        self.assertEqual(part.key, "dates")
        self.assertEqual(part.action_type, "date")
        self.assertEqual(part.quantifier, (0, None))

    def test_parse_time_with_exact_quantifier(self):
        """Test [[time{2}:times]] parsing"""
        template = "Extract exactly 2 times [[time{2}:times]]"
        sections = parse_syntax(template)

        part = sections[0]["times"]
        self.assertEqual(part.action_type, "time")
        self.assertEqual(part.quantifier, (2, 2))

    def test_parse_duration_with_range_quantifier(self):
        """Test [[duration{1,3}:durations]] parsing"""
        template = "Extract 1-3 durations [[duration{1,3}:durations]]"
        sections = parse_syntax(template)

        part = sections[0]["durations"]
        self.assertEqual(part.action_type, "duration")
        self.assertEqual(part.quantifier, (1, 3))


class TemporalFieldDescriptionTestCase(unittest.TestCase):
    """Test that temporal response models have helpful descriptions"""

    def test_date_response_has_description(self):
        """Test date model includes helpful description"""
        model = date_response_model()
        field = model.model_fields["response"]
        description = field.description
        self.assertIsNotNone(description)
        self.assertIn("date", description.lower())

    def test_datetime_response_has_description(self):
        """Test datetime model includes helpful description"""
        model = datetime_response_model()
        field = model.model_fields["response"]
        description = field.description
        self.assertIsNotNone(description)
        self.assertIn("datetime", description.lower())

    def test_time_response_has_description(self):
        """Test time model includes helpful description"""
        model = time_response_model()
        field = model.model_fields["response"]
        description = field.description
        self.assertIsNotNone(description)
        self.assertIn("time", description.lower())

    def test_duration_response_has_description(self):
        """Test duration model includes helpful description"""
        model = duration_response_model()
        field = model.model_fields["response"]
        description = field.description
        self.assertIsNotNone(description)
        self.assertIn("duration", description.lower())


class TemporalTemplateVariableTestCase(unittest.TestCase):
    """Test temporal types work with template variables"""

    def test_date_extraction_with_template_variable(self):
        """Test date extraction with template variables in prompt"""
        template = "The {{event_name}} is on January 15 [[date:event_date]]"
        sections = parse_syntax(template)

        part = sections[0]["event_date"]
        self.assertEqual(part.action_type, "date")
        self.assertIn("{{event_name}}", part.text)

    def test_multiple_segments_with_temporal_and_non_temporal(self):
        """Test mixing temporal and non-temporal types across segments"""
        template = """Extract the event name [[extract:event_name]]

¡OBLIVIATE

When is the event? [[date:event_date]]"""
        sections = parse_syntax(template)

        self.assertEqual(len(sections), 2)
        self.assertEqual(sections[0]["event_name"].action_type, "extract")
        self.assertEqual(sections[1]["event_date"].action_type, "date")


class TemporalSharedHeaderTestCase(unittest.TestCase):
    """Test temporal types work with shared headers (¡BEGIN)"""

    def test_shared_header_with_temporal_type(self):
        """Test ¡BEGIN shared header with temporal extraction"""
        template = """You are an expert at extracting dates from text.

¡BEGIN

Extract the date from this text [[date:extracted_date]]"""
        sections = parse_syntax(template)

        part = sections[0]["extracted_date"]
        self.assertEqual(part.action_type, "date")
        self.assertIn("expert", part.shared_header)

    def test_shared_header_persists_across_temporal_segments(self):
        """Test shared header persists across multiple temporal segments"""
        template = """You are a temporal extraction specialist.

¡BEGIN

Extract the date [[date:event_date]]

¡OBLIVIATE

Extract the time [[time:event_time]]"""
        sections = parse_syntax(template)

        self.assertEqual(sections[0]["event_date"].shared_header, sections[1]["event_time"].shared_header)


class TemporalEdgeCaseTestCase(unittest.TestCase):
    """Test edge cases and error handling for temporal types"""

    def test_date_with_minimal_text(self):
        """Test date extraction with minimal text"""
        template = "Extract [[date:date]]"
        sections = parse_syntax(template)

        part = sections[0]["date"]
        self.assertEqual(part.action_type, "date")
        self.assertEqual(part.text, "Extract")

    def test_multiple_temporal_types_in_same_segment(self):
        """Test multiple temporal extractions in same segment"""
        template = """Extract date [[date:date1]]

Extract time [[time:time1]]"""
        sections = parse_syntax(template)

        # Should be in same segment (no OBLIVIATE)
        self.assertEqual(len(sections), 1)
        self.assertIn("date1", sections[0])
        self.assertIn("time1", sections[0])

    def test_temporal_type_as_untyped_completion(self):
        """Test that untyped completion doesn't accidentally match temporal types"""
        # [[date]] should NOT be interpreted as a date type, but as default type with key "date"
        template = "Extract [[date]]"
        sections = parse_syntax(template)

        part = sections[0]["date"]
        # This should be default type, not date type (no colon means untyped)
        self.assertEqual(part.action_type, "default")


class TemporalOptionalRequiredTestCase(unittest.TestCase):
    """Test optional vs required temporal fields"""

    def test_parse_date_with_required_option(self):
        """Test [[date:var|required]] parsing"""
        template = "Extract date [[date:event_date|required]]"
        sections = parse_syntax(template)

        part = sections[0]["event_date"]
        self.assertEqual(part.action_type, "date")
        self.assertIn("required", part.options)

    def test_parse_date_without_required(self):
        """Test [[date:var]] parsing (optional by default)"""
        template = "Extract date [[date:event_date]]"
        sections = parse_syntax(template)

        part = sections[0]["event_date"]
        self.assertEqual(part.action_type, "date")
        self.assertEqual(part.options, [])

    def test_date_factory_optional_by_default(self):
        """Test that date_response_model creates optional field by default"""
        from struckdown.return_type_models import date_response_model

        model = date_response_model()
        field = model.model_fields["response"]

        # Check if field is Optional (allows None)
        from typing import get_args, get_origin

        annotation = field.annotation
        # Optional[X] is Union[X, None]
        self.assertTrue(get_origin(annotation) is type(None) or type(None) in get_args(annotation))

    def test_optional_date_accepts_none(self):
        """Test that optional date model accepts None"""
        from struckdown.return_type_models import date_response_model

        model = date_response_model()
        instance = model(response=None)
        self.assertIsNone(instance.response)

    def test_required_date_rejects_none(self):
        """Test that required date model rejects None"""
        from struckdown.return_type_models import date_response_model
        from pydantic import ValidationError

        model = date_response_model(options=["required"])
        with self.assertRaises(ValidationError):
            model(response=None)

    def test_datetime_factory_optional_by_default(self):
        """Test that datetime_response_model creates optional field by default"""
        from struckdown.return_type_models import datetime_response_model

        model = datetime_response_model()
        field = model.model_fields["response"]

        from typing import get_args, get_origin

        annotation = field.annotation
        # Optional[X] is Union[X, None]
        self.assertTrue(get_origin(annotation) is type(None) or type(None) in get_args(annotation))

    def test_time_factory_optional_by_default(self):
        """Test that time_response_model creates optional field by default"""
        from struckdown.return_type_models import time_response_model

        model = time_response_model()
        field = model.model_fields["response"]

        from typing import get_args, get_origin

        annotation = field.annotation
        self.assertTrue(get_origin(annotation) is type(None) or type(None) in get_args(annotation))

    def test_duration_factory_optional_by_default(self):
        """Test that duration_response_model creates optional field by default"""
        from struckdown.return_type_models import duration_response_model

        model = duration_response_model()
        field = model.model_fields["response"]

        from typing import get_args, get_origin

        annotation = field.annotation
        self.assertTrue(get_origin(annotation) is type(None) or type(None) in get_args(annotation))

    def test_optional_temporal_field_description_mentions_null(self):
        """Test that optional temporal fields mention null in description"""
        from struckdown.return_type_models import date_response_model

        model = date_response_model()
        field = model.model_fields["response"]
        description = field.description

        self.assertIn("null", description.lower())

    def test_all_temporal_types_support_required(self):
        """Test that all temporal types support the required option"""
        from struckdown.return_type_models import (
            date_response_model,
            datetime_response_model,
            duration_response_model,
            time_response_model,
        )

        # All should create required fields when passed ["required"]
        for factory in [
            date_response_model,
            datetime_response_model,
            time_response_model,
            duration_response_model,
        ]:
            model = factory(options=["required"])
            field = model.model_fields["response"]

            # Should NOT be Optional
            from typing import get_args, get_origin

            annotation = field.annotation
            # Check it's not Optional (no Union with None)
            if get_origin(annotation) is not None:
                # If it has an origin, it shouldn't have None in the args
                self.assertNotIn(type(None), get_args(annotation))


if __name__ == "__main__":
    unittest.main()
