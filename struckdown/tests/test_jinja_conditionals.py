"""
Tests for Jinja conditional syntax in struckdown templates.

Based on conditionals.sd test scenarios:
1. Nested conditionals
2. OR conditions
3. AND conditions
4. Negation
5. Elif chains with string comparison
"""

import pytest
from jinja2.sandbox import ImmutableSandboxedEnvironment

from struckdown.jinja_analysis import (
    analyze_template,
    extract_variables_from_expression,
    SlotDependency,
    TemplateAnalysis,
)
from struckdown.segment_processor import render_template
from struckdown.parsing import find_slots_with_positions


# =============================================================================
# Test Templates (extracted from conditionals.sd)
# =============================================================================

TEMPLATE_NESTED = """
Is the sun hot? [[!bool:sunhot]]

{% if sunhot %}
Is the moon cold? [[!bool:mooncold]]

{% if mooncold %}
Is the forest wet? [[!bool:forestwet]]
{% endif %}
{% endif %}

Describe what you know: [[respond:description]]
"""

TEMPLATE_OR = """
Is there water? [[!bool:has_water]]
Is there fire? [[!bool:has_fire]]

{% if has_water or has_fire %}
Which element dominates? [[!pick:dominant|water,fire,neither]]
{% endif %}
"""

TEMPLATE_AND = """
Is it daytime? [[!bool:is_day]]
Is it sunny? [[!bool:is_sunny]]

{% if is_day and is_sunny %}
How bright is it? [[!pick:brightness|dim,moderate,bright,blinding]]
{% endif %}
"""

TEMPLATE_NOT = """
Is it safe? [[!bool:is_safe]]

{% if not is_safe %}
What is the danger? [[respond:danger_description]]
{% endif %}
"""

TEMPLATE_ELIF = """
Rate the temperature: [[!pick:temperature|hot,warm,cold]]

{% if temperature == "hot" %}
Recommend cooling action: [[respond:cooling_action]]
{% elif temperature == "cold" %}
Recommend warming action: [[respond:warming_action]]
{% else %}
No action needed. Comfortable!
{% endif %}
"""


# =============================================================================
# TestJinjaAnalysis - Pure unit tests for jinja_analysis.py
# =============================================================================

class TestJinjaAnalysis:
    """Test the AST analysis of conditional templates."""

    def test_nested_conditional_analysis(self):
        """Nested conditionals correctly identify dependencies."""
        analysis = analyze_template(TEMPLATE_NESTED)

        # sunhot triggers re-render because mooncold depends on it
        assert analysis.triggers_rerender("sunhot"), "sunhot should trigger re-render"

        # mooncold triggers re-render because forestwet depends on it
        assert analysis.triggers_rerender("mooncold"), "mooncold should trigger re-render"

        # description is unconditional, forestwet doesn't trigger anything
        assert not analysis.triggers_rerender("description"), "description shouldn't trigger"
        assert not analysis.triggers_rerender("forestwet"), "forestwet shouldn't trigger"

    def test_nested_slot_dependencies(self):
        """Nested slots have correct dependency chains."""
        analysis = analyze_template(TEMPLATE_NESTED)

        # Find slots by key
        slot_map = {s.key: s for s in analysis.slots}

        # sunhot is unconditional
        assert not slot_map["sunhot"].is_conditional

        # mooncold depends on sunhot
        assert slot_map["mooncold"].is_conditional
        assert "sunhot" in slot_map["mooncold"].depends_on

        # forestwet depends on both sunhot and mooncold
        assert slot_map["forestwet"].is_conditional
        assert "sunhot" in slot_map["forestwet"].depends_on
        assert "mooncold" in slot_map["forestwet"].depends_on

        # description is unconditional
        assert not slot_map["description"].is_conditional

    def test_or_condition_analysis(self):
        """OR condition extracts both variables."""
        analysis = analyze_template(TEMPLATE_OR)

        slot_map = {s.key: s for s in analysis.slots}

        # dominant depends on both has_water and has_fire
        assert slot_map["dominant"].is_conditional
        assert "has_water" in slot_map["dominant"].depends_on
        assert "has_fire" in slot_map["dominant"].depends_on

        # Both variables trigger re-render
        assert analysis.triggers_rerender("has_water")
        assert analysis.triggers_rerender("has_fire")

    def test_and_condition_analysis(self):
        """AND condition extracts both variables."""
        analysis = analyze_template(TEMPLATE_AND)

        slot_map = {s.key: s for s in analysis.slots}

        # brightness depends on both is_day and is_sunny
        assert slot_map["brightness"].is_conditional
        assert "is_day" in slot_map["brightness"].depends_on
        assert "is_sunny" in slot_map["brightness"].depends_on

        # Both variables trigger re-render
        assert analysis.triggers_rerender("is_day")
        assert analysis.triggers_rerender("is_sunny")

    def test_negation_analysis(self):
        """Negation correctly identifies dependency."""
        analysis = analyze_template(TEMPLATE_NOT)

        slot_map = {s.key: s for s in analysis.slots}

        # danger_description depends on is_safe (via not)
        assert slot_map["danger_description"].is_conditional
        assert "is_safe" in slot_map["danger_description"].depends_on

        # is_safe triggers re-render
        assert analysis.triggers_rerender("is_safe")

    def test_elif_string_comparison(self):
        """Elif chain with string comparison correctly identifies dependency."""
        analysis = analyze_template(TEMPLATE_ELIF)

        slot_map = {s.key: s for s in analysis.slots}

        # Both slots depend on temperature
        assert slot_map["cooling_action"].is_conditional
        assert "temperature" in slot_map["cooling_action"].depends_on

        assert slot_map["warming_action"].is_conditional
        assert "temperature" in slot_map["warming_action"].depends_on

        # temperature triggers re-render
        assert analysis.triggers_rerender("temperature")

    def test_unconditional_slots_identified(self):
        """Unconditional slots are correctly identified."""
        analysis = analyze_template(TEMPLATE_NESTED)

        unconditional = analysis.unconditional_slots
        conditional = analysis.conditional_slots

        assert "sunhot" in unconditional
        assert "description" in unconditional
        assert "mooncold" in conditional
        assert "forestwet" in conditional


# =============================================================================
# TestSlotVisibility - Test slot visibility after Jinja rendering
# =============================================================================

class TestSlotVisibility:
    """Test which slots are visible after Jinja rendering with various contexts."""

    def test_nested_no_context(self):
        """Without context, only unconditional slots visible."""
        rendered = render_template(TEMPLATE_NESTED, {})

        # sunhot and description are always visible
        assert "[[!bool:sunhot]]" in rendered
        assert "[[respond:description]]" in rendered

        # Conditional slots should NOT be visible (conditions are falsy)
        assert "[[!bool:mooncold]]" not in rendered
        assert "[[!bool:forestwet]]" not in rendered

    def test_nested_outer_true_inner_false(self):
        """Outer true reveals first nested slot, but not deeper ones."""
        rendered = render_template(TEMPLATE_NESTED, {"sunhot": True})

        assert "[[!bool:sunhot]]" in rendered
        assert "[[!bool:mooncold]]" in rendered  # Now visible
        assert "[[!bool:forestwet]]" not in rendered  # Still hidden
        assert "[[respond:description]]" in rendered

    def test_nested_all_true(self):
        """All conditions true reveals all nested slots."""
        rendered = render_template(TEMPLATE_NESTED, {"sunhot": True, "mooncold": True})

        assert "[[!bool:sunhot]]" in rendered
        assert "[[!bool:mooncold]]" in rendered
        assert "[[!bool:forestwet]]" in rendered  # Now visible
        assert "[[respond:description]]" in rendered

    def test_nested_outer_false_hides_all(self):
        """Outer false hides all nested slots regardless of inner values."""
        rendered = render_template(
            TEMPLATE_NESTED, {"sunhot": False, "mooncold": True, "forestwet": True}
        )

        assert "[[!bool:mooncold]]" not in rendered
        assert "[[!bool:forestwet]]" not in rendered

    def test_or_neither_true(self):
        """OR: slot hidden when both false."""
        rendered = render_template(TEMPLATE_OR, {"has_water": False, "has_fire": False})
        assert "[[!pick:dominant" not in rendered

    def test_or_first_true(self):
        """OR: slot visible when first true."""
        rendered = render_template(TEMPLATE_OR, {"has_water": True, "has_fire": False})
        assert "[[!pick:dominant" in rendered

    def test_or_second_true(self):
        """OR: slot visible when second true."""
        rendered = render_template(TEMPLATE_OR, {"has_water": False, "has_fire": True})
        assert "[[!pick:dominant" in rendered

    def test_or_both_true(self):
        """OR: slot visible when both true."""
        rendered = render_template(TEMPLATE_OR, {"has_water": True, "has_fire": True})
        assert "[[!pick:dominant" in rendered

    def test_and_neither_true(self):
        """AND: slot hidden when both false."""
        rendered = render_template(TEMPLATE_AND, {"is_day": False, "is_sunny": False})
        assert "[[!pick:brightness" not in rendered

    def test_and_first_only_true(self):
        """AND: slot hidden when only first true."""
        rendered = render_template(TEMPLATE_AND, {"is_day": True, "is_sunny": False})
        assert "[[!pick:brightness" not in rendered

    def test_and_second_only_true(self):
        """AND: slot hidden when only second true."""
        rendered = render_template(TEMPLATE_AND, {"is_day": False, "is_sunny": True})
        assert "[[!pick:brightness" not in rendered

    def test_and_both_true(self):
        """AND: slot visible only when both true."""
        rendered = render_template(TEMPLATE_AND, {"is_day": True, "is_sunny": True})
        assert "[[!pick:brightness" in rendered

    def test_not_true_hides_slot(self):
        """NOT: slot hidden when condition true."""
        rendered = render_template(TEMPLATE_NOT, {"is_safe": True})
        assert "[[respond:danger_description]]" not in rendered

    def test_not_false_shows_slot(self):
        """NOT: slot visible when condition false."""
        rendered = render_template(TEMPLATE_NOT, {"is_safe": False})
        assert "[[respond:danger_description]]" in rendered

    def test_elif_hot_branch(self):
        """Elif: hot shows cooling_action only."""
        rendered = render_template(TEMPLATE_ELIF, {"temperature": "hot"})
        assert "[[respond:cooling_action]]" in rendered
        assert "[[respond:warming_action]]" not in rendered
        assert "Comfortable" not in rendered

    def test_elif_cold_branch(self):
        """Elif: cold shows warming_action only."""
        rendered = render_template(TEMPLATE_ELIF, {"temperature": "cold"})
        assert "[[respond:cooling_action]]" not in rendered
        assert "[[respond:warming_action]]" in rendered
        assert "Comfortable" not in rendered

    def test_elif_else_branch(self):
        """Elif: warm (else) shows no slot, just text."""
        rendered = render_template(TEMPLATE_ELIF, {"temperature": "warm"})
        assert "[[respond:cooling_action]]" not in rendered
        assert "[[respond:warming_action]]" not in rendered
        assert "Comfortable" in rendered


# =============================================================================
# TestEdgeCases - Truthiness and edge cases
# =============================================================================

class TestEdgeCases:
    """Test truthiness edge cases and boundary conditions."""

    def test_bool_true_is_truthy(self):
        """Python True is truthy in Jinja."""
        template = "{% if x %}visible{% endif %}"
        rendered = render_template(template, {"x": True})
        assert "visible" in rendered

    def test_bool_false_is_falsy(self):
        """Python False is falsy in Jinja."""
        template = "{% if x %}visible{% endif %}"
        rendered = render_template(template, {"x": False})
        assert "visible" not in rendered

    def test_string_true_is_truthy(self):
        """Non-empty string 'true' is truthy."""
        template = "{% if x %}visible{% endif %}"
        rendered = render_template(template, {"x": "true"})
        assert "visible" in rendered

    def test_string_false_is_truthy(self):
        """Non-empty string 'false' is truthy (it's a string, not boolean)."""
        template = "{% if x %}visible{% endif %}"
        rendered = render_template(template, {"x": "false"})
        assert "visible" in rendered  # "false" is a non-empty string = truthy

    def test_empty_string_is_falsy(self):
        """Empty string is falsy."""
        template = "{% if x %}visible{% endif %}"
        rendered = render_template(template, {"x": ""})
        assert "visible" not in rendered

    def test_integer_zero_is_falsy(self):
        """Integer 0 is falsy."""
        template = "{% if x %}visible{% endif %}"
        rendered = render_template(template, {"x": 0})
        assert "visible" not in rendered

    def test_integer_nonzero_is_truthy(self):
        """Non-zero integers are truthy."""
        template = "{% if x %}visible{% endif %}"
        rendered = render_template(template, {"x": 42})
        assert "visible" in rendered

    def test_none_is_falsy(self):
        """None is falsy."""
        template = "{% if x %}visible{% endif %}"
        rendered = render_template(template, {"x": None})
        assert "visible" not in rendered

    def test_empty_list_is_falsy(self):
        """Empty list is falsy."""
        template = "{% if x %}visible{% endif %}"
        rendered = render_template(template, {"x": []})
        assert "visible" not in rendered

    def test_nonempty_list_is_truthy(self):
        """Non-empty list is truthy."""
        template = "{% if x %}visible{% endif %}"
        rendered = render_template(template, {"x": [1, 2, 3]})
        assert "visible" in rendered

    def test_string_comparison_exact_match(self):
        """String comparison requires exact match."""
        template = '{% if x == "hot" %}match{% endif %}'
        assert "match" in render_template(template, {"x": "hot"})
        assert "match" not in render_template(template, {"x": "Hot"})
        assert "match" not in render_template(template, {"x": "HOT"})
        assert "match" not in render_template(template, {"x": "warm"})

    def test_undefined_variable_is_falsy(self):
        """Undefined variable treated as falsy (with SilentUndefined)."""
        template = "{% if undefined_var %}visible{% endif %}"
        rendered = render_template(template, {})
        assert "visible" not in rendered

    def test_complex_nested_expression(self):
        """Complex nested boolean expression."""
        template = "{% if (a or b) and (c or d) %}visible{% endif %}"

        # Neither side true
        assert "visible" not in render_template(
            template, {"a": False, "b": False, "c": False, "d": False}
        )

        # First side true, second false
        assert "visible" not in render_template(
            template, {"a": True, "b": False, "c": False, "d": False}
        )

        # Both sides true
        assert "visible" in render_template(
            template, {"a": True, "b": False, "c": False, "d": True}
        )


# =============================================================================
# TestSlotPositionTracking - Test find_slots_with_positions
# =============================================================================

class TestSlotPositionTracking:
    """Test slot position tracking for delta computation."""

    def test_find_single_slot(self):
        """Find single slot with correct position."""
        text = "Hello [[name]] world"
        slots = find_slots_with_positions(text)

        assert len(slots) == 1
        key, start, end, inner = slots[0]
        assert key == "name"
        assert text[start:end] == "[[name]]"

    def test_find_multiple_slots(self):
        """Find multiple slots in order."""
        text = "[[first]] middle [[second]] end [[third]]"
        slots = find_slots_with_positions(text)

        assert len(slots) == 3
        assert slots[0][0] == "first"
        assert slots[1][0] == "second"
        assert slots[2][0] == "third"

        # Verify positions are ascending
        assert slots[0][1] < slots[1][1] < slots[2][1]

    def test_find_typed_slot(self):
        """Find typed slot extracts correct key."""
        text = "Value: [[!bool:is_valid]]"
        slots = find_slots_with_positions(text)

        assert len(slots) == 1
        key, start, end, inner = slots[0]
        assert key == "is_valid"

    def test_find_slot_with_options(self):
        """Find slot with options extracts correct key."""
        text = "Choice: [[!pick:color|red,green,blue]]"
        slots = find_slots_with_positions(text)

        assert len(slots) == 1
        key, _, _, _ = slots[0]
        assert key == "color"

    def test_no_slots_returns_empty(self):
        """No slots returns empty list."""
        text = "Plain text without any slots"
        slots = find_slots_with_positions(text)
        assert slots == []


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
