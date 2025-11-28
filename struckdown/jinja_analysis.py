"""Jinja AST analysis for conditional slot dependencies.

Analyzes struckdown templates to understand which slots depend on which variables,
enabling smart re-rendering decisions during execution.
"""

from collections import defaultdict
from dataclasses import dataclass, field
from functools import reduce
from typing import FrozenSet, List, Dict, Tuple

from jinja2 import Environment, nodes

from .parsing import SLOT_PATTERN, extract_slot_key, find_slots_with_positions


@dataclass(frozen=True)
class SlotDependency:
    """A slot and its conditional dependencies."""
    key: str
    conditions: Tuple[Tuple[FrozenSet[str], bool], ...]  # ((test_vars, must_be_truthy), ...)

    @property
    def is_conditional(self) -> bool:
        return len(self.conditions) > 0

    @property
    def depends_on(self) -> FrozenSet[str]:
        """All variables this slot depends on (union of all condition variables)."""
        if not self.conditions:
            return frozenset()
        return reduce(lambda a, b: a | b, (vars for vars, _ in self.conditions))


@dataclass
class TemplateAnalysis:
    """Result of analyzing a template's structure."""
    slots: List[SlotDependency] = field(default_factory=list)
    triggers: Dict[str, List[str]] = field(default_factory=dict)

    @property
    def unconditional_slots(self) -> List[str]:
        """Slots that are always visible (no conditions)."""
        return [s.key for s in self.slots if not s.is_conditional]

    @property
    def conditional_slots(self) -> List[str]:
        """Slots that depend on other variables."""
        return [s.key for s in self.slots if s.is_conditional]

    def triggers_rerender(self, slot_key: str) -> bool:
        """Check if filling this slot should trigger a re-render."""
        return slot_key in self.triggers


def extract_variables_from_expression(node: nodes.Node) -> FrozenSet[str]:
    """Recursively extract all variable names from a Jinja expression node.

    Handles: Name, Compare, BinExpr, UnaryExpr, Call, Getattr, Filter, Test, etc.

    Args:
        node: A Jinja2 AST node

    Returns:
        Frozenset of variable names used in the expression
    """
    if isinstance(node, nodes.Name):
        return frozenset({node.name})

    if node is None:
        return frozenset()

    # Attributes that might contain sub-expressions
    expr_attrs = ('left', 'right', 'expr', 'node', 'test', 'arg', 'operand')
    list_attrs = ('args', 'ops', 'items', 'kwargs', 'values', 'keys')

    result = frozenset()

    # Single expression attributes
    for attr in expr_attrs:
        child = getattr(node, attr, None)
        if isinstance(child, nodes.Node):
            result = result | extract_variables_from_expression(child)

    # List attributes
    for attr in list_attrs:
        children = getattr(node, attr, None)
        if children:
            for child in children:
                if isinstance(child, nodes.Node):
                    result = result | extract_variables_from_expression(child)
                # Handle Pair nodes in kwargs
                elif isinstance(child, tuple) and len(child) == 2:
                    _, val = child
                    if isinstance(val, nodes.Node):
                        result = result | extract_variables_from_expression(val)

    return result


def find_slots_in_template_data(data: str) -> List[str]:
    """Find all slot keys in a TemplateData string."""
    return [extract_slot_key(m.group(1)) for m in SLOT_PATTERN.finditer(data)]


def analyze_template(template_str: str) -> TemplateAnalysis:
    """Parse template into Jinja AST and extract slot dependencies.

    Walks the AST to find all slots and determine which conditionals they're inside.
    Builds a trigger map showing which slots may appear when a variable changes.

    Args:
        template_str: The raw template string (before Jinja rendering)

    Returns:
        TemplateAnalysis with slots and triggers
    """
    try:
        ast = Environment().parse(template_str)
    except Exception:
        # If Jinja parsing fails, return empty analysis
        # (the error will surface later during actual rendering)
        return TemplateAnalysis()

    slots: List[SlotDependency] = []

    def walk(node: nodes.Node, conditions: Tuple[Tuple[FrozenSet[str], bool], ...] = ()):
        """Recursively walk AST, tracking active conditions."""

        if isinstance(node, nodes.TemplateData):
            # Found raw text - check for slots
            for key in find_slots_in_template_data(node.data):
                slots.append(SlotDependency(key=key, conditions=conditions))

        elif isinstance(node, nodes.If):
            # Conditional block - extract test variables
            test_vars = extract_variables_from_expression(node.test)

            # Walk body with this condition added (must be truthy)
            body_conditions = conditions + ((test_vars, True),)
            for child in node.body:
                walk(child, body_conditions)

            # Walk elif_ (list of If nodes)
            for elif_node in node.elif_:
                walk(elif_node, conditions)

            # Walk else_ with negated condition
            if node.else_:
                else_conditions = conditions + ((test_vars, False),)
                for child in node.else_:
                    walk(child, else_conditions)

        elif isinstance(node, nodes.For):
            # For loop - slots inside depend on iterable being non-empty
            iter_vars = extract_variables_from_expression(node.iter)
            loop_conditions = conditions + ((iter_vars, True),)
            for child in node.body:
                walk(child, loop_conditions)
            # else block runs if loop didn't iterate
            if node.else_:
                else_conditions = conditions + ((iter_vars, False),)
                for child in node.else_:
                    walk(child, else_conditions)

        elif isinstance(node, nodes.Output):
            # Output nodes contain expressions to print
            for child in node.nodes:
                walk(child, conditions)

        elif hasattr(node, 'body') and isinstance(node.body, list):
            # Generic container - walk children
            for child in node.body:
                walk(child, conditions)

    walk(ast)

    # Build trigger map: var -> [slots that may appear when it changes]
    triggers: Dict[str, List[str]] = defaultdict(list)
    for slot in slots:
        for test_vars, _ in slot.conditions:
            for var in test_vars:
                if slot.key not in triggers[var]:
                    triggers[var].append(slot.key)

    return TemplateAnalysis(slots=slots, triggers=dict(triggers))
