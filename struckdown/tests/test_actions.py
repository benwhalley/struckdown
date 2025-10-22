"""
Tests for the Actions registry system.

Tests custom action registration, parameter parsing, type coercion,
error handling, and integration with template syntax.
"""

import unittest

from struckdown.actions import Actions
from struckdown.parsing import parse_syntax
from struckdown.return_type_models import ResponseModel


class ActionRegistryTestCase(unittest.TestCase):
    """Test basic action registration and lookup"""

    def setUp(self):
        # clear registry before each test
        Actions._registry.clear()

    def test_register_action(self):
        """Test registering a simple action"""
        @Actions.register('test_action')
        def test_func(context):
            return "result"

        self.assertTrue(Actions.is_registered('test_action'))
        self.assertIn('test_action', Actions.list_registered())

    def test_action_not_registered(self):
        """Test checking for non-existent action"""
        self.assertFalse(Actions.is_registered('nonexistent'))
        self.assertEqual(Actions.list_registered(), [])

    def test_multiple_registrations(self):
        """Test registering multiple actions"""
        @Actions.register('action1')
        def func1(context):
            return "1"

        @Actions.register('action2')
        def func2(context):
            return "2"

        registered = Actions.list_registered()
        self.assertEqual(len(registered), 2)
        self.assertIn('action1', registered)
        self.assertIn('action2', registered)


class SimpleActionsTestCase(unittest.TestCase):
    """Test simple actions without parameters"""

    def setUp(self):
        Actions._registry.clear()

        # register uppercase action
        @Actions.register('uppercase')
        def uppercase_action(context):
            """Convert context to uppercase"""
            return "HELLO WORLD"

        @Actions.register('lowercase')
        def lowercase_action(context):
            """Convert to lowercase"""
            return "hello world"

    def test_uppercase_action_model_creation(self):
        """Test creating action model for uppercase"""
        model = Actions.create_action_model('uppercase', None, None, False)

        self.assertIsNotNone(model)
        self.assertTrue(hasattr(model, '_executor'))
        self.assertTrue(hasattr(model, '_is_function'))
        self.assertTrue(model._is_function)

    def test_uppercase_action_execution(self):
        """Test executing uppercase action"""
        model = Actions.create_action_model('uppercase', None, None, False)
        executor = model._executor

        result, completion = executor(context={}, rendered_prompt="")

        self.assertIsInstance(result, ResponseModel)
        self.assertEqual(result.response, "HELLO WORLD")
        self.assertIsNone(completion)

    def test_nonexistent_action_returns_none(self):
        """Test that nonexistent actions return None"""
        model = Actions.create_action_model('nonexistent', None, None, False)
        self.assertIsNone(model)


class ParameterizedActionsTestCase(unittest.TestCase):
    """Test actions with parameters"""

    def setUp(self):
        Actions._registry.clear()

        # register repeat action
        @Actions.register('repeat')
        def repeat_action(context, text: str, times: int = 2):
            """Repeat text N times"""
            return text * times

        # register concat action
        @Actions.register('concat')
        def concat_action(context, prefix: str = "", suffix: str = ""):
            """Add prefix and suffix"""
            return f"{prefix}MIDDLE{suffix}"

    def test_repeat_with_parameters(self):
        """Test action with required and optional parameters"""
        model = Actions.create_action_model(
            'repeat',
            ['text=hello', 'times=3'],
            None,
            False
        )

        executor = model._executor
        result, _ = executor(context={}, rendered_prompt="")

        self.assertEqual(result.response, "hellohellohello")

    def test_repeat_with_default_parameter(self):
        """Test action using default parameter value"""
        model = Actions.create_action_model(
            'repeat',
            ['text=hi'],
            None,
            False
        )

        executor = model._executor
        result, _ = executor(context={}, rendered_prompt="")

        self.assertEqual(result.response, "hihi")  # default times=2

    def test_concat_with_both_parameters(self):
        """Test action with multiple optional parameters"""
        model = Actions.create_action_model(
            'concat',
            ['prefix=START-', 'suffix=-END'],
            None,
            False
        )

        executor = model._executor
        result, _ = executor(context={}, rendered_prompt="")

        self.assertEqual(result.response, "START-MIDDLE-END")

    def test_concat_with_partial_parameters(self):
        """Test action with some parameters omitted"""
        model = Actions.create_action_model(
            'concat',
            ['prefix=>>>'],
            None,
            False
        )

        executor = model._executor
        result, _ = executor(context={}, rendered_prompt="")

        self.assertEqual(result.response, ">>>MIDDLE")


class TypeCoercionTestCase(unittest.TestCase):
    """Test automatic type coercion of parameters"""

    def setUp(self):
        Actions._registry.clear()

        # register multiply action with int type
        @Actions.register('multiply')
        def multiply_action(context, value: int, factor: int = 2):
            """Multiply value by factor"""
            return str(value * factor)

        # register boolean action
        @Actions.register('ifelse')
        def ifelse_action(context, condition: bool, true_val: str = "yes", false_val: str = "no"):
            """Return true_val if condition else false_val"""
            return true_val if condition else false_val

    def test_string_to_int_coercion(self):
        """Test that string parameters are coerced to int"""
        model = Actions.create_action_model(
            'multiply',
            ['value=10', 'factor=5'],
            None,
            False
        )

        executor = model._executor
        result, _ = executor(context={}, rendered_prompt="")

        self.assertEqual(result.response, "50")

    def test_string_to_bool_coercion(self):
        """Test that string parameters are coerced to bool"""
        model = Actions.create_action_model(
            'ifelse',
            ['condition=true', 'true_val=YES', 'false_val=NO'],
            None,
            False
        )

        executor = model._executor
        result, _ = executor(context={}, rendered_prompt="")

        self.assertEqual(result.response, "YES")

        # test with false
        model2 = Actions.create_action_model(
            'ifelse',
            ['condition=false'],
            None,
            False
        )

        result2, _ = model2._executor(context={}, rendered_prompt="")
        self.assertEqual(result2.response, "no")


class ContextVariablesTestCase(unittest.TestCase):
    """Test actions using context variables"""

    def setUp(self):
        Actions._registry.clear()

        # register action that uses context
        @Actions.register('greet')
        def greet_action(context, name: str):
            """Greet using name"""
            return f"Hello, {name}!"

        # register action that accesses context directly
        @Actions.register('count_vars')
        def count_vars_action(context):
            """Count variables in context"""
            return f"Context has {len(context)} variables"

    def test_parameter_from_context_variable(self):
        """Test parameter value resolved from context"""
        model = Actions.create_action_model(
            'greet',
            ['name={{username}}'],
            None,
            False
        )

        # execute with context containing username
        executor = model._executor
        result, _ = executor(context={'username': 'Alice'}, rendered_prompt="")

        self.assertEqual(result.response, "Hello, Alice!")

    def test_action_accessing_context_directly(self):
        """Test action that reads context directly"""
        model = Actions.create_action_model('count_vars', None, None, False)

        executor = model._executor
        result, _ = executor(
            context={'var1': 'a', 'var2': 'b', 'var3': 'c'},
            rendered_prompt=""
        )

        self.assertEqual(result.response, "Context has 3 variables")


class ErrorHandlingTestCase(unittest.TestCase):
    """Test error handling strategies"""

    def setUp(self):
        Actions._registry.clear()

    def test_error_propagate_strategy(self):
        """Test that errors are propagated by default"""
        @Actions.register('failing', on_error='propagate')
        def failing_action(context):
            raise ValueError("Intentional error")

        model = Actions.create_action_model('failing', None, None, False)
        executor = model._executor

        with self.assertRaises(ValueError) as ctx:
            executor(context={}, rendered_prompt="")

        self.assertIn("Intentional error", str(ctx.exception))

    def test_error_return_empty_strategy(self):
        """Test return_empty error strategy"""
        @Actions.register('failing', on_error='return_empty')
        def failing_action(context):
            raise ValueError("Intentional error")

        model = Actions.create_action_model('failing', None, None, False)
        executor = model._executor

        result, _ = executor(context={}, rendered_prompt="")
        self.assertEqual(result.response, "")

    def test_error_log_and_continue_strategy(self):
        """Test log_and_continue error strategy"""
        @Actions.register('failing', on_error='log_and_continue')
        def failing_action(context):
            raise ValueError("Intentional error")

        model = Actions.create_action_model('failing', None, None, False)
        executor = model._executor

        result, _ = executor(context={}, rendered_prompt="")
        self.assertEqual(result.response, "")


class TemplateSyntaxIntegrationTestCase(unittest.TestCase):
    """Test actions integrated with template parsing"""

    def setUp(self):
        Actions._registry.clear()

        # register uppercase action
        @Actions.register('uppercase')
        def uppercase_action(context, text: str = "default"):
            """Convert text to uppercase"""
            return text.upper()

        # register reverse action
        @Actions.register('reverse')
        def reverse_action(context, text: str):
            """Reverse text"""
            return text[::-1]

    def test_parse_function_call_syntax(self):
        """Test parsing [[@action:var|params]] syntax"""
        template = "Convert this: [[@uppercase:result|text=hello]]"
        sections = parse_syntax(template)

        self.assertEqual(len(sections), 1)
        self.assertIn('result', sections[0])

        part = sections[0]['result']
        self.assertEqual(part.action_type, 'uppercase')
        self.assertTrue(part.is_function)
        self.assertIn('text=hello', part.options)

    def test_action_in_template_with_variable(self):
        """Test action using template variable"""
        template = "Original: [[input]]\n\n¡OBLIVIATE\n\nReversed: [[@reverse:output|text={{input}}]]"
        sections = parse_syntax(template)

        self.assertEqual(len(sections), 2)

        # first section has input completion
        self.assertIn('input', sections[0])

        # second section has reverse action
        self.assertIn('output', sections[1])
        part = sections[1]['output']
        self.assertEqual(part.action_type, 'reverse')
        self.assertTrue(part.is_function)

    def test_multiple_actions_in_template(self):
        """Test template with multiple actions"""
        template = """
        [[@uppercase:upper|text=hello]]
        [[@reverse:rev|text=world]]
        """
        sections = parse_syntax(template)

        self.assertEqual(len(sections), 1)
        section = sections[0]

        self.assertIn('upper', section)
        self.assertIn('rev', section)

        self.assertEqual(section['upper'].action_type, 'uppercase')
        self.assertEqual(section['rev'].action_type, 'reverse')


class RealWorldExampleTestCase(unittest.TestCase):
    """Test realistic action use cases"""

    def setUp(self):
        Actions._registry.clear()

        # register uppercase action (like the user requested)
        @Actions.register('uppercase', on_error='return_empty')
        def uppercase_action(context, text: str):
            """Convert text to UPPERCASE"""
            return text.upper()

        # register transform action with multiple operations
        @Actions.register('transform')
        def transform_action(context, text: str, operation: str = "upper"):
            """Transform text using specified operation"""
            if operation == "upper":
                return text.upper()
            elif operation == "lower":
                return text.lower()
            elif operation == "title":
                return text.title()
            elif operation == "reverse":
                return text[::-1]
            else:
                return text

    def test_uppercase_simple(self):
        """Test simple uppercase action"""
        model = Actions.create_action_model(
            'uppercase',
            ['text=hello world'],
            None,
            False
        )

        result, _ = model._executor(context={}, rendered_prompt="")
        self.assertEqual(result.response, "HELLO WORLD")

    def test_uppercase_with_context_variable(self):
        """Test uppercase using context variable"""
        model = Actions.create_action_model(
            'uppercase',
            ['text={{user_input}}'],
            None,
            False
        )

        result, _ = model._executor(
            context={'user_input': 'make me loud'},
            rendered_prompt=""
        )
        self.assertEqual(result.response, "MAKE ME LOUD")

    def test_transform_with_different_operations(self):
        """Test transform action with multiple operations"""
        # test upper
        model = Actions.create_action_model(
            'transform',
            ['text=hello', 'operation=upper'],
            None,
            False
        )
        result, _ = model._executor(context={}, rendered_prompt="")
        self.assertEqual(result.response, "HELLO")

        # test lower
        model2 = Actions.create_action_model(
            'transform',
            ['text=HELLO', 'operation=lower'],
            None,
            False
        )
        result2, _ = model2._executor(context={}, rendered_prompt="")
        self.assertEqual(result2.response, "hello")

        # test title
        model3 = Actions.create_action_model(
            'transform',
            ['text=hello world', 'operation=title'],
            None,
            False
        )
        result3, _ = model3._executor(context={}, rendered_prompt="")
        self.assertEqual(result3.response, "Hello World")

        # test reverse
        model4 = Actions.create_action_model(
            'transform',
            ['text=hello', 'operation=reverse'],
            None,
            False
        )
        result4, _ = model4._executor(context={}, rendered_prompt="")
        self.assertEqual(result4.response, "olleh")

    def test_uppercase_in_complete_template(self):
        """Test uppercase action in a complete template workflow"""
        template = """
        Get user input: [[user_text]]

        ¡OBLIVIATE

        Transform to uppercase: [[@uppercase:loud_text|text={{user_text}}]]
        """

        sections = parse_syntax(template)
        self.assertEqual(len(sections), 2)

        # verify first section
        self.assertIn('user_text', sections[0])

        # verify second section has action
        self.assertIn('loud_text', sections[1])
        action_part = sections[1]['loud_text']
        self.assertEqual(action_part.action_type, 'uppercase')
        self.assertTrue(action_part.is_function)


if __name__ == "__main__":
    unittest.main()
