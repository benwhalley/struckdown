"""
Generate dependency graph visualizations from struckdown prompts.

Shows structure, dependencies, and flow of completions in Mermaid DAG format.
"""

import re
import textwrap
from typing import Dict, List, Set, Optional

from struckdown import extract_jinja_variables
from struckdown.parsing import get_completion_type


def analyze_sections(sections: List[Dict]) -> Dict:
    """Analyze parsed struckdown sections.

    Args:
        sections: List of OrderedDict sections from parse_syntax

    Returns:
        Dictionary with structure:
        {
            'sections': [
                {
                    'index': 0,
                    'completions': [
                        {
                            'key': 'checklist1',
                            'type': 'think',
                            'depends_on': ['source'],
                            'line_number': 162,
                            'is_blocking': False
                        },
                        ...
                    ]
                },
                ...
            ],
            'all_completions': {'checklist1', 'checklist2', ...},
            'all_dependencies': [('source', 'checklist1'), ...],
            'blocking_dependencies': [(0, 1), (0, 2), ...]  # (from_section, to_section) tuples
        }
    """
    all_completions = set()
    all_dependencies = []
    blocking_dependencies = []  # Track blocking section dependencies

    # Analyze each section
    structure = {
        'sections': [],
        'all_completions': set(),
        'all_dependencies': [],
        'blocking_dependencies': []
    }

    # First pass: collect completions and variable dependencies
    for section_idx, section in enumerate(sections):
        # Extract segment name if available (NamedSegment has segment_name property)
        segment_name = getattr(section, 'segment_name', None)
        section_data = {
            'index': section_idx,
            'name': segment_name,
            'completions': [],
            'has_blocking': False
        }

        for key, prompt_part in section.items():
            # Extract dependencies from prompt text
            depends_on = extract_jinja_variables(prompt_part.text)

            # Get completion type
            comp_type = get_completion_type(prompt_part)

            # Get line number
            line_num = prompt_part.line_number

            # Check if this completion is blocking
            is_blocking = hasattr(prompt_part, 'block') and prompt_part.block
            if is_blocking:
                section_data['has_blocking'] = True

            # Check if this is an end action (terminates execution)
            is_end_action = (
                hasattr(prompt_part, 'action_type') and
                prompt_part.action_type == 'end'
            )
            if is_end_action:
                section_data['has_end_action'] = True

            completion_data = {
                'key': key,
                'type': comp_type,
                'depends_on': sorted(depends_on),
                'line_number': line_num,
                'is_blocking': is_blocking
            }

            section_data['completions'].append(completion_data)
            all_completions.add(key)

            # Record dependencies
            for dep in depends_on:
                all_dependencies.append((dep, key))

        structure['sections'].append(section_data)

    # Second pass: handle blocking dependencies
    # If a section has a blocking completion, all subsequent sections depend on it
    # BUT: Skip sections with [[!end]] actions - they terminate execution, not block it
    for i, section in enumerate(structure['sections']):
        if section.get('has_blocking') and not section.get('has_end_action'):
            for j in range(i + 1, len(structure['sections'])):
                blocking_dependencies.append((i, j))

    structure['all_completions'] = all_completions
    structure['all_dependencies'] = all_dependencies
    structure['blocking_dependencies'] = blocking_dependencies

    return structure


def _wrap_text(text: str, max_length: int = 25) -> str:
    """Wrap text at word boundaries to fit within max_length.

    Args:
        text: Text to wrap
        max_length: Maximum characters per line

    Returns:
        Text with <br/> line breaks inserted
    """
    lines = textwrap.wrap(
        text,
        width=max_length,
        break_long_words=False,
        break_on_hyphens=False
    )
    return '<br/>'.join(lines)


def _escape_for_summary(text: str) -> str:
    """Escape text so it's not interpreted by struckdown when summarizing.

    Escapes both:
    - Struckdown commands (¡SYSTEM, ¡OBLIVIATE, etc.)
    - Template syntax ({{variables}} and [[slots]])

    Args:
        text: Text to escape

    Returns:
        Escaped text safe to pass through struckdown
    """
    from struckdown import escape_struckdown_syntax

    # First escape struckdown commands
    escaped, _ = escape_struckdown_syntax(text)

    # Then escape template syntax for Jinja2
    escaped = escaped.replace('{{', r'\{\{').replace('}}', r'\}\}')
    escaped = escaped.replace('[[', r'\[\[').replace(']]', r'\]\]')

    return escaped


def generate_mermaid_dag(structure: Dict, title: str = "Struckdown") -> str:
    """Generate Mermaid DAG from analyzed structure.

    Args:
        structure: Output from analyze_sections
        title: Diagram title

    Returns:
        Mermaid diagram as string
    """
    lines = ['graph TD']

    # Track which variables are defined vs referenced
    defined_vars = structure['all_completions']
    referenced_vars = {dep for dep, _ in structure['all_dependencies']}

    # External inputs (referenced but not defined)
    external_inputs = referenced_vars - defined_vars

    # Add external inputs
    for var in sorted(external_inputs):
        lines.append(f'    {var}([{var}])')

    if external_inputs:
        lines.append('')

    # Build mapping of completion variable -> section index (for unique node IDs)
    completion_to_section_map = {}
    for section in structure['sections']:
        for comp in section['completions']:
            completion_to_section_map[comp['key']] = section['index']

    # Add sections with completions
    for section in structure['sections']:
        section_idx = section['index']
        section_name = section.get('name')
        # Use segment name if available, otherwise fall back to "Checkpoint N"
        section_label = section_name if section_name else f"Checkpoint {section_idx + 1}"
        lines.append(f'    subgraph S{section_idx}["S{section_idx}: {section_label}"]')

        for comp in section['completions']:
            key = comp['key']
            depends_on = comp['depends_on']
            line_num = comp.get('line_number', 0)
            is_blocking = comp.get('is_blocking', False)

            # Use unique node ID: section_index + key
            node_id = f'S{section_idx}_{key}'

            # Format label with line number and dependencies
            if line_num:
                name_part = f'<strong>{key}</strong>: <small>{line_num}</small>'
            else:
                name_part = key

            # Add blocking marker
            if is_blocking:
                name_part += ' <strong>🔒</strong>'

            if depends_on:
                deps_str = ', '.join(depends_on)
                wrapped_deps = _wrap_text(deps_str, max_length=25)
                label = f'{name_part}<br/>← <span class="dependencies">{wrapped_deps}</span>'
            else:
                label = name_part

            lines.append(f'        {node_id}["{label}"]')

        lines.append('    end')
        lines.append('')

    # Build mapping of completion variable -> section index where it's defined
    # Handle duplicates: for each target, find which definition of dep it sees

    # Add dependencies
    lines.append('    %% Dependencies')

    # Add explicit dependencies from {{variable}} references
    for dep, target in structure['all_dependencies']:
        # Find which section defines target
        target_section = None
        for section in structure['sections']:
            for comp in section['completions']:
                if comp['key'] == target:
                    target_section = section['index']
                    break
            if target_section is not None:
                break

        # Find which section defines dep that target sees
        # (most recent definition before or in target's section)
        # Search in reverse order through earlier sections and same section
        dep_section = None
        if target_section is not None:
            # First check earlier sections (in reverse order for most recent)
            for section in reversed(structure['sections'][:target_section]):
                for comp in section['completions']:
                    if comp['key'] == dep:
                        dep_section = section['index']
                        break
                if dep_section is not None:
                    break

            # If not found in earlier sections, check same section (for internal refs)
            if dep_section is None:
                current_section = structure['sections'][target_section]
                for comp in current_section['completions']:
                    if comp['key'] == dep:
                        dep_section = target_section
                        break

        # Skip arrows from sections with [[!end]] to later sections
        # (those later sections never execute if [[!end]] triggers)
        if dep_section is not None and target_section is not None:
            dep_sec_data = structure['sections'][dep_section]
            if dep_sec_data.get('has_end_action') and dep_section < target_section:
                continue  # Skip this arrow

        # Create node IDs with section prefixes
        dep_id = f'S{dep_section}_{dep}' if dep_section is not None else dep
        target_id = f'S{target_section}_{target}' if target_section is not None else target

        lines.append(f'    {dep_id} --> {target_id}')

    lines.append('')

    # Add implicit sequential dependencies within sections
    lines.append('    %% Sequential dependencies within sections')
    for section in structure['sections']:
        section_idx = section['index']
        completions = section['completions']
        for i in range(len(completions) - 1):
            current = completions[i]['key']
            next_comp = completions[i + 1]['key']
            current_id = f'S{section_idx}_{current}'
            next_id = f'S{section_idx}_{next_comp}'
            lines.append(f'    {current_id} -.-> {next_id}')

    lines.append('')

    # Add styling
    lines.append('    %% Styling')
    lines.append('    classDef think fill:#90EE90')
    lines.append('    classDef respond fill:#87CEEB')
    lines.append('    classDef pick fill:#FFD700')
    lines.append('    classDef extract fill:#FFA07A')
    lines.append('    classDef action fill:#DDA0DD')
    lines.append('    classDef input fill:#FFB6C1')
    lines.append('    classDef default fill:#D3D3D3')
    lines.append('')

    # Classify nodes by type (using unique node IDs)
    by_type = {}
    for section in structure['sections']:
        section_idx = section['index']
        for comp in section['completions']:
            comp_type = comp['type']
            node_id = f'S{section_idx}_{comp["key"]}'
            by_type.setdefault(comp_type, []).append(node_id)

    # Add class assignments
    if external_inputs:
        lines.append(f'    class {",".join(sorted(external_inputs))} input')

    for comp_type, node_ids in by_type.items():
        if node_ids:
            lines.append(f'    class {",".join(node_ids)} {comp_type}')

    return '\n'.join(lines)


def generate_simple_slot_dag(structure: Dict) -> str:
    """Generate simplified Mermaid DAG with only slot names and arrows.

    Args:
        structure: Output from analyze_sections

    Returns:
        Mermaid diagram as string
    """
    lines = ['graph TD']

    # Track which variables are defined vs referenced
    defined_vars = structure['all_completions']
    referenced_vars = {dep for dep, _ in structure['all_dependencies']}

    # External inputs (referenced but not defined)
    external_inputs = referenced_vars - defined_vars

    # Add external inputs
    for var in sorted(external_inputs):
        lines.append(f'    {var}([{var}])')

    if external_inputs:
        lines.append('')

    # Add sections with simple slot nodes
    for section in structure['sections']:
        section_idx = section['index']
        section_name = section.get('name')
        # Use segment name if available, otherwise fall back to "Checkpoint N"
        section_label = section_name if section_name else f"Checkpoint {section_idx + 1}"
        lines.append(f'    subgraph S{section_idx}["{section_label}"]')

        for comp in section['completions']:
            key = comp['key']
            lines.append(f'        {key}["{key}"]')

        lines.append('    end')
        lines.append('')

    # Add explicit dependencies from {{variable}} references
    for dep, target in structure['all_dependencies']:
        lines.append(f'    {dep} --> {target}')

    return '\n'.join(lines)


def generate_section_dag(structure: Dict) -> str:
    """Generate simplified Mermaid DAG showing only section-level dependencies.

    Args:
        structure: Output from analyze_sections

    Returns:
        Mermaid diagram as string
    """
    lines = ['graph TD']

    # Build mapping of completion variable -> section index
    completion_to_section = {}
    for section in structure['sections']:
        section_idx = section['index']
        for comp in section['completions']:
            completion_to_section[comp['key']] = section_idx

    # Track which variables are defined vs referenced
    defined_vars = structure['all_completions']
    referenced_vars = {dep for dep, _ in structure['all_dependencies']}

    # External inputs (referenced but not defined)
    external_inputs = referenced_vars - defined_vars

    # Add external inputs as a single node if any exist
    if external_inputs:
        inputs_list = ', '.join(sorted(external_inputs))
        lines.append(f'    EXT["External: {inputs_list}"]')
        lines.append('')

    # Add section nodes
    for section in structure['sections']:
        section_idx = section['index']
        section_name = section.get('name')
        slot_names = [comp['key'] for comp in section['completions']]

        # Limit slot names to keep labels readable
        max_slots_to_show = 3
        if len(slot_names) <= max_slots_to_show:
            slots_str = ', '.join(slot_names)
        else:
            shown_slots = ', '.join(slot_names[:max_slots_to_show])
            slots_str = f'{shown_slots} +{len(slot_names) - max_slots_to_show} more'

        # Use segment name if available, otherwise fall back to "Checkpoint N"
        section_label = section_name if section_name else f"Checkpoint {section_idx + 1}"
        label = f'{section_label}<br/>{slots_str}'
        lines.append(f'    S{section_idx}["{label}"]')

    lines.append('')

    # Add styling for cleaner appearance
    lines.append('    %% Styling')
    lines.append('    classDef sectionNode fill:#e3f2fd,stroke:#2196f3,stroke-width:2px')
    lines.append('    classDef extNode fill:#fff3cd,stroke:#ffc107,stroke-width:2px')
    if external_inputs:
        lines.append('    class EXT extNode')
    for section in structure['sections']:
        lines.append(f'    class S{section["index"]} sectionNode')
    lines.append('')

    # Determine section-level dependencies
    section_dependencies = {}  # section_idx -> set of section indices it depends on
    for section in structure['sections']:
        section_idx = section['index']
        depends_on_sections = set()

        # Collect section-level dependencies (external to this section)
        section_deps = set()
        for comp in section['completions']:
            for dep in comp['depends_on']:
                # A dependency is "external" if it's not defined in this section
                if dep not in {c['key'] for c in section['completions']}:
                    section_deps.add(dep)

        # Map dependencies to their sections
        for dep in section_deps:
            if dep in completion_to_section:
                dep_section_idx = completion_to_section[dep]
                # Skip dependencies from sections with [[!end]] to later sections
                # (those later sections never execute if [[!end]] triggers)
                dep_section_data = structure['sections'][dep_section_idx]
                if dep_section_data.get('has_end_action') and dep_section_idx < section_idx:
                    continue  # Skip this dependency
                depends_on_sections.add(dep_section_idx)
            elif dep in external_inputs:
                # Track external input dependency
                depends_on_sections.add('EXT')

        section_dependencies[section_idx] = depends_on_sections

    # Helper function to compute transitive dependencies
    def get_transitive_dependencies(section_idx, deps_graph):
        """Get all sections that section_idx depends on (directly or transitively)."""
        visited = set()
        to_visit = set(deps_graph.get(section_idx, set()))
        # Exclude 'EXT' from transitive computation
        to_visit = {s for s in to_visit if s != 'EXT'}

        while to_visit:
            current = to_visit.pop()
            if current not in visited:
                visited.add(current)
                # Add dependencies of current section
                current_deps = deps_graph.get(current, set())
                to_visit.update(s for s in current_deps if s != 'EXT' and s not in visited)

        return visited

    # Add blocking dependencies (sections with blocking completions)
    # Only add if not already covered by variable dependencies (directly or transitively)
    for from_section, to_section in structure.get('blocking_dependencies', []):
        # Get all sections that to_section already depends on (transitively)
        all_transitive_deps = get_transitive_dependencies(to_section, section_dependencies)

        # Only add blocking dependency if to_section doesn't already depend on from_section
        if from_section not in all_transitive_deps:
            if to_section not in section_dependencies:
                section_dependencies[to_section] = set()
            section_dependencies[to_section].add(from_section)

    # Add dependency arrows
    for section_idx, deps in section_dependencies.items():
        # Separate external and section dependencies for proper sorting
        ext_deps = [d for d in deps if d == 'EXT']
        section_deps = [d for d in deps if d != 'EXT']

        # Add external dependencies first
        for dep in ext_deps:
            lines.append(f'    EXT --> S{section_idx}')

        # Add section dependencies in sorted order
        for dep in sorted(section_deps):
            lines.append(f'    S{dep} --> S{section_idx}')

    return '\n'.join(lines)


def generate_html(mermaid_code: str, title: str = "Struckdown Visualization") -> str:
    """Generate standalone HTML with Mermaid.js from CDN.

    Args:
        mermaid_code: Mermaid diagram code
        title: HTML page title

    Returns:
        Complete HTML document as string
    """
    return f'''<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>{title}</title>
    <script type="module">
import mermaid from 'https://cdn.jsdelivr.net/npm/mermaid@10/dist/mermaid.esm.min.mjs';
mermaid.initialize({{
    startOnLoad: true,
    theme: 'default',
    flowchart: {{
        useMaxWidth: false,
        htmlLabels: true,
        curve: 'basis',
        padding: 20,
        nodeSpacing: 50,
        rankSpacing: 120
    }},
    themeVariables: {{
        fontSize: '14px'
    }}
}});
    </script>
    <style>
        body {{
            font-family: system-ui, -apple-system, sans-serif;
            max-width: 1200px;
            margin: 40px auto;
            padding: 20px;
            background: #f5f5f5;
        }}

        h1 {{
            color: #2c3e50;
            border-bottom: 3px solid #3498db;
            padding-bottom: 15px;
            margin-bottom: 30px;
        }}

        .diagram-container {{
            background: white;
            padding: 30px;
            border-radius: 8px;
            box-shadow: 0 2px 4px rgba(0,0,0,0.1);
            display: flex;
            justify-content: center;
            min-height: 400px;
            overflow-x: auto;
        }}

        .mermaid {{
            max-width: 600px;
            min-width: 400px;
        }}

        .mermaid svg {{
            max-width: 100%;
            height: auto;
        }}
    </style>
</head>
<body>
    <h1>{title}</h1>
    <div class="diagram-container">
        <pre class="mermaid">
{mermaid_code}
        </pre>
    </div>
</body>
</html>'''


def summarize_completion(current_prompt_text: str, previous_prompt_text: str, completion_key: str) -> Optional[str]:
    """Generate a one-sentence summary of what a completion does.

    Args:
        current_prompt_text: The prompt text for this completion
        previous_prompt_text: Accumulated prompt text from earlier completions in this section
        completion_key: The name of the completion variable

    Returns:
        One-sentence summary or None if summarization fails
    """
    try:
        from struckdown import chatter

        # Escape both struckdown commands and template syntax
        escaped_current = _escape_for_summary(current_prompt_text)
        escaped_previous = _escape_for_summary(previous_prompt_text)

        # Build summary prompt with context (always include previous_instructions section)
        summary_prompt = f"""
<previous_instructions>

{escaped_previous}

</previous_instructions>

<instruction>

{escaped_current}

</instruction>


This is an instruction given to an LLM. We want to give a 1 line summary in a UI of the LAST instruction.
Summarise the instruction in ~12 words max. Use imperative form (i.e. as though you are telling someone what to do this instruction yourself). Be extremely concise:[[response]]"""

        result = chatter(summary_prompt)
        if result:
            return result['response'].strip()
        return None
    except Exception as e:
        return None


def build_execution_plan_data(structure: Dict, prompt_name: str = "Prompt", sections_data=None, summarize: bool = False) -> Dict:
    """Build execution plan data structure for template rendering.

    Args:
        structure: Output from analyze_sections
        prompt_name: Name of the prompt file
        sections_data: Original parsed sections data (for system prompt analysis)
        summarize: If True, generate LLM summaries for each completion

    Returns:
        Dictionary with execution plan data for template rendering
    """
    data = {
        'title': f'Execution Plan: {prompt_name}',
        'prompt_name': prompt_name,
        'total_completions': len(structure['all_completions']),
        'system_prompt': None,
        'external_inputs': [],
        'sections': []
    }

    # System prompt info (if sections_data provided)
    if sections_data and len(sections_data) > 0:
        first_section = sections_data[0]
        if first_section:
            first_comp = next(iter(first_section.values()))
            sys_msg = first_comp.system_message
            if sys_msg and sys_msg.strip():
                sys_len = len(sys_msg)
                sys_lines = sys_msg.count('\n') + 1
                preview = sys_msg.strip().split('\n')[0][:60]
                if len(preview) < len(sys_msg.strip().split('\n')[0]):
                    preview += "..."
                data['system_prompt'] = {
                    'length': sys_len,
                    'lines': sys_lines,
                    'preview': preview
                }

    # External inputs
    defined_vars = structure['all_completions']
    referenced_vars = {dep for dep, _ in structure['all_dependencies']}
    external_inputs = referenced_vars - defined_vars
    data['external_inputs'] = sorted(external_inputs)

    # Build mapping of completion variable -> section index
    completion_to_section = {}
    for section in structure['sections']:
        section_idx = section['index']
        for comp in section['completions']:
            completion_to_section[comp['key']] = section_idx

    # Build sections data
    for section in structure['sections']:
        section_idx = section['index']

        # Collect section-level dependencies (external to this section)
        section_deps = set()
        for comp in section['completions']:
            for dep in comp['depends_on']:
                if dep not in {c['key'] for c in section['completions']}:
                    section_deps.add(dep)

        # Determine which sections this section depends on
        depends_on_sections = set()
        for dep in section_deps:
            if dep in completion_to_section:
                depends_on_sections.add(completion_to_section[dep] + 1)  # +1 for human-readable

        # Build completions list
        completions_list = []
        accumulated_prompt = ""
        for comp in section['completions']:
            key = comp['key']
            comp_type = comp['type']
            line_num = comp.get('line_number', 0)

            completion_data = {
                'key': key,
                'type': comp_type,
                'line_number': line_num,
                'summary': None,
                'prompt_text': None
            }

            # Get prompt text from sections_data
            current_prompt_text = None
            if sections_data:
                for sect in sections_data:
                    if key in sect:
                        current_prompt_text = sect[key].text
                        break

            if current_prompt_text:
                completion_data['prompt_text'] = current_prompt_text

                # Generate summary if requested
                if summarize:
                    summary = summarize_completion(
                        current_prompt_text=current_prompt_text,
                        previous_prompt_text=accumulated_prompt,
                        completion_key=key
                    )
                    completion_data['summary'] = summary

                accumulated_prompt += current_prompt_text + "\n\n"

            completions_list.append(completion_data)

        section_data = {
            'index': section_idx,
            'name': section.get('name'),  # Include segment name if available
            'dependencies': sorted(section_deps),
            'depends_on_sections': sorted(depends_on_sections),
            'completions': completions_list
        }

        data['sections'].append(section_data)

    # Generate section-level DAG for visualization
    data['section_dag_mermaid'] = generate_section_dag(structure)

    return data


def render_execution_plan(data: Dict, format: str = 'text') -> str:
    """Render execution plan data using Jinja2 template.

    Args:
        data: Data dictionary from build_execution_plan_data
        format: Output format ('text' or 'html')

    Returns:
        Rendered template as string
    """
    from jinja2 import Environment, FileSystemLoader, select_autoescape
    from pathlib import Path

    # Set up Jinja2 environment
    templates_dir = Path(__file__).parent / "templates"
    env = Environment(
        loader=FileSystemLoader(templates_dir),
        autoescape=select_autoescape(['html']) if format == 'html' else False,
        trim_blocks=True,
        lstrip_blocks=True
    )

    # Add custom filters
    def number_format(value):
        """Format number with commas."""
        return f"{value:,}"

    def pluralize(count, singular, plural=None):
        """Return singular or plural form based on count."""
        if plural is None:
            plural = singular + 's'
        return singular if count == 1 else plural

    env.filters['number_format'] = number_format
    env.filters['pluralize'] = pluralize

    # Select template based on format
    if format == 'html':
        template = env.get_template('execution_plan.html')
    else:
        template = env.get_template('execution_plan.txt')

    return template.render(**data)


def markdown_to_html(markdown_text: str, title: str = "Execution Plan") -> str:
    """Convert markdown execution plan to simple HTML.

    Args:
        markdown_text: Markdown-formatted execution plan
        title: HTML page title

    Returns:
        Complete HTML document as string
    """
    import html

    lines = markdown_text.split('\n')
    html_lines = []
    in_list = False

    for line in lines:
        # Escape HTML entities
        line_escaped = html.escape(line)

        # Check for headers
        if line.startswith('# '):
            if in_list:
                html_lines.append('</ul>')
                in_list = False
            header_text = line_escaped[2:]
            html_lines.append(f'<h2>{header_text}</h2>')
        elif line.startswith('## '):
            if in_list:
                html_lines.append('</ul>')
                in_list = False
            header_text = line_escaped[3:]
            html_lines.append(f'<h3>{header_text}</h3>')
        # Check for horizontal rule
        elif line.startswith('===='):
            if in_list:
                html_lines.append('</ul>')
                in_list = False
            html_lines.append('<hr>')
        # Check for bullet points
        elif line.startswith('- '):
            if not in_list:
                html_lines.append('<ul>')
                in_list = True
            content = line_escaped[2:]
            # Highlight <TYPE> tags
            content = re.sub(r'&lt;([A-Z]+)&gt;', r'<code class="type">\1</code>', content)
            html_lines.append(f'<li>{content}</li>')
        # Check for bullet points with bullet character
        elif line.startswith('  • '):
            if not in_list:
                html_lines.append('<ul>')
                in_list = True
            content = line_escaped[4:]
            html_lines.append(f'<li>{content}</li>')
        # Empty line
        elif not line.strip():
            if in_list:
                html_lines.append('</ul>')
                in_list = False
            html_lines.append('<br>')
        # Regular text
        else:
            if in_list:
                html_lines.append('</ul>')
                in_list = False
            html_lines.append(f'<p>{line_escaped}</p>')

    # Close list if still open
    if in_list:
        html_lines.append('</ul>')

    body_content = '\n'.join(html_lines)

    return f'''<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>{html.escape(title)}</title>
    <style>
        body {{
            font-family: system-ui, -apple-system, sans-serif;
            max-width: 900px;
            margin: 40px auto;
            padding: 20px;
            line-height: 1.6;
            color: #333;
            background: #f5f5f5;
        }}
        .container {{
            background: white;
            padding: 40px;
            border-radius: 8px;
            box-shadow: 0 2px 4px rgba(0,0,0,0.1);
        }}
        h2 {{
            color: #2c3e50;
            margin-top: 30px;
            margin-bottom: 15px;
            padding-bottom: 10px;
            border-bottom: 2px solid #e0e0e0;
        }}
        h3 {{
            color: #34495e;
            margin-top: 20px;
            margin-bottom: 10px;
        }}
        hr {{
            border: none;
            border-top: 2px solid #e0e0e0;
            margin: 30px 0;
        }}
        ul {{
            list-style-type: disc;
            padding-left: 20px;
            margin: 10px 0;
        }}
        li {{
            margin: 5px 0;
        }}
        code.type {{
            background: #e8f5e9;
            color: #2e7d32;
            padding: 2px 6px;
            border-radius: 3px;
            font-family: 'Courier New', monospace;
            font-size: 0.9em;
            font-weight: bold;
        }}
        p {{
            margin: 8px 0;
        }}
        br {{
            display: block;
            content: "";
            margin: 0;
        }}
    </style>
</head>
<body>
    <div class="container">
{body_content}
    </div>
</body>
</html>'''
