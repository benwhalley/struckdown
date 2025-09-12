#!/usr/bin/env python3
"""
Test script that loops through all example .sd files and executes them.
This validates that all example syntax is correct and demonstrates the library features.
"""

import os
import glob
from pathlib import Path
from struckdown.parsing import parse_syntax, extract_all_placeholders
from struckdown import chatter


def test_all_examples():
    """Test all .sd example files in the examples/ directory"""
    examples_dir = Path(__file__).parent / "examples"
    sd_files = sorted(glob.glob(str(examples_dir / "*.sd")))
    
    print(f"Found {len(sd_files)} example files to test:\n")
    
    for sd_file in sd_files:
        filename = os.path.basename(sd_file)
        print(f"=" * 60)
        print(f"Testing: {filename}")
        print(f"=" * 60)
        
        try:
            # read the file
            with open(sd_file, 'r') as f:
                template_content = f.read()
            
            print(f"Template content:\n{template_content}\n")
            
            # test parsing
            sections = parse_syntax(template_content)
            print(f"✓ Parsing successful - {len(sections)} sections found")
            
            # show parsed structure  
            for i, section in enumerate(sections):
                print(f"  Section {i+1}:")
                for key, part in section.items():
                    print(f"    {key} ({part.return_type.__name__}): {repr(part.text[:100])}...")
            
            # extract placeholders that need user input
            placeholders = extract_all_placeholders(template_content)
            print(f"✓ Template variables required: {placeholders}")
            
            # create sample context for testing
            sample_context = create_sample_context(placeholders)
            if sample_context:
                print(f"  Using sample context: {sample_context}")
            
            # try to execute with minimal LLM calls (using mock or very small max_tokens)
            print(f"✓ Attempting execution...")
            try:
                result = chatter(
                    template_content, 
                    context=sample_context,
                    extra_kwargs={"max_tokens": 20}  # very small to minimize cost
                )
                print(f"✓ Execution successful - {len(result.results)} completions")
                for key, completion in result.results.items():
                    print(f"  {key}: {repr(str(completion.output)[:50])}...")
            except Exception as e:
                print(f"⚠ Execution failed (expected for some examples): {e}")
            
        except Exception as e:
            print(f"✗ Error testing {filename}: {e}")
            print(f"  This indicates a syntax error in the example")
            continue
        
        print()


def create_sample_context(placeholders):
    """Create sample context values for template variables"""
    context = {}
    
    # common sample values
    samples = {
        'topic': 'artificial intelligence',
        'style': 'haiku',
        'language': 'French',
        'domain': 'computer science',
        'count': 3,
        'item_type': 'book',
        'dataset_description': 'Customer purchase data from 2020-2024',
        'weird_input': 'Hello "world" & <test>',
        'ambiguous_prompt': 'make it better',
        'dataset': 'iris.csv'
    }
    
    for placeholder in placeholders:
        if placeholder in samples:
            context[placeholder] = samples[placeholder]
        else:
            # generic fallback
            context[placeholder] = f"sample_{placeholder}"
    
    return context


def test_shared_header_feature():
    """Specifically test the new shared header feature"""
    print(f"=" * 60)
    print("Testing Shared Header Feature")
    print(f"=" * 60)
    
    # test without shared header
    template1 = """Tell a joke: [[joke]]

¡OBLIVIATE

Rate the joke {{joke}} from 1-10: [[int:rating]]"""
    
    sections1 = parse_syntax(template1)
    print("Without shared header:")
    for i, section in enumerate(sections1):
        for key, part in section.items():
            print(f"  Section {i+1} - {key}: {repr(part.text)}")
    
    print()
    
    # test with shared header
    template2 = """You are a comedy expert who rates jokes professionally.

¡BEGIN

Tell a joke: [[joke]]

¡OBLIVIATE

Rate the joke {{joke}} from 1-10: [[int:rating]]"""
    
    sections2 = parse_syntax(template2)
    print("With shared header:")
    for i, section in enumerate(sections2):
        for key, part in section.items():
            print(f"  Section {i+1} - {key}: {repr(part.text)}")


if __name__ == "__main__":
    print("Struckdown Examples Test Suite")
    print("=" * 60)
    
    # test shared header feature specifically
    test_shared_header_feature()
    print()
    
    # test all examples
    test_all_examples()
    
    print("=" * 60)
    print("Test suite completed!")