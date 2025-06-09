#!/usr/bin/env python3
"""
every_other_token.py - Real-time LLM token interceptor

A research tool that intercepts every other token from OpenAI's streaming API
and applies transformations. Perfect for studying token dependencies and 
building novel LLM interaction patterns.

Usage: python every_other_token.py "your prompt here"
"""

import openai
import os
import sys
import time
from typing import Iterator, Callable

# Set up OpenAI client
client = openai.OpenAI(api_key=os.getenv("OPENAI_API_KEY"))

def reverse_transform(token: str, index: int) -> str:
    """Reverse every odd token"""
    if index % 2 == 0:
        return token  # Even: normal
    else:
        return token[::-1]  # Odd: reversed

def uppercase_transform(token: str, index: int) -> str:
    """Uppercase every odd token"""
    if index % 2 == 0:
        return token  # Even: normal
    else:
        return token.upper()  # Odd: uppercase

def mock_transform(token: str, index: int) -> str:
    """Mock every odd token (alternating case)"""
    if index % 2 == 0:
        return token  # Even: normal
    else:
        # Create mocking text (alternating case)
        result = ""
        for i, char in enumerate(token):
            if i % 2 == 0:
                result += char.lower()
            else:
                result += char.upper()
        return result

def noise_transform(token: str, index: int) -> str:
    """Add random characters to odd tokens"""
    if index % 2 == 0:
        return token  # Even: normal
    else:
        import random
        noise_chars = "~!@#$%^&*"
        return token + random.choice(noise_chars)

# Available transformations
TRANSFORMS = {
    'reverse': reverse_transform,
    'uppercase': uppercase_transform, 
    'mock': mock_transform,
    'noise': noise_transform
}

def every_other_token(prompt: str, transform_type: str = 'reverse', model: str = 'gpt-3.5-turbo'):
    """
    Stream response from OpenAI and transform every other token
    
    Args:
        prompt: The input prompt
        transform_type: Type of transformation ('reverse', 'uppercase', 'mock', 'noise')
        model: OpenAI model to use
    """
    
    if transform_type not in TRANSFORMS:
        print(f"Unknown transform: {transform_type}. Available: {list(TRANSFORMS.keys())}")
        return
        
    transform_func = TRANSFORMS[transform_type]
    
    print(f"🧬 EVERY OTHER TOKEN INTERCEPTOR")
    print(f"Transform: {transform_type}")
    print(f"Model: {model}")
    print(f"Prompt: {prompt}")
    print(f"\n{'='*50}")
    print("Response (with transformations):\n")
    
    try:
        # Create streaming chat completion
        response = client.chat.completions.create(
            model=model,
            messages=[{"role": "user", "content": prompt}],
            stream=True,
            temperature=0.7
        )
        
        token_index = 0
        full_response = ""
        
        for chunk in response:
            if chunk.choices[0].delta.content is not None:
                original_token = chunk.choices[0].delta.content
                
                # Apply transformation
                transformed_token = transform_func(original_token, token_index)
                
                # Print the transformed token
                print(transformed_token, end='', flush=True)
                
                # Store for analysis
                full_response += transformed_token
                token_index += 1
                
                # Small delay to make the streaming visible
                time.sleep(0.05)
        
        print(f"\n\n{'='*50}")
        print(f"✅ Complete! Processed {token_index} tokens.")
        print(f"🔬 Transform applied to {token_index // 2} tokens.")
        
    except Exception as e:
        print(f"❌ Error: {e}")
        print("Make sure OPENAI_API_KEY is set in your environment.")

def main():
    """CLI interface"""
    if len(sys.argv) < 2:
        print("🧬 Every Other Token Interceptor")
        print("\nUsage:")
        print("  python every_other_token.py \"your prompt here\"")
        print("  python every_other_token.py \"your prompt here\" [transform] [model]")
        print(f"\nAvailable transforms: {list(TRANSFORMS.keys())}")
        print("Default model: gpt-3.5-turbo")
        print("\nExamples:")
        print("  python every_other_token.py \"Tell me a story\"")
        print("  python every_other_token.py \"Explain quantum physics\" uppercase")
        print("  python every_other_token.py \"Write a poem\" mock gpt-4")
        return
    
    prompt = sys.argv[1]
    transform_type = sys.argv[2] if len(sys.argv) > 2 else 'reverse'
    model = sys.argv[3] if len(sys.argv) > 3 else 'gpt-3.5-turbo'
    
    # Check for API key
    if not os.getenv("OPENAI_API_KEY"):
        print("❌ OPENAI_API_KEY not found in environment variables.")
        print("Set it with: export OPENAI_API_KEY='your-api-key-here'")
        return
    
    every_other_token(prompt, transform_type, model)

if __name__ == "__main__":
    main()
