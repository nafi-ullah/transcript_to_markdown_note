#!/usr/bin/env python3
"""
Test script to validate the note generator with sample data.
This script can be used to test the functionality before running on the full transcript.
"""

from note_generator import TranscriptNoteGenerator
import os

def test_timestamp_parsing():
    """Test timestamp parsing functionality."""
    generator = TranscriptNoteGenerator()
    
    test_cases = [
        ("0:05", 5),
        ("1:30", 90),
        ("10:00", 600),
        ("1:00:00", 3600),
        ("2:30:45", 9045)
    ]
    
    print("Testing timestamp parsing...")
    for timestamp_str, expected_seconds in test_cases:
        result = generator.parse_timestamp(timestamp_str)
        status = "✅" if result == expected_seconds else "❌"
        print(f"{status} {timestamp_str} -> {result} seconds (expected {expected_seconds})")

def test_modules_parsing():
    """Test modules file parsing."""
    generator = TranscriptNoteGenerator()
    
    if os.path.exists("modules.txt"):
        modules = generator.parse_modules("modules.txt")
        print(f"\nTesting modules parsing...")
        print(f"Found {len(modules)} modules:")
        
        for module in modules:
            print(f"  Module {module['number']}: {module['name']}")
            print(f"    Start: {module['start_timestamp']} ({module['start_time']} seconds)")
            if module['end_time']:
                print(f"    End: {generator.seconds_to_timestamp(module['end_time'])} ({module['end_time']} seconds)")
    else:
        print("❌ modules.txt not found")

def test_transcript_parsing():
    """Test transcript file parsing."""
    generator = TranscriptNoteGenerator()
    
    transcript_file = "script/demo_transcript.txt"
    if os.path.exists(transcript_file):
        transcript_data = generator.parse_transcript(transcript_file)
        print(f"\nTesting transcript parsing...")
        print(f"Found {len(transcript_data)} transcript segments")
        
        # Show first few entries
        print("First 5 transcript segments:")
        for i, (timestamp, text) in enumerate(list(transcript_data.items())[:5]):
            print(f"  [{generator.seconds_to_timestamp(timestamp)}] {text[:100]}...")
    else:
        print(f"❌ {transcript_file} not found")

def main():
    """Run all tests."""
    print("🧪 Testing AI Note Generator Components")
    print("=" * 50)
    
    try:
        test_timestamp_parsing()
        test_modules_parsing()
        test_transcript_parsing()
        
        print("\n" + "=" * 50)
        print("✅ All tests completed!")
        print("\nTo generate actual notes, run: python note_generator.py")
        
    except Exception as e:
        print(f"❌ Test failed: {str(e)}")

if __name__ == "__main__":
    main()
