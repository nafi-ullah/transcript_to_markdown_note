#!/usr/bin/env python3
"""
Final verification script to ensure the AI Note Generator project is complete and working.
"""

import os
import sys

def check_project_completeness():
    """Verify all project components are present and working."""
    
    print("🔍 AI Note Generator - Final Verification")
    print("=" * 50)
    
    # Required files
    required_files = {
        "Core Scripts": [
            "note_generator.py",
            "demo_generator.py", 
            "run_generator.py",
            "test_generator.py",
            "manage.py"
        ],
        "Configuration": [
            "requirements.txt",
            ".env.template",
            "README.md",
            "PROJECT_SUMMARY.md"
        ],
        "Data Files": [
            "modules.txt",
            "script/demo_transcript.txt"
        ]
    }
    
    all_good = True
    
    for category, files in required_files.items():
        print(f"\n📂 {category}:")
        for file in files:
            if os.path.exists(file):
                size = os.path.getsize(file)
                print(f"   ✅ {file} ({size:,} bytes)")
            else:
                print(f"   ❌ {file} - MISSING!")
                all_good = False
    
    # Check directories
    print(f"\n📁 Directories:")
    dirs_to_check = ["script", "notes", ".venv"]
    for dir_name in dirs_to_check:
        if os.path.exists(dir_name):
            if dir_name == "notes":
                files_count = len([f for f in os.listdir(dir_name) if f.endswith('.md')])
                print(f"   ✅ {dir_name}/ ({files_count} markdown files)")
            else:
                print(f"   ✅ {dir_name}/")
        else:
            if dir_name == ".venv":
                print(f"   ⚠️  {dir_name}/ - Virtual environment not detected")
            else:
                print(f"   ❌ {dir_name}/ - MISSING!")
                all_good = False
    
    # Test imports
    print(f"\n🐍 Python Dependencies:")
    required_packages = ["openai", "dotenv", "re"]
    
    for package in required_packages:
        try:
            if package == "dotenv":
                __import__("dotenv")
            elif package == "re":
                import re
            else:
                __import__(package)
            print(f"   ✅ {package}")
        except ImportError:
            print(f"   ❌ {package} - Not installed!")
            all_good = False
    
    # Functionality test
    print(f"\n⚡ Quick Functionality Test:")
    try:
        # Test note_generator imports
        sys.path.insert(0, '.')
        from note_generator import TranscriptNoteGenerator
        
        # Test basic functionality
        generator = TranscriptNoteGenerator.__new__(TranscriptNoteGenerator)
        generator.notes_dir = "notes"
        
        # Test timestamp parsing
        test_time = generator.parse_timestamp("1:30")
        if test_time == 90:
            print("   ✅ Timestamp parsing")
        else:
            print("   ❌ Timestamp parsing failed")
            all_good = False
            
        # Test modules parsing
        if os.path.exists("modules.txt"):
            modules = generator.parse_modules("modules.txt")
            if len(modules) == 8:
                print("   ✅ Module parsing (8 modules found)")
            else:
                print(f"   ⚠️  Module parsing ({len(modules)} modules found, expected 8)")
        else:
            print("   ❌ modules.txt not found")
            all_good = False
            
        # Test transcript parsing
        if os.path.exists("script/demo_transcript.txt"):
            transcript_data = generator.parse_transcript("script/demo_transcript.txt")
            if len(transcript_data) > 400:
                print(f"   ✅ Transcript parsing ({len(transcript_data)} segments)")
            else:
                print(f"   ⚠️  Transcript parsing ({len(transcript_data)} segments, seems low)")
        else:
            print("   ❌ demo_transcript.txt not found")
            all_good = False
            
    except Exception as e:
        print(f"   ❌ Functionality test failed: {e}")
        all_good = False
    
    # Final assessment
    print(f"\n{'=' * 50}")
    if all_good:
        print("🎉 PROJECT VERIFICATION COMPLETE!")
        print("✅ All components are present and working correctly")
        print("\n🚀 Ready to use! Try these commands:")
        print("   • python manage.py           (Interactive menu)")
        print("   • python demo_generator.py   (Demo without API)")
        print("   • python test_generator.py   (Run all tests)")
        print("   • python note_generator.py   (Generate notes)")
    else:
        print("⚠️  PROJECT VERIFICATION ISSUES FOUND")
        print("Some components are missing or not working correctly.")
        print("Please check the issues listed above.")
    
    return all_good

def show_usage_examples():
    """Show practical usage examples."""
    print(f"\n📚 Usage Examples:")
    print("=" * 30)
    
    examples = [
        ("🎬 Demo Mode", "python demo_generator.py", "Test without API key"),
        ("🧪 Run Tests", "python test_generator.py", "Validate all components"), 
        ("🎯 Interactive", "python run_generator.py", "Guided setup and execution"),
        ("⚡ Direct Run", "python note_generator.py", "Process with default files"),
        ("🎛️  Management", "python manage.py", "Full project management interface")
    ]
    
    for name, command, description in examples:
        print(f"{name}")
        print(f"   Command: {command}")
        print(f"   Purpose: {description}")
        print()

def main():
    """Run the verification."""
    success = check_project_completeness()
    
    if success:
        show_usage_examples()
    
    return 0 if success else 1

if __name__ == "__main__":
    sys.exit(main())
