#!/usr/bin/env python3
"""
Configuration and utility script for the AI Note Generator.
This script allows you to customize settings and run the generator with different configurations.
"""

import os
import sys
from note_generator import TranscriptNoteGenerator

class Config:
    """Configuration settings for the note generator."""
    
    # File paths
    DEFAULT_TRANSCRIPT_FILE = "script/demo_transcript.txt"
    DEFAULT_MODULES_FILE = "modules.txt"
    DEFAULT_NOTES_DIR = "notes"
    
    # AI settings
    CHUNK_DURATION_MINUTES = 10
    GPT_MODEL = "gpt-3.5-turbo"
    MAX_TOKENS = 2000
    TEMPERATURE = 0.3
    
    # Rate limiting
    API_DELAY_SECONDS = 1
    
    @classmethod
    def get_available_transcripts(cls):
        """Get list of available transcript files."""
        script_dir = "script"
        if not os.path.exists(script_dir):
            return []
        
        transcripts = []
        for file in os.listdir(script_dir):
            if file.endswith('.txt'):
                transcripts.append(os.path.join(script_dir, file))
        return transcripts
    
    @classmethod
    def get_available_modules(cls):
        """Get list of available module files."""
        module_files = []
        for file in os.listdir('.'):
            if file.endswith('.txt') and ('module' in file.lower() or file == 'modules.txt'):
                module_files.append(file)
        return module_files

def interactive_setup():
    """Interactive setup for choosing files and settings."""
    print("🎯 AI Note Generator - Interactive Setup")
    print("=" * 50)
    
    # Choose transcript file
    transcripts = Config.get_available_transcripts()
    if not transcripts:
        print("❌ No transcript files found in script/ directory")
        return None, None
    
    print("📜 Available transcript files:")
    for i, transcript in enumerate(transcripts, 1):
        print(f"  {i}. {transcript}")
    
    while True:
        try:
            choice = input(f"Choose transcript file (1-{len(transcripts)}) or press Enter for default: ").strip()
            if not choice:
                transcript_file = Config.DEFAULT_TRANSCRIPT_FILE
                break
            else:
                transcript_file = transcripts[int(choice) - 1]
                break
        except (ValueError, IndexError):
            print("Invalid choice. Please try again.")
    
    # Choose modules file
    modules = Config.get_available_modules()
    if not modules:
        print("❌ No module files found")
        return None, None
    
    print("\n📋 Available module files:")
    for i, module_file in enumerate(modules, 1):
        print(f"  {i}. {module_file}")
    
    while True:
        try:
            choice = input(f"Choose modules file (1-{len(modules)}) or press Enter for default: ").strip()
            if not choice:
                modules_file = Config.DEFAULT_MODULES_FILE
                break
            else:
                modules_file = modules[int(choice) - 1]
                break
        except (ValueError, IndexError):
            print("Invalid choice. Please try again.")
    
    return transcript_file, modules_file

def verify_api_key():
    """Verify that OpenAI API key is set up."""
    api_key = os.getenv('OPENAI_API_KEY')
    if not api_key or api_key == 'your_openai_api_key_here' or api_key == 'sk-your-openai-api-key-here':
        print("❌ OpenAI API key not properly configured!")
        print("\nTo set up your API key:")
        print("1. Copy .env.template to .env")
        print("2. Edit .env and add your actual OpenAI API key")
        print("3. Get your API key from: https://platform.openai.com/api-keys")
        return False
    return True

def estimate_cost(modules, transcript_data):
    """Estimate the cost of running the note generator."""
    total_chunks = 0
    for module in modules:
        if module['end_time']:
            duration = module['end_time'] - module['start_time']
        else:
            # Assume 1 hour for modules without end time
            duration = 3600
        
        chunks = max(1, duration // (Config.CHUNK_DURATION_MINUTES * 60))
        total_chunks += chunks
    
    # Rough estimate: each chunk uses ~1000 tokens (input + output)
    estimated_tokens = total_chunks * 1000
    estimated_cost = estimated_tokens * 0.002 / 1000  # GPT-3.5-turbo pricing
    
    print(f"📊 Cost Estimation:")
    print(f"  Total modules: {len(modules)}")
    print(f"  Estimated chunks: {total_chunks}")
    print(f"  Estimated tokens: {estimated_tokens:,}")
    print(f"  Estimated cost: ${estimated_cost:.3f}")

def main():
    """Main function with interactive setup."""
    try:
        # Check for command line arguments
        if len(sys.argv) > 1:
            if sys.argv[1] in ['-h', '--help']:
                print("AI Note Generator Usage:")
                print("  python run_generator.py          # Interactive mode")
                print("  python note_generator.py         # Direct mode with default files")
                print("  python test_generator.py         # Test functionality")
                return
        
        # Verify API key
        if not verify_api_key():
            return
        
        # Interactive setup
        transcript_file, modules_file = interactive_setup()
        if not transcript_file or not modules_file:
            print("❌ Setup cancelled")
            return
        
        # Verify files exist
        if not os.path.exists(transcript_file):
            print(f"❌ Transcript file not found: {transcript_file}")
            return
        
        if not os.path.exists(modules_file):
            print(f"❌ Modules file not found: {modules_file}")
            return
        
        print(f"\n✅ Selected files:")
        print(f"   Transcript: {transcript_file}")
        print(f"   Modules: {modules_file}")
        
        # Create generator and parse files for estimation
        generator = TranscriptNoteGenerator()
        modules = generator.parse_modules(modules_file)
        transcript_data = generator.parse_transcript(transcript_file)
        
        # Show cost estimation
        print()
        estimate_cost(modules, transcript_data)
        
        # Confirm before proceeding
        confirm = input(f"\n🚀 Proceed with note generation? (y/N): ").strip().lower()
        if confirm not in ['y', 'yes']:
            print("Operation cancelled")
            return
        
        # Generate notes
        generator.generate_all_notes(transcript_file, modules_file)
        
    except KeyboardInterrupt:
        print("\n❌ Operation cancelled by user")
    except Exception as e:
        print(f"❌ Error: {str(e)}")

if __name__ == "__main__":
    main()
