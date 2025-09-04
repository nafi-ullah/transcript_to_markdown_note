#!/usr/bin/env python3
"""
Demo script that shows how the note generator would work without actually calling the OpenAI API.
This is useful for testing the transcript processing logic without incurring API costs.
"""

import os
from note_generator import TranscriptNoteGenerator

class DemoNoteGenerator(TranscriptNoteGenerator):
    """Demo version that doesn't require API key and generates mock notes."""
    
    def __init__(self):
        """Initialize without OpenAI client."""
        self.notes_dir = "notes"
        os.makedirs(self.notes_dir, exist_ok=True)
    
    def generate_notes_with_gpt(self, transcript_segment: str, module_name: str, chunk_info: str) -> str:
        """Generate demo notes without calling the API."""
        # Count words and lines in the segment
        word_count = len(transcript_segment.split())
        line_count = len(transcript_segment.split('\n'))
        
        # Extract first few concepts mentioned
        words = transcript_segment.split()
        concepts = [word.strip('.,!?') for word in words if len(word) > 5 and word.isalpha()][:5]
        
        demo_notes = f"""## Key Topics Covered

This section covers several important concepts related to **{module_name.lower()}**.

### Main Concepts
- **Topic Analysis**: This segment contains {word_count} words across {line_count} lines
- **Key Terms Mentioned**: {', '.join(concepts[:3]) if concepts else 'Various technical terms'}
- **Content Focus**: Educational material on machine learning and AI concepts

### Important Points
1. **Learning Objectives**: Understanding fundamental concepts
2. **Practical Applications**: Real-world implementation examples  
3. **Technical Details**: In-depth explanation of methodologies

### Summary
This section provides comprehensive coverage of the topic with practical examples and detailed explanations. The content is structured to build understanding progressively.

> **Note**: This is a demo note generated without AI processing. In actual usage, GPT would analyze the transcript content and generate detailed, contextual notes.

#### Raw Transcript Preview
```
{transcript_segment[:300]}...
```
"""
        return demo_notes

def main():
    """Run the demo generator."""
    print("🎬 Demo AI Note Generator (No API Required)")
    print("=" * 50)
    print("This demo shows how the system processes transcripts without calling OpenAI API")
    print()
    
    try:
        generator = DemoNoteGenerator()
        
        # Default file paths
        transcript_file = "script/demo_transcript.txt"
        modules_file = "modules.txt"
        
        # Check if files exist
        if not os.path.exists(transcript_file):
            print(f"❌ Demo transcript file not found: {transcript_file}")
            return
        
        if not os.path.exists(modules_file):
            print(f"❌ Modules file not found: {modules_file}")
            return
        
        print("📊 This will process:")
        modules = generator.parse_modules(modules_file)
        transcript_data = generator.parse_transcript(transcript_file)
        
        print(f"   📜 {len(transcript_data)} transcript segments")
        print(f"   📚 {len(modules)} modules")
        
        # Process just the first module for demo
        if modules:
            print(f"\n🎯 Processing first module only (demo mode)")
            first_module = modules[0]
            
            module_notes = generator.process_module(first_module, transcript_data)
            
            # Save demo notes
            filename = f"DEMO_{first_module['number']} {first_module['name']}.md"
            filepath = os.path.join(generator.notes_dir, filename)
            
            with open(filepath, 'w', encoding='utf-8') as f:
                f.write(module_notes)
            
            print(f"✅ Demo notes generated: {filename}")
            print(f"📁 Saved in: {os.path.abspath(filepath)}")
        
        print("\n" + "=" * 50)
        print("🎉 Demo completed!")
        print("💡 To run with actual AI generation, set up OpenAI API key and use: python note_generator.py")
        
    except Exception as e:
        print(f"❌ Demo error: {str(e)}")

if __name__ == "__main__":
    main()
