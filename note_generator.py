#!/usr/bin/env python3
"""
AI Note Generator for Video Transcripts
This script processes video transcripts and generates markdown notes for each module using OpenAI GPT API.
"""

import os
import re
import time
from datetime import datetime, timedelta
from typing import List, Tuple, Dict
from dotenv import load_dotenv
from openai import OpenAI

# Load environment variables
load_dotenv()

class TranscriptNoteGenerator:
    """
    A class to generate structured notes from video transcripts using AI.
    """
    
    def __init__(self):
        """Initialize the note generator with OpenAI client."""
        api_key = os.getenv('OPENAI_API_KEY')
        if not api_key:
            raise ValueError("OPENAI_API_KEY not found in environment variables. Please set it in .env file.")
        
        self.client = OpenAI(api_key=api_key)
        self.notes_dir = "notes"
        
        # Ensure notes directory exists
        os.makedirs(self.notes_dir, exist_ok=True)
    
    def parse_timestamp(self, timestamp_str: str) -> int:
        """
        Parse timestamp string to seconds.
        Supports formats: MM:SS, HH:MM:SS
        """
        timestamp_str = timestamp_str.strip()
        parts = timestamp_str.split(':')
        
        if len(parts) == 2:  # MM:SS
            minutes, seconds = map(int, parts)
            return minutes * 60 + seconds
        elif len(parts) == 3:  # HH:MM:SS
            hours, minutes, seconds = map(int, parts)
            return hours * 3600 + minutes * 60 + seconds
        else:
            raise ValueError(f"Invalid timestamp format: {timestamp_str}")
    
    def seconds_to_timestamp(self, seconds: int) -> str:
        """Convert seconds to timestamp string HH:MM:SS or MM:SS format."""
        hours = seconds // 3600
        minutes = (seconds % 3600) // 60
        secs = seconds % 60
        
        if hours > 0:
            return f"{hours}:{minutes:02d}:{secs:02d}"
        else:
            return f"{minutes}:{secs:02d}"
    
    def parse_modules(self, modules_file: str) -> List[Dict]:
        """
        Parse modules file to extract module information.
        Returns list of modules with start_time, end_time, and name.
        """
        modules = []
        
        with open(modules_file, 'r', encoding='utf-8') as f:
            lines = f.readlines()
        
        for i, line in enumerate(lines):
            line = line.strip()
            if not line or not line.startswith('⌨️'):
                continue
            
            # Extract timestamp and module name using regex
            match = re.search(r'\((\d{2}:\d{2}:\d{2})\)\s*Module\s*(\d+):\s*(.+)', line)
            if match:
                timestamp_str = match.group(1)
                module_num = int(match.group(2))
                module_name = match.group(3).strip()
                
                start_seconds = self.parse_timestamp(timestamp_str)
                
                # Calculate end time (start of next module or end of transcript)
                end_seconds = None
                for j in range(i + 1, len(lines)):
                    next_line = lines[j].strip()
                    if next_line.startswith('⌨️'):
                        next_match = re.search(r'\((\d{2}:\d{2}:\d{2})\)', next_line)
                        if next_match:
                            end_seconds = self.parse_timestamp(next_match.group(1))
                            break
                
                modules.append({
                    'number': module_num,
                    'name': module_name,
                    'start_time': start_seconds,
                    'end_time': end_seconds,
                    'start_timestamp': timestamp_str
                })
        
        return modules
    
    def parse_transcript(self, transcript_file: str) -> Dict[int, str]:
        """
        Parse transcript file and return a dictionary mapping seconds to transcript text.
        """
        transcript_data = {}
        current_time = None
        current_text = []
        
        with open(transcript_file, 'r', encoding='utf-8') as f:
            lines = f.readlines()
        
        for line in lines:
            line = line.strip()
            if not line:
                continue
            
            # Check if line is a timestamp (format: M:SS or MM:SS or H:MM:SS)
            if re.match(r'^\d+:\d{2}$', line) or re.match(r'^\d{2}:\d{2}:\d{2}$', line):
                # Save previous timestamp data
                if current_time is not None and current_text:
                    transcript_data[current_time] = ' '.join(current_text)
                
                # Start new timestamp
                current_time = self.parse_timestamp(line)
                current_text = []
            else:
                # Add text to current timestamp
                if current_time is not None:
                    current_text.append(line)
        
        # Save the last timestamp data
        if current_time is not None and current_text:
            transcript_data[current_time] = ' '.join(current_text)
        
        return transcript_data
    
    def get_transcript_segment(self, transcript_data: Dict[int, str], start_seconds: int, end_seconds: int) -> str:
        """
        Extract transcript text for a specific time segment.
        """
        segment_text = []
        
        # Find all timestamps within the range
        for timestamp, text in transcript_data.items():
            if start_seconds <= timestamp < end_seconds:
                segment_text.append(f"[{self.seconds_to_timestamp(timestamp)}] {text}")
        
        return '\n'.join(segment_text)
    
    def find_nearest_timestamp(self, transcript_data: Dict[int, str], target_seconds: int) -> int:
        """
        Find the nearest available timestamp in the transcript data.
        """
        available_timestamps = list(transcript_data.keys())
        if not available_timestamps:
            return target_seconds
        
        # Find the closest timestamp
        closest_timestamp = min(available_timestamps, key=lambda x: abs(x - target_seconds))
        return closest_timestamp
    
    def generate_notes_with_gpt(self, transcript_segment: str, module_name: str, chunk_info: str) -> str:
        """
        Generate notes for a transcript segment using OpenAI GPT.
        """
        prompt = f"""
You are an expert note-taker for educational content. Please create comprehensive, well-structured markdown notes for the following transcript segment from the module "{module_name}".

{chunk_info}

Transcript:
{transcript_segment}

Please create notes that:
1. Use proper markdown formatting with headers (##, ###)
2. Extract key concepts and definitions
3. Organize information logically
4. Include important details and examples mentioned
5. Use bullet points and numbered lists where appropriate
6. Highlight important terms with **bold** or *italics*
7. Create a clear, study-friendly format

Focus on educational value and clarity. Make the notes comprehensive but concise.
"""

        try:
            response = self.client.chat.completions.create(
                model="gpt-3.5-turbo",
                messages=[
                    {"role": "system", "content": "You are an expert educational note-taker who creates comprehensive, well-structured markdown notes from lecture transcripts."},
                    {"role": "user", "content": prompt}
                ],
                max_tokens=2000,
                temperature=0.3
            )
            
            return response.choices[0].message.content.strip()
        
        except Exception as e:
            print(f"Error calling OpenAI API: {e}")
            return f"# Notes for {chunk_info}\n\n*Error generating notes: {str(e)}*\n\n## Raw Transcript\n{transcript_segment}"
    
    def process_module(self, module: Dict, transcript_data: Dict[int, str]) -> str:
        """
        Process a single module by breaking it into 10-minute chunks and generating notes.
        """
        module_name = module['name']
        start_time = module['start_time']
        end_time = module['end_time']
        
        print(f"\nProcessing Module {module['number']}: {module_name}")
        print(f"Time range: {self.seconds_to_timestamp(start_time)} to {self.seconds_to_timestamp(end_time) if end_time else 'END'}")
        
        # If no end time specified, use a reasonable default (e.g., 1 hour after start)
        if end_time is None:
            # Find the maximum timestamp in transcript or use start + 1 hour
            max_transcript_time = max(transcript_data.keys()) if transcript_data else start_time + 3600
            end_time = max_transcript_time
        
        all_notes = []
        chunk_duration = 600  # 10 minutes in seconds
        
        # Add module header
        all_notes.append(f"# Module {module['number']}: {module_name}\n")
        
        current_start = start_time
        chunk_number = 1
        
        while current_start < end_time:
            current_end = min(current_start + chunk_duration, end_time)
            
            # Find nearest available timestamps
            actual_start = self.find_nearest_timestamp(transcript_data, current_start)
            actual_end = self.find_nearest_timestamp(transcript_data, current_end)
            
            # Get transcript segment
            segment = self.get_transcript_segment(transcript_data, actual_start, actual_end)
            
            if segment.strip():
                chunk_info = f"Chunk {chunk_number} ({self.seconds_to_timestamp(current_start)} - {self.seconds_to_timestamp(current_end)})"
                print(f"  Processing {chunk_info}...")
                
                # Generate notes for this chunk
                notes = self.generate_notes_with_gpt(segment, module_name, chunk_info)
                all_notes.append(f"\n## {chunk_info}\n\n{notes}\n")
                
                # Add a small delay to avoid rate limiting
                time.sleep(1)
            
            current_start = current_end
            chunk_number += 1
        
        return '\n'.join(all_notes)
    
    def generate_all_notes(self, transcript_file: str, modules_file: str) -> None:
        """
        Main method to generate notes for all modules.
        """
        print("🤖 AI Note Generator Starting...")
        print("=" * 50)
        
        # Parse input files
        print("📖 Parsing transcript file...")
        transcript_data = self.parse_transcript(transcript_file)
        print(f"Found {len(transcript_data)} transcript segments")
        
        print("📋 Parsing modules file...")
        modules = self.parse_modules(modules_file)
        print(f"Found {len(modules)} modules")
        
        # Process each module
        for module in modules:
            try:
                # Generate notes for the module
                module_notes = self.process_module(module, transcript_data)
                
                # Save to markdown file
                filename = f"{module['number']} {module['name']}.md"
                filepath = os.path.join(self.notes_dir, filename)
                
                with open(filepath, 'w', encoding='utf-8') as f:
                    f.write(module_notes)
                
                print(f"✅ Generated notes for Module {module['number']}: {filename}")
                
            except Exception as e:
                print(f"❌ Error processing Module {module['number']}: {str(e)}")
        
        print("\n" + "=" * 50)
        print("🎉 Note generation completed!")
        print(f"📁 Notes saved in: {os.path.abspath(self.notes_dir)}")


def main():
    """Main function to run the note generator."""
    try:
        generator = TranscriptNoteGenerator()
        
        # Default file paths
        transcript_file = "script/demo_transcript.txt"
        modules_file = "demo_module.txt"
        
        # Check if files exist
        if not os.path.exists(transcript_file):
            print(f"❌ Transcript file not found: {transcript_file}")
            return
        
        if not os.path.exists(modules_file):
            print(f"❌ Modules file not found: {modules_file}")
            return
        
        # Generate notes
        generator.generate_all_notes(transcript_file, modules_file)
        
    except Exception as e:
        print(f"❌ Fatal error: {str(e)}")
        return 1

if __name__ == "__main__":
    main()
