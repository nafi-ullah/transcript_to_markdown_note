# 🤖 AI Note Generator

This project automatically generates structured markdown notes from video lecture transcripts using OpenAI's GPT API.

## 🌟 Features

- 📚 **Module-based Processing**: Automatically splits transcript into modules based on timestamps
- ⏱️ **10-minute Chunks**: Processes each module in 10-minute segments for optimal AI processing
- 🤖 **AI-Powered Notes**: Uses GPT-3.5 to generate comprehensive, well-structured markdown notes
- 📝 **Markdown Output**: Creates properly formatted markdown files for each module
- 🔄 **Automatic Concatenation**: Combines notes from multiple chunks within the same module
- 🎯 **Interactive Setup**: Easy-to-use interface for selecting files and configurations
- 🎬 **Demo Mode**: Test functionality without API costs
- 💰 **Cost Estimation**: Preview API costs before processing

## 🚀 Quick Start

### Option 1: Interactive Mode (Recommended)
```bash
python manage.py
```
This opens an interactive menu with all options including demo mode, testing, and guided setup.

### Option 2: Direct Run
```bash
python note_generator.py
```

### Option 3: Demo Mode (No API Key Required)
```bash
python demo_generator.py
```

## 📁 Project Structure

```
notemaking_transcript/
├── 📜 Script Files
│   ├── note_generator.py      # Main AI note generator
│   ├── demo_generator.py      # Demo mode (no API needed)
│   ├── run_generator.py       # Interactive setup
│   ├── test_generator.py      # Test components
│   └── manage.py              # Project management utility
├── 📊 Data Files
│   ├── modules.txt            # Module definitions with timestamps
│   └── script/
│       └── demo_transcript.txt # Sample transcript file
├── ⚙️ Configuration
│   ├── .env.template          # Environment variables template
│   ├── requirements.txt       # Python dependencies
│   └── README.md             # This file
└── 📝 Output
    └── notes/                 # Generated markdown notes
```

## ⚙️ Setup

### 1. Install Dependencies
```bash
pip install -r requirements.txt
```

### 2. Set up OpenAI API Key
```bash
# Copy the template
cp .env.template .env

# Edit .env and add your actual API key
OPENAI_API_KEY=sk-your-actual-api-key-here
```
Get your API key from: [OpenAI Platform](https://platform.openai.com/api-keys)

### 3. Prepare Your Files
- **Transcript**: Place your transcript file in `script/` directory
- **Modules**: Ensure your modules file is named `modules.txt` in root directory

### 4. Run the Generator
Choose one of the options from Quick Start above!

## 📚 Usage Examples

### Basic Usage
```bash
python note_generator.py
```

### Interactive Setup
```bash
python run_generator.py
```

### Test Everything
```bash
python test_generator.py
```

### Demo Mode
```bash
python demo_generator.py
```

## How It Works

1. **Parse Input Files**: Reads and parses the transcript and modules files
2. **Calculate Time Segments**: Determines the time range for each module
3. **Create 10-minute Chunks**: Breaks each module into 10-minute segments
4. **Generate Notes**: Sends each chunk to GPT for note generation
5. **Concatenate & Save**: Combines all chunks for each module into a single markdown file

## Output Structure

For each module, you'll get a markdown file named like:
- `1 Machine Learning Fundamentals.md`
- `2 Introduction to TensorFlow.md`
- etc.

Each file contains:
- Module header
- Section headers for each 10-minute chunk
- Comprehensive notes with proper markdown formatting
- Key concepts, definitions, and examples

## Configuration

The script automatically:
- Creates the `notes/` directory if it doesn't exist
- Handles various timestamp formats (MM:SS, HH:MM:SS)
- Finds the nearest available timestamps in the transcript
- Includes error handling and rate limiting for API calls

## Troubleshooting

- **API Key Error**: Make sure your OpenAI API key is correctly set in the `.env` file
- **File Not Found**: Check that your transcript and modules files are in the correct locations
- **Rate Limiting**: The script includes delays between API calls to avoid rate limits

## Example

With the provided demo files:
- Input: `demo_transcript.txt` (TensorFlow course transcript)
- Input: `modules.txt` (8 modules defined)
- Output: 8 markdown files with comprehensive notes for each module

## Requirements

- Python 3.7+
- OpenAI API key
- Internet connection for API calls
