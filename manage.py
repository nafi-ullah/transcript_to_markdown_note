#!/usr/bin/env python3
"""
Project management utility for the AI Note Generator.
Provides easy access to all project functionality.
"""

import os
import sys
import subprocess
from pathlib import Path

def print_header():
    """Print the project header."""
    print("🤖 AI Note Generator - Project Manager")
    print("=" * 50)

def check_setup():
    """Check if the project is properly set up."""
    issues = []
    
    # Check if required files exist
    required_files = [
        "note_generator.py",
        "requirements.txt",
        "modules.txt"
    ]
    
    for file in required_files:
        if not os.path.exists(file):
            issues.append(f"Missing required file: {file}")
    
    # Check if script directory exists
    if not os.path.exists("script"):
        issues.append("Missing 'script' directory for transcript files")
    elif not any(f.endswith('.txt') for f in os.listdir("script")):
        issues.append("No transcript files (.txt) found in script directory")
    
    # Check if .env file exists
    if not os.path.exists(".env"):
        if os.path.exists(".env.template"):
            issues.append("Copy .env.template to .env and add your OpenAI API key")
        else:
            issues.append("Create .env file with your OpenAI API key")
    
    # Check virtual environment
    venv_python = ".venv/bin/python"
    if not os.path.exists(venv_python):
        issues.append("Virtual environment not found. Run: python -m venv .venv")
    
    return issues

def install_dependencies():
    """Install project dependencies."""
    print("📦 Installing dependencies...")
    try:
        subprocess.run([sys.executable, "-m", "pip", "install", "-r", "requirements.txt"], 
                      check=True)
        print("✅ Dependencies installed successfully!")
    except subprocess.CalledProcessError as e:
        print(f"❌ Failed to install dependencies: {e}")

def show_project_status():
    """Show current project status."""
    print("📊 Project Status:")
    print("-" * 30)
    
    # Check files
    files_status = {
        "note_generator.py": "✅" if os.path.exists("note_generator.py") else "❌",
        "modules.txt": "✅" if os.path.exists("modules.txt") else "❌",
        ".env": "✅" if os.path.exists(".env") else "❌",
        "script/": "✅" if os.path.exists("script") else "❌",
        "notes/": "✅" if os.path.exists("notes") else "📁 (will be created)"
    }
    
    for file, status in files_status.items():
        print(f"  {status} {file}")
    
    # Check transcript files
    if os.path.exists("script"):
        transcript_files = [f for f in os.listdir("script") if f.endswith('.txt')]
        print(f"\n📜 Transcript files found: {len(transcript_files)}")
        for file in transcript_files:
            print(f"     • {file}")
    
    # Check modules
    if os.path.exists("modules.txt"):
        with open("modules.txt", 'r') as f:
            module_lines = [line for line in f.readlines() if line.strip().startswith('⌨️')]
        print(f"\n📚 Modules found: {len(module_lines)}")
    
    # Check generated notes
    if os.path.exists("notes"):
        note_files = [f for f in os.listdir("notes") if f.endswith('.md')]
        print(f"\n📝 Generated notes: {len(note_files)}")
        if note_files:
            print("     Recent notes:")
            for file in sorted(note_files)[:5]:
                print(f"     • {file}")

def show_menu():
    """Show the main menu."""
    print("\n🎯 Available Actions:")
    print("1. 🧪 Test system components")
    print("2. 🎬 Run demo (no API key needed)")
    print("3. ⚙️  Interactive setup and run")
    print("4. 🚀 Generate notes (direct mode)")
    print("5. 📊 Show project status")
    print("6. 📦 Install dependencies")
    print("7. 📖 View README")
    print("8. 🔧 Check setup issues")
    print("9. 🚪 Exit")

def run_command(command_file):
    """Run a Python command file."""
    python_path = ".venv/bin/python" if os.path.exists(".venv/bin/python") else sys.executable
    try:
        subprocess.run([python_path, command_file], check=True)
    except subprocess.CalledProcessError as e:
        print(f"❌ Command failed: {e}")
    except KeyboardInterrupt:
        print("\n❌ Command interrupted by user")

def main():
    """Main menu loop."""
    print_header()
    
    while True:
        show_menu()
        
        try:
            choice = input("\nChoose an action (1-9): ").strip()
            
            if choice == "1":
                print("\n" + "="*30)
                run_command("test_generator.py")
                
            elif choice == "2":
                print("\n" + "="*30)
                run_command("demo_generator.py")
                
            elif choice == "3":
                print("\n" + "="*30)
                run_command("run_generator.py")
                
            elif choice == "4":
                print("\n" + "="*30)
                run_command("note_generator.py")
                
            elif choice == "5":
                print("\n" + "="*30)
                show_project_status()
                
            elif choice == "6":
                print("\n" + "="*30)
                install_dependencies()
                
            elif choice == "7":
                print("\n" + "="*30)
                if os.path.exists("README.md"):
                    with open("README.md", 'r') as f:
                        print(f.read())
                else:
                    print("❌ README.md not found")
                
            elif choice == "8":
                print("\n" + "="*30)
                issues = check_setup()
                if not issues:
                    print("✅ Setup looks good!")
                else:
                    print("⚠️  Setup issues found:")
                    for issue in issues:
                        print(f"   • {issue}")
                
            elif choice == "9":
                print("\n👋 Goodbye!")
                break
                
            else:
                print("❌ Invalid choice. Please try again.")
                
        except KeyboardInterrupt:
            print("\n\n👋 Goodbye!")
            break
        except Exception as e:
            print(f"❌ Error: {e}")
        
        input("\nPress Enter to continue...")
        print("\n" + "="*50)

if __name__ == "__main__":
    main()
