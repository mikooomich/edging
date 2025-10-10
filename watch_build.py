#!/usr/bin/env python3
"""
Auto-build script using watchdog
Monitors .cpp and .h files and runs make on changes
"""

import os
import re
import subprocess
import sys
import time
from datetime import datetime
from watchdog.observers import Observer
from watchdog.events import FileSystemEventHandler

# ANSI color codes
RED = '\033[91m'
GREEN = '\033[92m'
YELLOW = '\033[93m'
RESET = '\033[0m'

class BuildHandler(FileSystemEventHandler):
    def __init__(self, build_dir, project_dir):
        self.build_dir = build_dir
        self.project_dir = project_dir
        self.last_build_time = 0
        self.debounce_seconds = 1  # Wait 1 second before rebuilding
        
    def on_modified(self, event):
        # Only trigger on .cpp and .h files
        if event.is_directory:
            return
        
        if not (event.src_path.endswith('.cpp') or event.src_path.endswith('.h')):
            return
            
        # Debounce: ignore if built recently
        current_time = time.time()
        if current_time - self.last_build_time < self.debounce_seconds:
            return
            
        self.last_build_time = current_time
        self.build()
    
    def parse_errors(self, output):
        """Parse compiler errors and format them for easy navigation"""
        errors = []
        
        # Pattern for gcc/g++ errors: file.cpp:line:col: error: message
        error_pattern = re.compile(r'(/[^\s:]+):(\d+):(\d+):\s*(error|warning):\s*(.+)')
        
        # Pattern for linker errors: undefined reference to 'function'
        linker_pattern = re.compile(r"undefined reference to [`']([^']+)'")
        
        lines = output.split('\n')
        for i, line in enumerate(lines):
            # Compiler errors
            match = error_pattern.search(line)
            if match:
                file_path = match.group(1)
                line_num = match.group(2)
                col_num = match.group(3)
                error_type = match.group(4)
                message = match.group(5)
                
                errors.append({
                    'type': error_type,
                    'file': file_path,
                    'line': line_num,
                    'col': col_num,
                    'message': message,
                    'full_line': line.strip()
                })
            
            # Linker errors
            linker_match = linker_pattern.search(line)
            if linker_match:
                function_name = linker_match.group(1)
                # Try to find the source file in previous lines
                file_info = None
                for j in range(max(0, i-10), i):
                    # Look for patterns like: CMakeFiles/main.dir/main.cpp.o
                    cpp_match = re.search(r'([a-zA-Z_][a-zA-Z0-9_/]*\.cpp)', lines[j])
                    if cpp_match:
                        # Extract just the filename
                        cpp_file = cpp_match.group(1).split('/')[-1]
                        # Construct full path (assuming it's in project directory)
                        file_info = os.path.join(self.project_dir, cpp_file)
                        if os.path.exists(file_info):
                            break
                
                if not file_info or not os.path.exists(file_info):
                    # Default to main.cpp if we can't find the file
                    file_info = os.path.join(self.project_dir, 'main.cpp')
                
                errors.append({
                    'type': 'linker error',
                    'file': file_info,
                    'line': '?',
                    'col': '?',
                    'message': f"undefined reference to '{function_name}'",
                    'full_line': line.strip()
                })
        
        return errors
    
    def print_error_summary(self, errors):
        """Print formatted error summary"""
        if not errors:
            return
        
        print(f"\n{YELLOW}{'='*60}{RESET}")
        print(f"{YELLOW}Error Summary:{RESET}")
        print(f"{YELLOW}{'='*60}{RESET}\n")
        
        for idx, error in enumerate(errors, 1):
            error_type = error['type'].upper()
            
            # Print error name (red)
            print(f"{RED}{error_type}: {error['message']}{RESET}")
            
            # Print clickable location - always use file:line:col format
            # If line/col are unknown, default to 1:1 (beginning of file)
            line_num = error['line'] if error['line'] != '?' else '1'
            col_num = error['col'] if error['col'] != '?' else '1'
            location = f"{error['file']}:{line_num}:{col_num}"
            print(f"  → {location}")
            print()
    
    def build(self):
        print(f"\n{'='*60}")
        print(f"[{datetime.now().strftime('%H:%M:%S')}] Building...")
        print('='*60)
        
        try:
            result = subprocess.run(
                ['make'],
                cwd=self.build_dir,
                capture_output=True,
                text=True,
                timeout=30
            )
            
            # Print full output
            if result.stdout:
                print(result.stdout)
            if result.stderr:
                print(result.stderr, file=sys.stderr)
            
            # Parse and print error summary
            if result.returncode != 0:
                combined_output = result.stdout + result.stderr
                errors = self.parse_errors(combined_output)
                self.print_error_summary(errors)
                print(f"\n{RED}✗ Build failed with exit code {result.returncode}{RESET}")
            else:
                print(f"\n{GREEN}✓ Build successful!{RESET}")
                
        except subprocess.TimeoutExpired:
            print(f"{RED}✗ Build timeout (>30s){RESET}")
        except Exception as e:
            print(f"{RED}✗ Build error: {e}{RESET}")

def main():
    project_dir = os.path.dirname(os.path.abspath(__file__))
    build_dir = os.path.join(project_dir, 'build')
    
    if not os.path.exists(build_dir):
        print(f"Error: Build directory not found: {build_dir}")
        sys.exit(1)
    
    print(f"Watching {project_dir} for changes...")
    print(f"Build directory: {build_dir}")
    print("Press Ctrl+C to stop\n")
    
    # Initial build
    handler = BuildHandler(build_dir, project_dir)
    handler.build()
    
    # Start watching
    observer = Observer()
    observer.schedule(handler, project_dir, recursive=True)
    observer.start()
    
    try:
        while True:
            time.sleep(1)
    except KeyboardInterrupt:
        observer.stop()
        print("\n\nStopped watching.")
    
    observer.join()

if __name__ == "__main__":
    main()

