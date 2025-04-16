import os
import time
import groq
from pathlib import Path
import logging
from ratelimit import limits, sleep_and_retry
from typing import Optional, Dict, List, Tuple, Any
import json
import re
import difflib
import ast
import tempfile
import shutil
import fnmatch

class CodebaseAssistant:
    def __init__(self, model_name: str = "deepseek-r1-distill-llama-70b"):
        self.api_key = os.getenv("GROQ_API_KEY")
        if not self.api_key:
            raise ValueError("GROQ_API_KEY environment variable not set")
            
        self.model_name = model_name
        self.client = groq.Client(api_key=self.api_key)
        self.codebase_context = {}
        self.project_summary = ""
        self.file_summaries = {}
        self.codebase_root = ""
        self.pending_changes = {}
        self.all_files_info = {}  # Store information about all files
        self.directory_structure = {}  # Store the directory structure
        self.current_directory = ""  # Current directory for navigation
        self.scan_depth = 0  # Track depth for nested scanning
        self.scanned_extensions = set()  # Track all extensions found during scanning
        
        logging.basicConfig(
            level=logging.INFO,
            format='%(asctime)s - %(levelname)s - %(message)s'
        )
        self.logger = logging.getLogger(__name__)
        
    def scan_codebase(self, root_dir: str, ignore_dirs: List[str] = None, max_depth: int = None) -> Dict:
        """
        Scan and analyze the entire codebase including all nested directories
        
        Args:
            root_dir: The root directory to start scanning from
            ignore_dirs: List of directory names to ignore
            max_depth: Maximum depth for nested directory scanning (None for unlimited)
            
        Returns:
            Dictionary with scan results
        """
        if ignore_dirs is None:
            ignore_dirs = ['.git', 'venv', 'env', '__pycache__', 'node_modules', '.vscode', '.idea']
        
        self.codebase_root = os.path.abspath(root_dir)
        self.current_directory = self.codebase_root
        self.logger.info(f"Scanning codebase in {self.codebase_root} (max depth: {'unlimited' if max_depth is None else max_depth})")
        file_contents = {}
        all_files = {}
        self.scanned_extensions = set()
        
        # Build directory structure
        self.directory_structure = self._build_directory_structure(self.codebase_root, ignore_dirs, max_depth)
        
        # Track scan progress
        total_dirs = sum(1 for _, dirs, _ in os.walk(self.codebase_root) 
                         if not any(d in dirs for d in ignore_dirs))
        processed_dirs = 0
        
        for root, dirs, files in os.walk(self.codebase_root):
            # Skip ignored directories
            dirs[:] = [d for d in dirs if d not in ignore_dirs]
            
            # Check depth limit if specified
            if max_depth is not None:
                current_depth = root.replace(self.codebase_root, '').count(os.sep)
                if current_depth > max_depth:
                    dirs[:] = []  # Don't go deeper
                    continue
            
            processed_dirs += 1
            if processed_dirs % 10 == 0 or processed_dirs == total_dirs:
                self.logger.info(f"Scanning directories... {processed_dirs}/{total_dirs}")
            
            for file in files:
                file_path = os.path.join(root, file)
                relative_path = os.path.relpath(file_path, self.codebase_root)
                file_extension = os.path.splitext(file)[1].lower()
                
                # Add to scanned extensions set
                if file_extension:
                    self.scanned_extensions.add(file_extension[1:])  # Remove the dot
                
                # Store basic info about ALL files for querying later
                try:
                    file_size = os.path.getsize(file_path)
                    all_files[relative_path] = {
                        'path': relative_path,
                        'extension': file_extension[1:] if file_extension else '',
                        'size': file_size,
                        'last_modified': os.path.getmtime(file_path)
                    }
                except Exception as e:
                    self.logger.error(f"Error getting file info for {file_path}: {str(e)}")
                
                # Load content for all text-based files
                # Expanded list of text-based file extensions
                if file_extension in [
                    '.py', '.js', '.ts', '.java', '.cpp', '.h', '.cs', '.go', '.rs', 
                    '.php', '.rb', '.swift', '.kt', '.c', '.jsx', '.tsx', '.html', 
                    '.css', '.scss', '.json', '.yml', '.yaml', '.md', '.txt',
                    '.xml', '.sql', '.sh', '.bat', '.ps1', '.r', '.dart', '.lua',
                    '.m', '.mm', '.pl', '.pm', '.conf', '.config', '.ini', '.toml',
                    '.vue', '.svelte', '.elm', '.clj', '.scala', '.groovy', '.ex', '.exs',
                    '.erl', '.hs', '.jl', '.tf', '.gitignore', '.env', '.gradle'
                ]:
                    try:
                        # Use a safeguard for file size
                        if file_size > 10 * 1024 * 1024:  # Skip files larger than 10MB
                            self.logger.warning(f"Skipping large file: {file_path} ({file_size/1024/1024:.2f} MB)")
                            continue
                            
                        with open(file_path, 'r', encoding='utf-8', errors='replace') as f:
                            content = f.read()
                            language = file_extension[1:]  # Remove the dot
                            file_contents[relative_path] = {
                                'content': content,
                                'language': language,
                                'size': len(content),
                                'lines': len(content.splitlines()),
                                'path': relative_path,
                                'last_modified': os.path.getmtime(file_path),
                                'depth': relative_path.count(os.sep)  # Track nesting depth
                            }
                    except Exception as e:
                        self.logger.error(f"Error reading {file_path}: {str(e)}")
        
        self.codebase_context = file_contents
        self.all_files_info = all_files
        
        # Calculate max observed depth
        max_observed_depth = max((info.get('depth', 0) for info in file_contents.values()), default=0)
        self.scan_depth = max_observed_depth
        
        # Generate project structure tree
        project_structure = self._generate_project_structure(self.codebase_root, ignore_dirs)
        
        # Generate project summary after scanning
        self._generate_project_summary()
        
        # Generate file type statistics
        file_type_stats = self._generate_file_type_statistics()
        
        return {
            "files_analyzed": len(file_contents), 
            "total_files": len(all_files),
            "project_structure": project_structure,
            "max_depth": max_observed_depth,
            "extensions_found": sorted(self.scanned_extensions),
            "file_type_stats": file_type_stats
        }
    
    def _generate_file_type_statistics(self) -> Dict:
        """Generate statistics about file types in the codebase"""
        stats = {}
        
        # Count files by extension
        for file_path, info in self.all_files_info.items():
            ext = info['extension']
            if ext:
                if ext not in stats:
                    stats[ext] = {
                        'count': 0,
                        'total_size': 0,
                        'paths': []
                    }
                stats[ext]['count'] += 1
                stats[ext]['total_size'] += info['size']
                # Store a limited number of example paths
                if len(stats[ext]['paths']) < 5:
                    stats[ext]['paths'].append(file_path)
        
        # Sort by count
        sorted_stats = {k: stats[k] for k in sorted(stats, key=lambda x: stats[x]['count'], reverse=True)}
        
        return sorted_stats
    
    def _build_directory_structure(self, root_dir: str, ignore_dirs: List[str], max_depth: int = None) -> Dict:
        """Build a detailed directory structure for navigation"""
        structure = {}
        
        for root, dirs, files in os.walk(root_dir):
            # Skip ignored directories
            dirs[:] = [d for d in dirs if d not in ignore_dirs]
            
            # Check depth limit if specified
            if max_depth is not None:
                current_depth = root.replace(root_dir, '').count(os.sep)
                if current_depth > max_depth:
                    dirs[:] = []  # Don't go deeper
                    continue
            
            # Get the relative path from the root directory
            rel_path = os.path.relpath(root, root_dir)
            if rel_path == ".":
                rel_path = ""
                
            # Initialize this directory in the structure
            current_dir = {
                'path': rel_path,
                'abs_path': root,
                'dirs': [d for d in dirs if d not in ignore_dirs],
                'files': files,
                'parent': os.path.dirname(rel_path),
                'depth': rel_path.count(os.sep) if rel_path else 0
            }
            
            # Store in structure
            structure[rel_path] = current_dir
            
        return structure
    
    def auto_analyze(self, max_files: int = 10, min_size: int = 500) -> Dict:
        """
        Automatically analyze the most important files in the codebase
        
        Args:
            max_files: Maximum number of files to analyze
            min_size: Minimum file size in bytes to consider
            
        Returns:
            Dictionary with analysis results
        """
        if not self.codebase_context:
            return {"error": "Please scan a codebase first using the scan command."}
            
        results = {}
        important_files = []
        
        # Find potentially important files
        for path, info in self.codebase_context.items():
            # Skip files that are too small
            if info['size'] < min_size:
                continue
                
            # Look for key files
            filename = os.path.basename(path).lower()
            is_important = False
            
            # Check if it's a significant file by name
            if filename in ['main.py', 'app.py', 'index.js', 'app.js', 'main.js', 
                           'settings.py', 'config.py', 'package.json', 'readme.md',
                           'dockerfile', 'docker-compose.yml', 'makefile', 'setup.py']:
                is_important = True
                score = 100  # High importance score
            
            # Check if it might be important by location (root directory files)
            elif path.count(os.sep) == 0:
                is_important = True
                score = 50  # Medium importance
            
            # Check by size (larger files might be more important)
            elif info['size'] > 5000:
                is_important = True
                score = info['size'] / 1000  # Score based on size
                
            # Check if it's likely a model/core logic file
            elif filename.endswith(('models.py', 'views.py', 'controllers.py', 'service.py')):
                is_important = True
                score = 80  # Higher importance
            
            if is_important:
                important_files.append({
                    'path': path,
                    'size': info['size'],
                    'lines': info['lines'],
                    'score': score
                })
        
        # Sort by importance score
        important_files.sort(key=lambda x: x['score'], reverse=True)
        
        # Analyze the top files
        for file_info in important_files[:max_files]:
            path = file_info['path']
            print_colorized(f"🔍 Auto-analyzing: {path}", 36)
            analysis = self.analyze_file(path)
            results[path] = {
                'size': file_info['size'],
                'lines': file_info['lines'],
                'analysis': analysis
            }
        
        return {
            'count': len(results),
            'results': results
        }
    
    def scan_nested_directories(self, target_dir: str = None, max_depth: int = None) -> Dict:
        """
        Scan a specific directory and all its nested subdirectories
        
        Args:
            target_dir: The target directory to scan (relative or absolute)
            max_depth: Maximum depth to scan (None for unlimited)
            
        Returns:
            Dictionary with scan results
        """
        # If no target directory specified, use current directory
        if target_dir is None:
            target_dir = self.current_directory
        elif not os.path.isabs(target_dir):
            # If it's a relative path, make it absolute
            target_dir = os.path.join(self.current_directory, target_dir)
            
        # Make sure the target is within the codebase
        if not target_dir.startswith(self.codebase_root):
            return {"error": f"Target directory {target_dir} is outside the scanned codebase."}
            
        # Make sure it's a directory
        if not os.path.isdir(target_dir):
            return {"error": f"{target_dir} is not a valid directory."}
            
        # Get the relative path from the codebase root
        rel_path = os.path.relpath(target_dir, self.codebase_root)
        
        # Find all files in this directory and subdirectories
        matching_files = {}
        nested_dirs = set()
        
        for path, info in self.all_files_info.items():
            # Check if the path starts with the target directory
            if path == rel_path or path.startswith(rel_path + os.sep):
                # Check depth limit if specified
                if max_depth is not None:
                    # Calculate relative depth from the target directory
                    rel_depth = path.replace(rel_path, '').count(os.sep)
                    if rel_depth > max_depth:
                        continue
                
                matching_files[path] = info
                
                # Track directories
                dir_path = os.path.dirname(path)
                nested_dirs.add(dir_path)
        
        # Sort nested directories by depth
        sorted_dirs = sorted(nested_dirs, key=lambda x: x.count(os.sep))
        
        return {
            'target_directory': rel_path if rel_path != '.' else '/',
            'file_count': len(matching_files),
            'directory_count': len(sorted_dirs),
            'max_depth': max(info.get('depth', 0) - rel_path.count(os.sep) for info in matching_files.values()) if matching_files else 0,
            'files': matching_files,
            'directories': sorted_dirs
        }
    
    def find_all_files_by_extension(self, extension: str, include_content: bool = False) -> Dict:
        """
        Find all files with a specific extension across the entire codebase
        
        Args:
            extension: File extension to search for (without the dot)
            include_content: Whether to include file content in results
            
        Returns:
            Dictionary with search results
        """
        if not self.all_files_info:
            return {"error": "Please scan a codebase first using the scan command."}
            
        # Normalize extension (remove dot if present)
        if extension.startswith('.'):
            extension = extension[1:]
            
        matching_files = {}
        
        for path, info in self.all_files_info.items():
            if info['extension'].lower() == extension.lower():
                matching_files[path] = {
                    'path': path,
                    'size': info['size'],
                    'last_modified': info['last_modified']
                }
                
                # Include content if requested and available
                if include_content and path in self.codebase_context:
                    matching_files[path]['content'] = self.codebase_context[path]['content']
                    matching_files[path]['lines'] = self.codebase_context[path]['lines']
        
        return {
            'extension': extension,
            'count': len(matching_files),
            'files': matching_files
        }
    
    def list_directory(self, directory: str = None) -> Dict:
        """
        List contents of a directory within the codebase
        
        Args:
            directory: Directory path relative to codebase root, or None for current directory
            
        Returns:
            Dictionary with directory contents
        """
        if not self.codebase_root:
            return {"error": "Please scan a codebase first using the scan command."}
        
        # If no directory specified, use current directory
        if directory is None:
            target_dir = self.current_directory
        else:
            # Handle relative paths
            if directory.startswith(".."):
                # Going up a level
                parent_dir = os.path.dirname(self.current_directory)
                if parent_dir == self.current_directory:  # Already at root
                    target_dir = self.codebase_root
                else:
                    target_dir = parent_dir
            elif directory == ".":
                target_dir = self.current_directory
            elif os.path.isabs(directory):
                # Absolute path - make sure it's within the codebase
                if not directory.startswith(self.codebase_root):
                    return {"error": f"Directory '{directory}' is outside the scanned codebase."}
                target_dir = directory
            else:
                # Relative to current directory
                target_dir = os.path.normpath(os.path.join(self.current_directory, directory))
        
        # Make sure it's a directory and exists
        if not os.path.isdir(target_dir):
            return {"error": f"'{target_dir}' is not a valid directory."}
            
        # Get the contents
        try:
            # Update current directory
            self.current_directory = target_dir
            
            # Get relative path from codebase root
            rel_path = os.path.relpath(target_dir, self.codebase_root)
            if rel_path == ".":
                rel_path = ""
                
            # Get subdirectories and files in this directory
            dirs = []
            files = []
            
            for item in os.listdir(target_dir):
                full_path = os.path.join(target_dir, item)
                rel_item_path = os.path.relpath(full_path, self.codebase_root)
                
                if os.path.isdir(full_path):
                    # Count files and subdirectories for this directory
                    subdir_files = 0
                    subdir_dirs = 0
                    
                    for root, subdirs, subfiles in os.walk(full_path):
                        subdir_files += len(subfiles)
                        subdir_dirs += len(subdirs)
                    
                    dirs.append({
                        'name': item,
                        'path': rel_item_path,
                        'abs_path': full_path,
                        'contains_files': subdir_files,
                        'contains_dirs': subdir_dirs
                    })
                else:
                    file_extension = os.path.splitext(item)[1].lower()
                    file_size = os.path.getsize(full_path)
                    
                    files.append({
                        'name': item,
                        'path': rel_item_path,
                        'extension': file_extension[1:] if file_extension else '',
                        'size': file_size,
                        'last_modified': time.strftime('%Y-%m-%d %H:%M:%S', 
                                                      time.localtime(os.path.getmtime(full_path)))
                    })
            
            return {
                'current_dir': rel_path if rel_path else '/',
                'abs_path': target_dir,
                'parent_dir': os.path.dirname(rel_path) if rel_path else None,
                'directories': sorted(dirs, key=lambda x: x['name']),
                'files': sorted(files, key=lambda x: x['name']),
                'dir_count': len(dirs),
                'file_count': len(files),
                'depth': rel_path.count(os.sep) if rel_path else 0
            }
            
        except Exception as e:
            self.logger.error(f"Error listing directory: {str(e)}")
            return {"error": f"Error listing directory: {str(e)}"}
    
    def change_directory(self, directory: str) -> Dict:
        """
        Change the current working directory for navigation
        
        Args:
            directory: Directory path to change to (relative or absolute)
            
        Returns:
            Dictionary with new directory information
        """
        return self.list_directory(directory)
    
    def _generate_project_structure(self, root_dir: str, ignore_dirs: List[str]) -> Dict:
        """Generate a tree structure of the project"""
        structure = {"name": os.path.basename(root_dir), "type": "directory", "children": []}
        
        def build_tree(directory, tree_node):
            entries = os.listdir(directory)
            for entry in entries:
                full_path = os.path.join(directory, entry)
                if os.path.isdir(full_path):
                    if entry not in ignore_dirs:
                        dir_node = {"name": entry, "type": "directory", "children": []}
                        tree_node["children"].append(dir_node)
                        build_tree(full_path, dir_node)
                else:
                    file_extension = os.path.splitext(entry)[1].lower()
                    tree_node["children"].append({
                        "name": entry, 
                        "type": "file", 
                        "extension": file_extension[1:] if file_extension else ""  # Remove the dot
                    })
        
        build_tree(root_dir, structure)
        return structure

    def _generate_project_summary(self) -> None:
        """Generate an overall project summary based on the codebase"""
        if not self.codebase_context:
            self.logger.warning("Cannot generate project summary: No codebase scanned yet")
            return
            
        file_types = set(info['language'] for info in self.codebase_context.values())
        total_lines = sum(info['lines'] for info in self.codebase_context.values())
        
        # Extract filenames for important files that might indicate project type/purpose
        important_files = [
            path for path in self.codebase_context.keys() 
            if os.path.basename(path).lower() in ['readme.md', 'package.json', 'setup.py', 'requirements.txt', 
                                                 'pom.xml', 'build.gradle', 'cargo.toml', 'go.mod',
                                                 'dockerfile', 'docker-compose.yml', 'makefile']
        ]
        
        # Extract content from README if exists
        readme_content = ""
        for file_path in self.codebase_context:
            if os.path.basename(file_path).lower() == 'readme.md':
                readme_content = self.codebase_context[file_path]['content']
                break
                
        # Prepare the prompt for the project summary
        prompt = f"""Analyze this codebase information and provide a comprehensive summary:

1. Project Statistics:
   - Total files: {len(self.all_files_info)}
   - Analyzed files: {len(self.codebase_context)}
   - Total lines of code (in analyzed files): {total_lines}
   - File types: {', '.join(sorted(file_types))}
   - Directory depth: Up to {self.scan_depth} levels of nesting
   - Important configuration files found: {', '.join(important_files) if important_files else 'None'}

2. README content (if available):
{readme_content[:2000] + '...' if len(readme_content) > 2000 else readme_content}

Based on this information, provide:
1. A concise summary of what this project appears to be about
2. The likely main purpose and functionality
3. The technologies and frameworks used
4. The project's architecture at a high level
5. Notable observations about the directory structure and organization

Keep the response under 500 words and focus on being accurate based on the provided information."""

        try:
            summary_response = self.client.chat.completions.create(
                messages=[
                    {
                        "role": "system",
                        "content": "You are a code analysis assistant tasked with understanding and summarizing codebases. Be concise but thorough."
                    },
                    {
                        "role": "user",
                        "content": prompt
                    }
                ],
                model=self.model_name,
                temperature=0.1
            )
            self.project_summary = summary_response.choices[0].message.content
            self.logger.info("Generated project summary")
        except Exception as e:
            self.logger.error(f"Error generating project summary: {str(e)}")
            self.project_summary = "Error generating project summary."

    def analyze_file(self, file_path: str) -> str:
        """Analyze a specific file and provide a detailed explanation"""
        if file_path not in self.codebase_context:
            # Try to load the file if it exists but wasn't loaded during the initial scan
            try:
                # Check if path is relative or absolute
                if os.path.isabs(file_path):
                    full_path = file_path
                    if not full_path.startswith(self.codebase_root):
                        return f"File '{file_path}' is outside the scanned codebase."
                    relative_path = os.path.relpath(full_path, self.codebase_root)
                else:
                    # Check if relative to current directory or codebase root
                    full_path = os.path.join(self.current_directory, file_path)
                    if not os.path.exists(full_path):
                        full_path = os.path.join(self.codebase_root, file_path)
                    relative_path = os.path.relpath(full_path, self.codebase_root)
                
                if os.path.exists(full_path) and os.path.isfile(full_path):
                    with open(full_path, 'r', encoding='utf-8', errors='replace') as f:
                        content = f.read()
                        file_extension = os.path.splitext(file_path)[1].lower()
                        language = file_extension[1:] if file_extension else "text"  # Remove the dot
                        self.codebase_context[relative_path] = {
                            'content': content,
                            'language': language,
                            'size': len(content),
                            'lines': len(content.splitlines()),
                            'path': relative_path,
                            'last_modified': os.path.getmtime(full_path),
                            'depth': relative_path.count(os.sep)
                        }
                        self.logger.info(f"Loaded file {relative_path} for analysis on-demand")
                        # Use the relative path moving forward
                        file_path = relative_path
                else:
                    return f"File '{file_path}' not found in the codebase."
            except Exception as e:
                return f"Error loading file '{file_path}': {str(e)}"
            
        # If we already have a summary for this file, return it
        if file_path in self.file_summaries:
            return self.file_summaries[file_path]
            
        file_info = self.codebase_context[file_path]
        file_content = file_info['content']
        language = file_info['language']
        
        # Prepare the prompt for file analysis
        prompt = f"""Analyze this file and provide a detailed explanation:

File: {file_path}
Language: {language}
Size: {file_info['size']} bytes
Lines: {file_info['lines']}

Content:
```{language}
{file_content[:10000] + '...' if len(file_content) > 10000 else file_content}
```

Please provide:
1. A summary of what this file does and its purpose in the project
2. Key functions, classes, or components and their roles
3. Any dependencies or imports and what they're used for
4. Notable design patterns or architectural choices
5. Potential issues, improvements, or optimizations

Keep the response under 500 words and focus on being accurate and insightful."""

        try:
            analysis_response = self.client.chat.completions.create(
                messages=[
                    {
                        "role": "system",
                        "content": "You are a code analysis assistant tasked with understanding and explaining code files. Be thorough but concise."
                    },
                    {
                        "role": "user",
                        "content": prompt
                    }
                ],
                model=self.model_name,
                temperature=0.1
            )
            file_summary = analysis_response.choices[0].message.content
            self.file_summaries[file_path] = file_summary
            return file_summary
        except Exception as e:
            self.logger.error(f"Error analyzing file: {str(e)}")
            return f"Error analyzing file: {str(e)}"

    @sleep_and_retry
    @limits(calls=30, period=60)
    def chat_with_codebase(self, user_input: str) -> str:
        """
        Interactive chat about the codebase with improved context
        """
        if not self.codebase_context:
            return "Please scan a codebase first using the scan command."
            
        # Create a context-aware prompt with more detailed information
        project_info = f"""
Project Summary: {self.project_summary[:500]}...

Total files: {len(self.all_files_info)}
Analyzed files: {len(self.codebase_context)} files
File types: {', '.join(set(info['language'] for info in self.codebase_context.values()))}
Total lines of code (in analyzed files): {sum(info['lines'] for info in self.codebase_context.values())}
Directory depth: Up to {self.scan_depth} levels of nesting
Current directory: {os.path.relpath(self.current_directory, self.codebase_root) if self.current_directory != self.codebase_root else '/'}
"""
        
        # Include information about key files
        key_files_info = ""
        for i, (path, info) in enumerate(sorted(
            # Continuing from where we left off...

            self.codebase_context.items(), 
            key=lambda x: x[1]['lines'], 
            reverse=True
        )[:10]):  # Show top 10 largest files
            key_files_info += f"- {path} ({info['lines']} lines, {info['language']})\n"
            
        prompt = f"""As a code assistant with access to the following codebase:

{project_info}

Key files:
{key_files_info}

User question: {user_input}

Please provide a detailed response based on the codebase context. If you need information about a specific file that isn't mentioned here, say so."""

        try:
            chat_completion = self.client.chat.completions.create(
                messages=[
                    {
                        "role": "system",
                        "content": "You are a code assistant with deep knowledge of the scanned codebase. Provide specific, contextual answers. If asked about specific files, you have detailed information available."
                    },
                    {
                        "role": "user",
                        "content": prompt
                    }
                ],
                model=self.model_name,
                temperature=0.1
            )
            return chat_completion.choices[0].message.content
        except Exception as e:
            self.logger.error(f"Error querying Groq API: {str(e)}")
            return f"Error: {str(e)}"

    def search_in_codebase(self, query: str, context_lines: int = 0, include_all_files: bool = True) -> Dict:
        """
        Enhanced search for specific patterns or code in the codebase
        
        Args:
            query: String to search for
            context_lines: Number of context lines to include before and after matches
            include_all_files: Whether to search in all files regardless of current directory
            
        Returns:
            Dictionary with search results
        """
        results = {}
        total_matches = 0
        
        # Determine if we should search only in current directory or entire codebase
        search_paths = []
        if include_all_files:
            search_paths = list(self.codebase_context.keys())
        else:
            current_rel_dir = os.path.relpath(self.current_directory, self.codebase_root)
            if current_rel_dir == '.':
                # At root, search everything
                search_paths = list(self.codebase_context.keys())
            else:
                # In a subdirectory, search only files in this directory
                search_paths = [path for path in self.codebase_context.keys() 
                                if path.startswith(current_rel_dir)]
        
        for file_path in search_paths:
            info = self.codebase_context[file_path]
            content = info['content']
            matches = []
            
            # Case-insensitive search
            pattern = re.compile(re.escape(query), re.IGNORECASE)
            lines = content.splitlines()
            
            # Find all matches with line numbers
            for i, line in enumerate(lines, 1):
                for match in pattern.finditer(line):
                    # Get context lines if requested
                    context_before = []
                    context_after = []
                    
                    if context_lines > 0:
                        start_line = max(0, i - context_lines - 1)
                        end_line = min(len(lines), i + context_lines)
                        
                        context_before = [(j, lines[j]) for j in range(start_line, i-1)]
                        context_after = [(j, lines[j]) for j in range(i, end_line)]
                    
                    matches.append({
                        'line_number': i,
                        'line_content': line.strip(),
                        'start': match.start(),
                        'end': match.end(),
                        'context_before': context_before,
                        'context_after': context_after
                    })
            
            if matches:
                total_matches += len(matches)
                results[file_path] = {
                    'language': info['language'],
                    'match_count': len(matches),
                    'matches': matches[:10]  # Limit to first 10 matches per file
                }
        
        return {
            'total_matches': total_matches,
            'files_with_matches': len(results),
            'results': results,
            'search_scope': 'entire_codebase' if include_all_files else 'current_directory'
        }
    
    def find_files(self, pattern: str = None, extension: str = None, path_pattern: str = None, 
                  in_current_dir: bool = False, recursive: bool = True, limit: int = 100) -> Dict:
        """
        Find files in the codebase based on various criteria
        
        Args:
            pattern: Pattern to match against filenames (e.g., "model*.py")
            extension: File extension to filter by (e.g., "py")
            path_pattern: Pattern to match against file paths (e.g., "*/models.py")
            in_current_dir: Whether to search only in current directory (vs entire codebase)
            recursive: Whether to search recursively in subdirectories
            limit: Maximum number of results to return
            
        Returns:
            Dictionary with matched files
        """
        if not self.all_files_info:
            return {"error": "Please scan a codebase first using the scan command."}
            
        results = []
        current_rel_dir = os.path.relpath(self.current_directory, self.codebase_root)
        if current_rel_dir == '.':
            current_rel_dir = ''
        
        for file_path, info in self.all_files_info.items():
            # Skip if not in current directory when that filter is applied
            if in_current_dir:
                if recursive:
                    # Include current directory and all subdirectories
                    if not (file_path == current_rel_dir or 
                           file_path.startswith(current_rel_dir + os.sep)):
                        continue
                else:
                    # Include only files directly in current directory (no subdirectories)
                    file_dir = os.path.dirname(file_path)
                    if file_dir != current_rel_dir:
                        continue
                
            matched = True
            
            if pattern and not fnmatch.fnmatch(os.path.basename(file_path), pattern):
                matched = False
                
            if extension and not file_path.lower().endswith(f".{extension.lower()}"):
                matched = False
                
            if path_pattern and not fnmatch.fnmatch(file_path, path_pattern):
                matched = False
                
            if matched:
                results.append({
                    'path': file_path,
                    'size': info['size'],
                    'extension': info['extension'],
                    'last_modified': time.strftime('%Y-%m-%d %H:%M:%S', 
                                                  time.localtime(info['last_modified']))
                })
                
            if len(results) >= limit:
                break
                
        return {
            'count': len(results),
            'results': results,
            'has_more': len(results) >= limit,
            'search_scope': 'current_directory' if in_current_dir else 'entire_codebase',
            'recursive': recursive
        }
        
    def get_file_content(self, file_path: str) -> Dict:
        """
        Get the content of a specific file
        """
        if not self.codebase_root:
            return {"error": "Please scan a codebase first using the scan command."}
        
        # Check if path is relative or absolute
        if os.path.isabs(file_path):
            full_path = file_path
            if not full_path.startswith(self.codebase_root):
                return {"error": f"File '{file_path}' is outside the scanned codebase."}
            relative_path = os.path.relpath(full_path, self.codebase_root)
        else:
            # Check if relative to current directory or codebase root
            full_path = os.path.join(self.current_directory, file_path)
            if os.path.exists(full_path) and os.path.isfile(full_path):
                relative_path = os.path.relpath(full_path, self.codebase_root)
            else:
                full_path = os.path.join(self.codebase_root, file_path)
                if os.path.exists(full_path) and os.path.isfile(full_path):
                    relative_path = file_path
                else:
                    return {"error": f"File '{file_path}' not found."}
            
        # Check if the file is already loaded in memory
        if relative_path in self.codebase_context:
            return {
                'path': relative_path,
                'content': self.codebase_context[relative_path]['content'],
                'language': self.codebase_context[relative_path]['language'],
                'lines': self.codebase_context[relative_path]['lines']
            }
            
        # Try to load the file
        try:
            if os.path.exists(full_path) and os.path.isfile(full_path):
                with open(full_path, 'r', encoding='utf-8', errors='replace') as f:
                    content = f.read()
                    file_extension = os.path.splitext(full_path)[1].lower()
                    language = file_extension[1:] if file_extension else "text"  # Remove the dot
                    
                    # Add to codebase context for future use
                    self.codebase_context[relative_path] = {
                        'content': content,
                        'language': language,
                        'size': len(content),
                        'lines': len(content.splitlines()),
                        'path': relative_path,
                        'last_modified': os.path.getmtime(full_path),
                        'depth': relative_path.count(os.sep)
                    }
                    
                    return {
                        'path': relative_path,
                        'content': content,
                        'language': language,
                        'lines': len(content.splitlines())
                    }
            else:
                return {"error": f"File '{file_path}' not found in the codebase."}
        except Exception as e:
            self.logger.error(f"Error reading file: {str(e)}")
            return {"error": f"Error reading file: {str(e)}"}
    
    def get_current_directory(self) -> str:
        """Get the current directory for navigation"""
        if not self.current_directory:
            return self.codebase_root
        return self.current_directory
    
    def suggest_feature_implementation(self, feature_description: str) -> Dict:
        """
        Suggest implementation details for a new feature based on the codebase
        """
        if not self.codebase_context:
            return {"error": "Please scan a codebase first using the scan command."}
            
        # Get key files for context (focusing on largest/most important files)
        key_files = sorted(
            self.codebase_context.items(), 
            key=lambda x: x[1]['lines'], 
            reverse=True
        )[:5]  # Top 5 largest files
        
        key_files_content = ""
        for path, info in key_files:
            truncated_content = info['content'][:2000] + "..." if len(info['content']) > 2000 else info['content']
            key_files_content += f"\nFile: {path}\n```{info['language']}\n{truncated_content}\n```\n"
            
        # Create the prompt for feature implementation suggestion
        prompt = f"""Based on the current codebase, suggest how to implement this new feature:

Feature description: {feature_description}

Project summary: {self.project_summary[:500]}

Here are some key files from the codebase for context:
{key_files_content}

Please provide:
1. Overall approach for implementing this feature
2. Files that need to be modified or created
3. Specific code changes with implementation details
4. Any potential challenges or considerations

Be specific and provide actual code snippets where appropriate."""

        try:
            suggestion_response = self.client.chat.completions.create(
                messages=[
                    {
                        "role": "system",
                        "content": "You are a code implementation assistant tasked with suggesting how to implement new features in an existing codebase. Be specific and practical."
                    },
                    {
                        "role": "user",
                        "content": prompt
                    }
                ],
                model=self.model_name,
                temperature=0.2
            )
            response_content = suggestion_response.choices[0].message.content
            
            # Parse the response to extract file changes
            proposed_changes = self._extract_proposed_changes(response_content)
            
            # Store as pending changes (to be approved later)
            change_id = f"change_{int(time.time())}"
            self.pending_changes[change_id] = {
                'feature_description': feature_description,
                'suggestion': response_content,
                'proposed_changes': proposed_changes,
                'timestamp': time.time()
            }
            
            return {
                'change_id': change_id,
                'suggestion': response_content,
                'proposed_changes': proposed_changes
            }
            
        except Exception as e:
            self.logger.error(f"Error generating feature suggestion: {str(e)}")
            return {"error": f"Error generating feature suggestion: {str(e)}"}
            
    def _extract_proposed_changes(self, suggestion_text: str) -> Dict:
        """
        Extract proposed file changes from the suggestion text
        """
        proposed_changes = {
            'files_to_modify': [],
            'files_to_create': []
        }
        
        # Simple regex-based extraction of code blocks with file paths
        # This is a basic implementation and could be enhanced
        
        # Look for patterns like "File: path/to/file.py" followed by code blocks
        file_pattern = r"(?:File|Create file|Modify file):\s*([^\n]+)\n```(?:\w+)?\n(.*?)```"
        matches = re.finditer(file_pattern, suggestion_text, re.DOTALL)
        
        for match in matches:
            file_path = match.group(1).strip()
            code_content = match.group(2).strip()
            
            # Determine if this is a new file or modified file
            if file_path in self.codebase_context:
                # Existing file to modify
                original_content = self.codebase_context[file_path]['content']
                
                # Generate a diff between original and proposed content
                diff = list(difflib.unified_diff(
                    original_content.splitlines(),
                    code_content.splitlines(),
                    fromfile=f"a/{file_path}",
                    tofile=f"b/{file_path}",
                    lineterm=''
                ))
                
                proposed_changes['files_to_modify'].append({
                    'path': file_path,
                    'original_content': original_content,
                    'new_content': code_content,
                    'diff': '\n'.join(diff)
                })
            else:
                # New file to create
                proposed_changes['files_to_create'].append({
                    'path': file_path,
                    'content': code_content
                })
                
        return proposed_changes
            
    def approve_changes(self, change_id: str) -> Dict:
        """
        Approve and apply the pending changes to the codebase
        """
        if change_id not in self.pending_changes:
            return {"error": f"Change ID {change_id} not found in pending changes."}
            
        changes = self.pending_changes[change_id]
        results = {
            'modified_files': [],
            'created_files': [],
            'errors': []
        }
        
        # Apply the changes to the codebase
        try:
            # First, make a backup of the codebase
            backup_dir = self._backup_codebase()
            
            # Apply modifications to existing files
            for file_change in changes['proposed_changes']['files_to_modify']:
                file_path = os.path.join(self.codebase_root, file_change['path'])
                try:
                    with open(file_path, 'w', encoding='utf-8') as f:
                        f.write(file_change['new_content'])
                    
                    # Update the in-memory codebase context
                    relative_path = os.path.relpath(file_path, self.codebase_root)
                    self.codebase_context[relative_path]['content'] = file_change['new_content']
                    self.codebase_context[relative_path]['size'] = len(file_change['new_content'])
                    self.codebase_context[relative_path]['lines'] = len(file_change['new_content'].splitlines())
                    
                    results['modified_files'].append(relative_path)
                except Exception as e:
                    results['errors'].append(f"Error modifying {file_path}: {str(e)}")
            
            # Create new files
            for file_create in changes['proposed_changes']['files_to_create']:
                file_path = os.path.join(self.codebase_root, file_create['path'])
                
                # Ensure the directory exists
                os.makedirs(os.path.dirname(file_path), exist_ok=True)
                
                try:
                    with open(file_path, 'w', encoding='utf-8') as f:
                        f.write(file_create['content'])
                    
                    # Add to the in-memory codebase context
                    relative_path = os.path.relpath(file_path, self.codebase_root)
                    language = os.path.splitext(file_path)[1][1:]  # Get extension without dot
                    
                    self.codebase_context[relative_path] = {
                        'content': file_create['content'],
                        'language': language,
                        'size': len(file_create['content']),
                        'lines': len(file_create['content'].splitlines()),
                        'path': relative_path,
                        'last_modified': time.time(),
                        'depth': relative_path.count(os.sep)
                    }
                    
                    # Add to all files info
                    self.all_files_info[relative_path] = {
                        'path': relative_path,
                        'extension': language,
                        'size': len(file_create['content']),
                        'last_modified': time.time()
                    }
                    
                    results['created_files'].append(relative_path)
                except Exception as e:
                    results['errors'].append(f"Error creating {file_path}: {str(e)}")
            
            # Remove from pending changes if successful
            if not results['errors']:
                del self.pending_changes[change_id]
                results['success'] = True
                results['backup_dir'] = backup_dir
            else:
                results['success'] = False
                
            return results
            
        except Exception as e:
            self.logger.error(f"Error applying changes: {str(e)}")
            return {
                'success': False,
                'error': f"Error applying changes: {str(e)}"
            }
            
    def _backup_codebase(self) -> str:
        """
        Create a backup of the current codebase before making changes
        """
        timestamp = time.strftime("%Y%m%d-%H%M%S")
        backup_dir = os.path.join(
            tempfile.gettempdir(), 
            f"codebase_backup_{timestamp}"
        )
        
        try:
            # Copy the entire codebase to the backup directory
            shutil.copytree(
                self.codebase_root, 
                backup_dir,
                ignore=shutil.ignore_patterns(
                    '.git', 'venv', 'env', '__pycache__', 'node_modules', '.vscode', '.idea'
                )
            )
            self.logger.info(f"Created backup at {backup_dir}")
            return backup_dir
        except Exception as e:
            self.logger.error(f"Error creating backup: {str(e)}")
            raise

    def get_project_summary(self) -> str:
        """
        Get the generated project summary
        """
        if not self.project_summary:
            return "No project summary available. Please scan a codebase first."
        return self.project_summary
        
    def list_pending_changes(self) -> Dict:
        """
        List all pending changes that are waiting for approval
        """
        result = []
        for change_id, change_info in self.pending_changes.items():
            result.append({
                'change_id': change_id,
                'feature_description': change_info['feature_description'],
                'timestamp': change_info['timestamp'],
                'files_to_modify': len(change_info['proposed_changes']['files_to_modify']),
                'files_to_create': len(change_info['proposed_changes']['files_to_create'])
            })
        
        return {
            'count': len(result),
            'changes': result
        }
        
    def get_change_details(self, change_id: str) -> Dict:
        """
        Get detailed information about a specific pending change
        """
        if change_id not in self.pending_changes:
            return {"error": f"Change ID {change_id} not found in pending changes."}
            
        return self.pending_changes[change_id]
        
    def reject_changes(self, change_id: str) -> Dict:
        """
        Reject and remove a pending change
        """
        if change_id not in self.pending_changes:
            return {"error": f"Change ID {change_id} not found in pending changes."}
            
        del self.pending_changes[change_id]
        return {"success": True, "message": f"Change {change_id} has been rejected and removed."}
        
    def examine_models(self) -> Dict:
        """
        Find and analyze all model files in the codebase
        """
        if not self.all_files_info:
            return {"error": "Please scan a codebase first using the scan command."}
            
        # Find all models.py files
        model_files = []
        for file_path in self.all_files_info:
            if file_path.endswith('models.py'):
                model_files.append(file_path)
                
        results = {
            'count': len(model_files),
            'model_files': model_files,
            'models_data': {}
        }
        
        # Analyze each model file
        for model_file in model_files:
            # Ensure the file content is loaded
            file_data = self.get_file_content(model_file)
            
            if 'error' in file_data:
                results['models_data'][model_file] = {"error": file_data['error']}
                continue
                
            content = file_data['content']
            
            # Extract model information
            models_info = self._extract_python_models(content, model_file)
            results['models_data'][model_file] = models_info
            
        return results
        
    def _extract_python_models(self, content: str, file_path: str) -> Dict:
        """
        Extract model classes and their fields from a Django-style models.py file
        """
        models = {}
        
        try:
            # Parse the Python code
            tree = ast.parse(content)
            
            # Find all class definitions
            for node in ast.walk(tree):
                if isinstance(node, ast.ClassDef):
                    # Check if it's likely a model class (inherits from Model or has Meta inner class)
                    is_model = False
                    model_fields = []
                    model_relationships = []
                    model_meta = {}
                    
                    # Check base classes
                    for base in node.bases:
                        if isinstance(base, ast.Name) and base.id in ['Model', 'models.Model']:
                            is_model = True
                            break
                        elif isinstance(base, ast.Attribute) and base.attr == 'Model':
                            is_model = True
                            break
                    
                    # Look for class variables and inner classes
                    for item in node.body:
                        # Check for field assignments
                        if isinstance(item, ast.Assign):
                            for target in item.targets:
                                if isinstance(target, ast.Name):
                                    field_name = target.id
                                    
                                    # Try to determine the field type
                                    field_type = "Unknown"
                                    if isinstance(item.value, ast.Call):
                                        if isinstance(item.value.func, ast.Name):
                                            field_type = item.value.func.id
                                        elif isinstance(item.value.func, ast.Attribute):
                                            field_type = item.value.func.attr
                                    
                                    # Check for relationship fields
                                    if field_type in ['ForeignKey', 'OneToOneField', 'ManyToManyField']:
                                        related_model = "Unknown"
                                        # Try to get the related model from the first argument
                                        if item.value.args and isinstance(item.value.args[0], ast.Str):
                                            related_model = item.value.args[0].s
                                        elif item.value.args and isinstance(item.value.args[0], ast.Constant):
                                            related_model = item.value.args[0].value
                                        
                                        model_relationships.append({
                                            'name': field_name,
                                            'type': field_type,
                                            'related_model': related_model
                                        })
                                    else:
                                        model_fields.append({
                                            'name': field_name,
                                            'type': field_type
                                        })
                        
                        # Check for Meta inner class
                        elif isinstance(item, ast.ClassDef) and item.name == 'Meta':
                            is_model = True  # Presence of Meta class is a good indicator
                            
                            # Extract Meta options
                            for meta_item in item.body:
                                if isinstance(meta_item, ast.Assign) and isinstance(meta_item.targets[0], ast.Name):
                                    meta_name = meta_item.targets[0].id
                                    
                                    # Extract the value
                                    if isinstance(meta_item.value, ast.Str):
                                        model_meta[meta_name] = meta_item.value.s
                                    elif isinstance(meta_item.value, ast.Constant):
                                        model_meta[meta_name] = meta_item.value.value
                                    elif isinstance(meta_item.value, ast.List):
                                        values = []
                                        for elt in meta_item.value.elts:
                                            if isinstance(elt, ast.Str):
                                                values.append(elt.s)
                                            elif isinstance(elt, ast.Constant):
                                                values.append(elt.value)
                                        model_meta[meta_name] = values
                                    elif isinstance(meta_item.value, ast.Tuple):
                                        values = []
                                        for elt in meta_item.value.elts:
                                            if isinstance(elt, ast.Str):
                                                values.append(elt.s)
                                            elif isinstance(elt, ast.Constant):
                                                values.append(elt.value)
                                        model_meta[meta_name] = tuple(values)
                    
                    if is_model:
                        models[node.name] = {
                            'fields': model_fields,
                            'relationships': model_relationships,
                            'meta': model_meta
                        }
            
            return {
                'models': models,
                'count': len(models)
            }
            
        except Exception as e:
            self.logger.error(f"Error parsing models in {file_path}: {str(e)}")
            return {
                'error': f"Error parsing models: {str(e)}",
                'models': {},
                'count': 0
            }

    def auto_scan_nested_directories(self, max_depth: int = None) -> Dict:
        """
        Automatically scan and analyze all nested directories in the codebase
        
        Args:
            max_depth: Maximum depth for nested scanning (None for unlimited)
            
        Returns:
            Dictionary with scan statistics
        """
        if not self.directory_structure:
            return {"error": "Please scan a codebase first using the scan command."}
            
        # Find all directories
        all_dirs = list(self.directory_structure.keys())
        
        # Sort by depth to process them in order (shallow to deep)
        all_dirs.sort(key=lambda x: x.count(os.sep))
        
        # Skip directories beyond max_depth if specified
        if max_depth is not None:
            all_dirs = [d for d in all_dirs if d.count(os.sep) <= max_depth]
            
        results = {
            'scanned_directories': 0,
            'total_directories': len(all_dirs),
            'skipped_directories': 0,
            'directory_stats': {},
            'file_extensions': set()
        }
        
        # Scan each directory
        for dir_path in all_dirs:
            abs_dir_path = os.path.join(self.codebase_root, dir_path) if dir_path else self.codebase_root
            
            # Skip if not a directory (shouldn't happen, but just in case)
            if not os.path.isdir(abs_dir_path):
                results['skipped_directories'] += 1
                continue
                
            # Get files in this directory
            dir_files = []
            dir_extensions = set()
            
            for file_path, info in self.all_files_info.items():
                if os.path.dirname(file_path) == dir_path:
                    dir_files.append(file_path)
                    if info['extension']:
                        dir_extensions.add(info['extension'])
                        results['file_extensions'].add(info['extension'])
            
            results['directory_stats'][dir_path] = {
                'file_count': len(dir_files),
                'file_extensions': sorted(list(dir_extensions)),
                'depth': dir_path.count(os.sep) if dir_path else 0
            }
            
            results['scanned_directories'] += 1
            
        return results

def print_colorized(text: str, color_code: int = 0, end="\n") -> None:
    """Print text with ANSI color codes"""
    print(f"\033[{color_code}m{text}\033[0m", end=end)
    
def print_header(text: str) -> None:
    """Print a header with a box around it"""
    width = len(text) + 4
    print_colorized("┌" + "─" * width + "┐", 36)
    print_colorized("│  " + text + "  │", 36)
    print_colorized("└" + "─" * width + "┘", 36)

def print_project_summary(summary: str) -> None:
    """Print the project summary with formatting"""
    print_header("📊 PROJECT SUMMARY")
    print_colorized(summary, 37)
    print()

def print_file_analysis(file_path: str, analysis: str) -> None:
    """Print the file analysis with formatting"""
    print_header(f"📄 FILE ANALYSIS: {file_path}")
    print_colorized(analysis, 37)
    print()

def print_search_results(results: Dict) -> None:
    """Print search results with formatting"""
    scope = "current directory" if results['search_scope'] == 'current_directory' else "entire codebase"
    print_header(f"🔍 SEARCH RESULTS: {results['total_matches']} matches in {results['files_with_matches']} files (scope: {scope})")
    
    for file_path, info in results['results'].items():
        print_colorized(f"\n📄 {file_path} ({info['match_count']} matches)", 33)
        
        for i, match in enumerate(info['matches'], 1):
            if i > 5:  # Limit to 5 matches per file in output
                print_colorized(f"   ... and {info['match_count'] - 5} more matches", 90)
                break
                
            # Print context lines before match if available
            if match.get('context_before'):
                for line_num, line in match['context_before']:
                    print_colorized(f"   Line {line_num+1}: {line}", 90)
                
            # Print the matching line with highlight
            line_content = match['line_content']
            start = match['start']
            end = match['end']
            
            # Highlight the match in the line
            highlighted = (
                line_content[:start] + 
                f"\033[41m{line_content[start:end]}\033[0m" + 
                line_content[end:]
            )
            
            print_colorized(f"   Line {match['line_number']}: {highlighted}", 37)
            
            # Print context lines after match if available
            if match.get('context_after'):
                for line_num, line in match['context_after']:
                    print_colorized(f"   Line {line_num+1}: {line}", 90)
            
            # Add separator between matches
            if i < min(5, info['match_count']):
                print_colorized("   -----------", 90)
    
    print()

def print_directory_listing(directory_info: Dict) -> None:
    """Print directory listing with formatting"""
    if 'error' in directory_info:
        print_colorized(f"❌ Error: {directory_info['error']}", 31)
        return
        
    current_dir = directory_info['current_dir']
    if current_dir == '/':
        current_dir = '.'  # Root directory
        
    print_header(f"📂 DIRECTORY: {current_dir}")
    print_colorized(f"Full path: {directory_info['abs_path']}", 90)
    print_colorized(f"{directory_info['dir_count']} directories, {directory_info['file_count']} files\n", 90)
    
    # Print parent directory option if not at root
    if directory_info['parent_dir'] is not None:
        print_colorized(f"📁 .. (Parent Directory)", 34)
    
    # Print subdirectories
    for directory in directory_info['directories']:
        # Show number of files/directories if available
        extra_info = ""
        if 'contains_files' in directory:
            extra_info = f" ({directory['contains_files']} files, {directory['contains_dirs']} subdirs)"
        print_colorized(f"📁 {directory['name']}/{extra_info}", 34)
    
    # Print files
    for file in directory_info['files']:
        size_kb = file['size'] / 1024
        print_colorized(f"📄 {file['name']} ({size_kb:.1f} KB)", 37)
    
    print()
    print_colorized("Use 'cd <directory>' to navigate, 'viewfile <filename>' to view file contents.", 90)
    print()

def print_feature_suggestion(suggestion: Dict) -> None:
    """Print feature implementation suggestion with formatting"""
    print_header(f"💡 FEATURE IMPLEMENTATION SUGGESTION (ID: {suggestion['change_id']})")
    
    print_colorized("\n🔹 OVERVIEW:", 36)
    print_colorized(suggestion['suggestion'], 37)
    
    print_colorized("\n🔹 PROPOSED CHANGES:", 36)
    print_colorized(f"  Files to modify: {len(suggestion['proposed_changes']['files_to_modify'])}", 33)
    for file in suggestion['proposed_changes']['files_to_modify']:
        print_colorized(f"  - {file['path']}", 37)
        
    print_colorized(f"\n  Files to create: {len(suggestion['proposed_changes']['files_to_create'])}", 33)
    for file in suggestion['proposed_changes']['files_to_create']:
        print_colorized(f"  - {file['path']}", 37)
    
    print()
    print_colorized("Use 'approve <change_id>' to apply these changes or 'reject <change_id>' to discard them.", 90)
    print()

def print_pending_changes(changes: Dict) -> None:
    """Print list of pending changes with formatting"""
    print_header(f"🕒 PENDING CHANGES ({changes['count']})")
    
    for change in changes['changes']:
        timestamp = time.strftime('%Y-%m-%d %H:%M:%S', time.localtime(change['timestamp']))
        print_colorized(f"\n🔹 ID: {change['change_id']} ({timestamp})", 36)
        print_colorized(f"  Feature: {change['feature_description']}", 37)
        print_colorized(f"  Files to modify: {change['files_to_modify']}", 33)
        print_colorized(f"  Files to create: {change['files_to_create']}", 33)
    
    print()

def print_change_result(result: Dict) -> None:
    """Print the result of applying changes with formatting"""
    if result.get('success', False):
        print_header("✅ CHANGES APPLIED SUCCESSFULLY")
        
        if result['modified_files']:
            print_colorized(f"\n🔹 Modified {len(result['modified_files'])} files:", 36)
            for file in result['modified_files']:
                print_colorized(f"  - {file}", 37)
                
        if result['created_files']:
            print_colorized(f"\n🔹 Created {len(result['created_files'])} files:", 36)
            for file in result['created_files']:
                print_colorized(f"  - {file}", 37)
                
        print_colorized(f"\n📁 Backup created at: {result['backup_dir']}", 90)
    else:
        print_header("❌ ERROR APPLYING CHANGES")
        
        if 'error' in result:
            print_colorized(result['error'], 31)
        
        if 'errors' in result and result['errors']:
            for error in result['errors']:
                print_colorized(f"- {error}", 31)
    
    print()

def print_file_list(files: Dict) -> None:
    """Print list of files with formatting"""
    scope = "current directory" if files.get('search_scope') == 'current_directory' else "entire codebase"
    recursive = "including subdirectories" if files.get('recursive', True) else "current directory only"
    count = files['count']
    print_header(f"📁 FILE LIST: {count} file(s) found (scope: {scope}, {recursive})")
    
    for file in files['results']:
        size_kb = file['size'] / 1024
        print_colorized(f"  {file['path']} ({size_kb:.1f} KB, {file['extension']})", 37)
    
    if files.get('has_more', False):
        print_colorized("\nNote: Not all matching files are shown. Refine your search criteria for more specific results.", 90)
    
    print()

def print_auto_analysis_results(results: Dict) -> None:
    """Print results of automatic file analysis"""
    print_header(f"🔍 AUTO-ANALYSIS RESULTS: {results['count']} key files analyzed")
    
    for file_path, info in results['results'].items():
        print_colorized(f"\n📄 {file_path} ({info['lines']} lines)", 33)
        print_colorized(info['analysis'], 37)
        print_colorized("-----------------------------------", 90)
    
    print()

def print_models_analysis(models_data: Dict) -> None:
    """Print analysis of model files and their structure"""
    print_header(f"🔍 MODELS ANALYSIS: {models_data['count']} model file(s)")
    
    for file_path, data in models_data['models_data'].items():
        print_colorized(f"\n📄 {file_path}", 33)
        
        if 'error' in data:
            print_colorized(f"  Error: {data['error']}", 31)
            continue
            
        if data['count'] == 0:
            print_colorized("  No models found in this file.", 90)
            continue
            
        print_colorized(f"  Found {data['count']} model(s):", 36)
        
        for model_name, model_info in data['models'].items():
            print_colorized(f"\n  📊 {model_name}", 36)
            
            # Print fields
            if model_info['fields']:
                print_colorized("    Fields:", 37)
                for field in model_info['fields']:
                    print_colorized(f"      - {field['name']}: {field['type']}", 37)
            
            # Print relationships
            if model_info['relationships']:
                print_colorized("    Relationships:", 37)
                for rel in model_info['relationships']:
                    print_colorized(f"      - {rel['name']}: {rel['type']} to {rel['related_model']}", 37)
            
            # Print meta options
            if model_info['meta']:
                print_colorized("    Meta:", 37)
                for meta_key, meta_value in model_info['meta'].items():
                    print_colorized(f"      - {meta_key}: {meta_value}", 37)
    
    print()

def print_file_content(file_data: Dict) -> None:
    """Print content of a file with syntax highlighting based on file type"""
    if 'error' in file_data:
        print_colorized(f"❌ Error: {file_data['error']}", 31)
        return
        
    print_header(f"📄 FILE CONTENT: {file_data['path']}")
    print_colorized(f"Language: {file_data['language']}, Lines: {file_data['lines']}\n", 90)
    
    # Simple line-by-line printing with line numbers
    for i, line in enumerate(file_data['content'].splitlines(), 1):
        line_num = f"{i:4d} | "
        print_colorized(line_num, 90, end="")
        print_colorized(line, 37)
    
    print()

def print_nested_scan_results(results: Dict) -> None:
    """Print results of a nested directory scan"""
    if 'error' in results:
        print_colorized(f"❌ Error: {results['error']}", 31)
        return
        
    print_header(f"📂 NESTED SCAN: {results['target_directory']}")
    print_colorized(f"Found {results['file_count']} files in {results['directory_count']} directories", 36)
    print_colorized(f"Maximum nesting depth: {results['max_depth']} levels\n", 36)
    
    # Print directories by nesting level
    print_colorized("🔹 Directories:", 33)
    
    # Group directories by depth for cleaner output
    by_depth = {}
    for dir_path in results['directories']:
        depth = dir_path.count(os.sep)
        if depth not in by_depth:
            by_depth[depth] = []
        by_depth[depth].append(dir_path)
    
    # Print directories grouped by depth
    for depth in sorted(by_depth.keys()):
        print_colorized(f"\n  Level {depth}:", 36)
        for dir_path in sorted(by_depth[depth]):
            # Count files directly in this directory
            direct_files = sum(1 for f in results['files'] if os.path.dirname(f) == dir_path)
            print_colorized(f"  - {dir_path}/ ({direct_files} files)", 37)
    
    print()

def print_extensions_report(results: Dict) -> None:
    """Print report of file extensions found in the codebase"""
    extension = results['extension']
    count = results['count']
    
    print_header(f"📊 FILE EXTENSION REPORT: Found {count} .{extension} files")
    
    # Sort files by size
    sorted_files = sorted(
        results['files'].items(), 
        key=lambda x: x[1]['size'], 
        reverse=True
    )
    
    # Print the top files
    for i, (path, info) in enumerate(sorted_files[:20]):  # Limit to first 20
        size_kb = info['size'] / 1024
        print_colorized(f"  {path} ({size_kb:.1f} KB)", 37)
        
    if count > 20:
        print_colorized(f"\n  ... and {count - 20} more .{extension} files", 90)
    
    print()

def print_auto_scan_results(results: Dict) -> None:
    """Print results of auto-scanning all nested directories"""
    if 'error' in results:
        print_colorized(f"❌ Error: {results['error']}", 31)
        return
        
    print_header("🔍 AUTO-SCAN NESTED DIRECTORIES")
    print_colorized(f"Scanned {results['scanned_directories']} directories out of {results['total_directories']} total", 36)
    print_colorized(f"Skipped {results['skipped_directories']} directories", 36)
    print_colorized(f"Found file extensions: {', '.join(sorted(results['file_extensions']))}\n", 36)
    
    # Print directory statistics by depth
    by_depth = {}
    for dir_path, stats in results['directory_stats'].items():
        depth = stats['depth']
        if depth not in by_depth:
            by_depth[depth] = []
        by_depth[depth].append((dir_path, stats))
    
    # Print directories grouped by depth
    for depth in sorted(by_depth.keys()):
        dirs_at_depth = len(by_depth[depth])
        files_at_depth = sum(stats['file_count'] for _, stats in by_depth[depth])
        
        print_colorized(f"Level {depth}: {dirs_at_depth} directories, {files_at_depth} files", 33)
        
        # Print top directories at this level by file count
        sorted_dirs = sorted(by_depth[depth], key=lambda x: x[1]['file_count'], reverse=True)
        for i, (dir_path, stats) in enumerate(sorted_dirs[:5]):  # Top 5
            dir_name = dir_path if dir_path else '/'
            extensions = ", ".join(stats['file_extensions']) if stats['file_extensions'] else "none"
            print_colorized(f"  - {dir_name}: {stats['file_count']} files ({extensions})", 37)
            
        if len(sorted_dirs) > 5:
            print_colorized(f"  ... and {len(sorted_dirs) - 5} more directories at this level", 90)
    
    print()

def print_help() -> None:
    """Print help information with formatting"""
    print_header("🤖 ENHANCED CODEBASE ASSISTANT COMMANDS")
    
    commands = [
        ("scan <path>", "Scan and analyze a codebase directory"),
        ("summary", "Display project summary"),
        ("ls", "List files and directories in current location"),
        ("cd <directory>", "Change to specified directory"),
        ("pwd", "Show current directory path"),
        ("analyze <file_path>", "Analyze a specific file"),
        ("autoanalyze", "Automatically analyze important files"),
        ("search <query>", "Search for a pattern in the codebase"),
        ("context <query> <num>", "Search with context lines (e.g., 'context TODO 3')"),
        ("findfiles <pattern>", "Find files matching a pattern (e.g., '*.py')"),
        ("findhere <pattern>", "Find files in current directory (e.g., 'findhere *.py')"),
        ("viewfile <file_path>", "View the contents of a specific file"),
        ("models", "Analyze all model files in the codebase"),
        ("suggest <feature>", "Suggest implementation for a new feature"),
        ("pending", "List all pending changes"),
        ("details <change_id>", "Show details of a specific change"),
        ("approve <change_id>", "Approve and apply pending changes"),
        ("reject <change_id>", "Reject and discard pending changes"),
        ("scandir <dir>", "Scan all files in a directory and its subdirectories"),
        ("extension <ext>", "Find all files with a specific extension"),
        ("autoscan", "Automatically scan and analyze all nested directories"),
        ("help", "Display this help information"),
        ("quit", "Exit the application")
    ]
    
    for cmd, desc in commands:
        print_colorized(f"  {cmd.ljust(25)}", 36, end="")
        print_colorized(f"{desc}", 37)
    
    print()

def main():
    print_colorized("\n🤖 Welcome to Enhanced Codebase Assistant!", 36)
    print_colorized("A powerful tool to analyze, understand, and modify your codebase\n", 37)
    
    # Initialize the assistant
    assistant = CodebaseAssistant()
    
    print_help()
    
    # Command loop
    while True:
        try:
            current_dir = os.path.relpath(assistant.get_current_directory(), 
                                         assistant.codebase_root) if assistant.codebase_root else ""
            if current_dir == ".":
                current_dir = "/"
                
            prompt = f"\n[{current_dir}] > " if assistant.codebase_root else "\nCommand > "
            user_input = input(prompt).strip()
            
            if user_input.lower() == 'quit':
                print_colorized("\nThank you for using Enhanced Codebase Assistant! Goodbye! 👋", 36)
                break
                
            elif user_input.lower() == 'help':
                print_help()
                
            elif user_input.lower() == 'pwd':
                if assistant.codebase_root:
                    print_colorized(f"Current directory: {assistant.get_current_directory()}", 37)
                    rel_path = os.path.relpath(assistant.get_current_directory(), assistant.codebase_root)
                    if rel_path == ".":
                        rel_path = "/"
                    print_colorized(f"Relative to codebase root: {rel_path}", 37)
                else:
                    print_colorized("No codebase has been scanned yet.", 31)
                
            elif user_input.lower() == 'ls':
                if not assistant.codebase_root:
                    print_colorized("Please scan a codebase first using the scan command.", 31)
                    continue
                    
                dir_info = assistant.list_directory()
                print_directory_listing(dir_info)
                
            elif user_input.lower().startswith('cd '):
                if not assistant.codebase_root:
                    print_colorized("Please scan a codebase first using the scan command.", 31)
                    continue
                    
                directory = user_input[3:].strip()
                dir_info = assistant.change_directory(directory)
                
                if 'error' in dir_info:
                    print_colorized(f"❌ Error: {dir_info['error']}", 31)
                else:
                    print_directory_listing(dir_info)
                
            elif user_input.lower().startswith('scan '):
                path = user_input[5:].strip()
                
                if not os.path.exists(path):
                    print_colorized("❌ Invalid path. Please provide a valid directory path.", 31)
                    continue
                    
                print_colorized("\n📂 Scanning codebase... This may take a moment.", 36)
                result = assistant.scan_codebase(path)
                print_colorized(f"✅ Analyzed {result['files_analyzed']} files out of {result['total_files']} total files", 32)
                print_colorized(f"📊 Found {len(result['extensions_found'])} different file extensions", 32)
                print_colorized(f"📂 Maximum directory nesting depth: {result['max_depth']} levels", 32)
                
                # Display project summary after scan
                summary = assistant.get_project_summary()
                print_project_summary(summary)
                
                # Display the directory listing
                dir_info = assistant.list_directory()
                print_directory_listing(dir_info)
                
            elif user_input.lower() == 'summary':
                summary = assistant.get_project_summary()
                print_project_summary(summary)
                
            elif user_input.lower().startswith('analyze '):
                file_path = user_input[8:].strip()
                print_colorized(f"\n🔍 Analyzing file: {file_path}", 36)
                analysis = assistant.analyze_file(file_path)
                print_file_analysis(file_path, analysis)
                
            elif user_input.lower() == 'autoanalyze':
                print_colorized("\n🔍 Auto-analyzing key files in the codebase...", 36)
                results = assistant.auto_analyze()
                
                if 'error' in results:
                    print_colorized(f"❌ Error: {results['error']}", 31)
                else:
                    print_auto_analysis_results(results)
                
            elif user_input.lower().startswith('search '):
                query = user_input[7:].strip()
                print_colorized(f"\n🔍 Searching for: {query}", 36)
                results = assistant.search_in_codebase(query)
                print_search_results(results)
                
            elif user_input.lower().startswith('context '):
                # Format: context <query> <num_lines>
                parts = user_input[8:].strip().split()
                if len(parts) < 2:
                    print_colorized("❌ Usage: context <query> <num_lines>", 31)
                    continue
                    
                try:
                    query = parts[0]
                    context_lines = int(parts[1])
                    print_colorized(f"\n🔍 Searching for: {query} (with {context_lines} context lines)", 36)
                    results = assistant.search_in_codebase(query, context_lines)
                    print_search_results(results)
                except ValueError:
                    print_colorized("❌ Context lines must be a number", 31)
                
            elif user_input.lower().startswith('findfiles '):
                pattern = user_input[10:].strip()
                print_colorized(f"\n🔍 Finding files matching pattern: {pattern}", 36)
                results = assistant.find_files(pattern=pattern, recursive=True)
                print_file_list(results)
                
            elif user_input.lower().startswith('findhere '):
                pattern = user_input[9:].strip()
                print_colorized(f"\n🔍 Finding files in current directory matching: {pattern}", 36)
                results = assistant.find_files(pattern=pattern, in_current_dir=True, recursive=False)
                print_file_list(results)
                
            elif user_input.lower().startswith('viewfile '):
                file_path = user_input[9:].strip()
                print_colorized(f"\n📄 Viewing file: {file_path}", 36)
                file_data = assistant.get_file_content(file_path)
                print_file_content(file_data)
                
            elif user_input.lower() == 'models':
                print_colorized("\n🔍 Analyzing models...", 36)
                models_data = assistant.examine_models()
                
                if 'error' in models_data:
                    print_colorized(f"❌ Error: {models_data['error']}", 31)
                else:
                    print_models_analysis(models_data)
                
            elif user_input.lower().startswith('suggest '):
                feature = user_input[8:].strip()
                print_colorized(f"\n💡 Generating implementation suggestion for: {feature}", 36)
                print_colorized("This may take a moment...", 90)
                suggestion = assistant.suggest_feature_implementation(feature)
                
                if 'error' in suggestion:
                    print_colorized(f"❌ Error: {suggestion['error']}", 31)
                else:
                    print_feature_suggestion(suggestion)
                    
            elif user_input.lower() == 'pending':
                changes = assistant.list_pending_changes()
                print_pending_changes(changes)
                
            elif user_input.lower().startswith('details '):
                change_id = user_input[8:].strip()
                details = assistant.get_change_details(change_id)
                
                if 'error' in details:
                    print_colorized(f"❌ Error: {details['error']}", 31)
                else:
                    print_header(f"📋 CHANGE DETAILS: {change_id}")
                    print_colorized(f"Feature: {details['feature_description']}", 37)
                    print_colorized("\nSuggestion:", 36)
                    print_colorized(details['suggestion'], 37)
                    
                    # Show diff for files to modify
                    if details['proposed_changes']['files_to_modify']:
                        print_colorized("\nFiles to modify:", 36)
                        for file in details['proposed_changes']['files_to_modify']:
                            print_colorized(f"\n--- {file['path']} ---", 33)
                            print(file['diff'])
                    
                    # Show content for new files
                    if details['proposed_changes']['files_to_create']:
                        print_colorized("\nFiles to create:", 36)
                        for file in details['proposed_changes']['files_to_create']:
                            print_colorized(f"\n+++ {file['path']} +++", 32)
                            print(file['content'][:500] + "..." if len(file['content']) > 500 else file['content'])
                
            elif user_input.lower().startswith('approve '):
                change_id = user_input[8:].strip()
                print_colorized(f"\n⚠️ Applying changes for {change_id}...", 33)
                result = assistant.approve_changes(change_id)
                print_change_result(result)
                
            elif user_input.lower().startswith('reject '):
                change_id = user_input[7:].strip()
                result = assistant.reject_changes(change_id)
                
                if 'error' in result:
                    print_colorized(f"❌ Error: {result['error']}", 31)
                else:
                    print_colorized(f"✅ {result['message']}", 32)
                    
            elif user_input.lower().startswith('scandir '):
                target_dir = user_input[8:].strip()
                print_colorized(f"\n📂 Scanning directory: {target_dir}", 36)
                results = assistant.scan_nested_directories(target_dir)
                print_nested_scan_results(results)
                
            elif user_input.lower().startswith('extension '):
                extension = user_input[10:].strip()
                print_colorized(f"\n🔍 Finding all .{extension} files...", 36)
                results = assistant.find_all_files_by_extension(extension)
                print_extensions_report(results)
                
            elif user_input.lower() == 'autoscan':
                print_colorized("\n📂 Auto-scanning all nested directories...", 36)
                results = assistant.auto_scan_nested_directories()
                print_auto_scan_results(results)
            
            else:
                # Treat as a question for the codebase chat
                print_colorized("\n🤖 Assistant: ", 36, end='')
                response = assistant.chat_with_codebase(user_input)
                print_colorized(response, 37)
                
        except KeyboardInterrupt:
            print_colorized("\n\nOperation cancelled by user.", 33)
            continue
            
        except Exception as e:
            print_colorized(f"\n❌ Error: {str(e)}", 31)
            # Print the exception traceback for debugging
            import traceback
            print_colorized(traceback.format_exc(), 31)

if __name__ == "__main__":
    main()