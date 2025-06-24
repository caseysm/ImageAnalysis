#!/usr/bin/env python3
"""Generate a summary of the project structure."""

import os
from pathlib import Path
import json
import datetime

def count_files_by_extension():
    """Count files by extension."""
    extension_counts = {}
    total_files = 0
    python_files = 0
    
    for root, _, files in os.walk('.'):
        # Skip hidden directories and results directory
        if '/.' in root or root.startswith('./.') or 'results' in root:
            continue
            
        for file in files:
            # Skip hidden files
            if file.startswith('.'):
                continue
                
            total_files += 1
            ext = os.path.splitext(file)[1].lower()
            
            if ext == '.py':
                python_files += 1
                
            extension_counts[ext] = extension_counts.get(ext, 0) + 1
    
    return extension_counts, total_files, python_files

def count_directories():
    """Count directories."""
    count = 0
    
    for root, dirs, _ in os.walk('.'):
        # Skip hidden directories and results directory
        if '/.' in root or root.startswith('./.') or 'results' in root:
            continue
            
        # Skip hidden directories in the count
        dirs[:] = [d for d in dirs if not d.startswith('.') and d != 'results']
        count += len(dirs)
    
    return count

def get_directory_structure(directory, max_depth=3, current_depth=0):
    """Generate a tree of the directory structure."""
    if current_depth > max_depth:
        return "    " * current_depth + "...\n"
        
    result = ""
    
    try:
        entries = sorted(os.listdir(directory))
    except PermissionError:
        return result
        
    dirs = []
    files = []
    
    for entry in entries:
        # Skip hidden files/dirs and results directory
        if entry.startswith('.') or entry == 'results':
            continue
            
        full_path = os.path.join(directory, entry)
        
        if os.path.isdir(full_path):
            dirs.append(entry)
        else:
            files.append(entry)
    
    # Process directories
    for i, d in enumerate(dirs):
        is_last_dir = i == len(dirs) - 1
        if is_last_dir and not files:
            prefix = "    " * current_depth + "└── "
            new_prefix = "    " * (current_depth + 1)
        else:
            prefix = "    " * current_depth + "├── "
            new_prefix = "    " * current_depth + "│   "
            
        result += prefix + d + "/\n"
        result += get_directory_structure(
            os.path.join(directory, d),
            max_depth,
            current_depth + 1
        )
    
    # Process files
    for i, f in enumerate(files):
        is_last = i == len(files) - 1
        if is_last:
            prefix = "    " * current_depth + "└── "
        else:
            prefix = "    " * current_depth + "├── "
            
        result += prefix + f + "\n"
    
    return result

def get_package_modules():
    """List all modules in the package."""
    modules = []
    package_dir = "./imageanalysis"
    
    if not os.path.exists(package_dir):
        return modules
        
    for root, dirs, files in os.walk(package_dir):
        # Skip hidden directories
        dirs[:] = [d for d in dirs if not d.startswith('.')]
        
        # Check if directory is a Python package
        if "__init__.py" in files:
            module_path = os.path.relpath(root, ".")
            module_path = module_path.replace(os.path.sep, '.')
            modules.append(module_path)
    
    return sorted(modules)

def main():
    """Main function."""
    print("ImageAnalysis Project Summary")
    print("============================")
    print(f"Generated: {datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print()
    
    # Count files
    extension_counts, total_files, python_files = count_files_by_extension()
    
    # Count directories
    dir_count = count_directories()
    
    # Statistics
    print("Statistics:")
    print(f"  Total Files: {total_files}")
    print(f"  Python Files: {python_files}")
    print(f"  Directories: {dir_count}")
    print()
    
    # File types
    print("File Types:")
    for ext, count in sorted(extension_counts.items(), key=lambda x: x[1], reverse=True)[:10]:
        if ext:
            print(f"  {ext}: {count}")
        else:
            print(f"  No extension: {count}")
    print()
    
    # Package modules
    modules = get_package_modules()
    print("Package Modules:")
    for module in modules:
        print(f"  {module}")
    print()
    
    # Directory structure
    print("Directory Structure:")
    print(get_directory_structure(".", max_depth=3))
    
    # Documentation files
    docs = [f for f in os.listdir(".") if f.endswith(".md")]
    print("Documentation Files:")
    for doc in sorted(docs):
        print(f"  {doc}")
    print()
    
    # Create report
    report = {
        "generated": datetime.datetime.now().isoformat(),
        "statistics": {
            "total_files": total_files,
            "python_files": python_files,
            "directories": dir_count,
            "file_types": extension_counts
        },
        "modules": modules,
        "documentation": docs
    }
    
    with open("project_summary.json", "w") as f:
        json.dump(report, f, indent=2)
    
    print("JSON report saved to project_summary.json")

if __name__ == "__main__":
    main()