import os
import sys

try:
    import pyperclip
except ImportError:
    print("Error: The 'pyperclip' module is not installed.")
    print("Please install it by running: pip install pyperclip")
    sys.exit(1)

# Configuration - Kept completely original
ALLOWED_EXTENSIONS = {'.py', '.js', '.ts', '.tsx', '.rs', '.json', '.html', '.md', '.css', 'Makefile', 'Cargo.toml', '.sh', 'Dockerfile', '.yml', '.c', '.h'}
IGNORE_DIRS = {'node_modules', 'onnxruntime', 'dist', 'pkg', 'target', '.git', '__pycache__', 'wandb'}
IGNORE_FILES = {'package-lock.json', 'yarn.lock', 'pnpm-lock.yaml'}

def is_text_file(filename):
    return any(filename.endswith(ext) for ext in ALLOWED_EXTENSIONS)

def parse_args(args):
    """Separates standard filters from type filters based on -t / --types flags."""
    standard_filters = []
    type_filters = []
    i = 0
    while i < len(args):
        if args[i] in ('-t', '--types'):
            i += 1
            # Gather all arguments after the flag until another flag is hit
            while i < len(args) and not args[i].startswith('-'):
                type_filters.append(args[i])
                i += 1
        else:
            standard_filters.append(args[i])
            i += 1
    return standard_filters, type_filters

def generate_context(standard_filters, type_filters, start_path='.'):
    output_lines = []
    file_count = 0

    # Pre-lowercase filters for case-insensitive matching
    s_filters = [term.lower() for term in standard_filters]
    t_filters = [term.lower().rstrip('/') for term in type_filters]

    for root, dirs, files in os.walk(start_path):
        # Determine if we should allow 'dist' traversal for this specific root
        keep_dist = False
        if 'dist' in dirs and t_filters:
            root_normalized = root.lower().replace('\\', '/').rstrip('/')
            for t in t_filters:
                # Matches if the folder is exactly the target or ends with /target (e.g., packages/core)
                if root_normalized == t or root_normalized.endswith(f"/{t}") or root_normalized == f"./{t}":
                    keep_dist = True
                    break

        # Modify dirs in-place to skip ignored directories, unless it's an approved 'dist'
        dirs[:] = [d for d in dirs if not d.endswith('.0x') and (d not in IGNORE_DIRS or (d == 'dist' and keep_dist))]

        for file in files:
            file_path = os.path.join(root, file)
            rel_path = os.path.relpath(file_path, start_path)
            path_lower = rel_path.lower().replace('\\', '/')
            
            # 1. Check if this specific file is a requested type file
            is_requested_type = False
            if file.lower() == 'index.d.ts':
                for t in t_filters:
                    if f"{t}/dist/index.d.ts" in path_lower:
                        is_requested_type = True
                        break
            
            # 2. Guard: If we are inside ANY dist folder, ignore it UNLESS it's our requested type file
            if '/dist/' in path_lower or path_lower.startswith('dist/'):
                if not is_requested_type:
                    continue
            
            # 3. Standard filtering for all other files
            if not is_requested_type:
                # Must be a valid text file and not ignored
                if not is_text_file(file) or file in IGNORE_FILES:
                    continue
                # Must match standard filters if any exist
                if s_filters:
                    if not any(term in path_lower for term in s_filters):
                        continue

            try:
                with open(file_path, 'r', encoding='utf-8', errors='ignore') as f:
                    content = f.read()
                    
                output_lines.append(f"### File: `{rel_path}`")
                output_lines.append("```")
                output_lines.append(content)
                output_lines.append("```")
                output_lines.append("-" * 10) 
                output_lines.append("") 
                
                # Visual indicator if it was pulled in via the --types flag
                if is_requested_type:
                    print(f"✅ Added {rel_path} to context (Type Def).")
                else:
                    print(f"✅ Added {rel_path} to context.")
                file_count += 1
            except Exception as e:
                print(f"Skipping {rel_path}: {e}")

    return "\n".join(output_lines), file_count

if __name__ == "__main__":
    args = sys.argv[1:]
    standard_filters, type_filters = parse_args(args)
    
    # Dynamic status message based on what arguments were provided
    if standard_filters and type_filters:
        print(f"Scanning for paths containing: {', '.join(standard_filters)} | And Types for: {', '.join(type_filters)}...")
    elif standard_filters:
        print(f"Scanning repository for files containing: {', '.join(standard_filters)}...")
    elif type_filters:
        print(f"Scanning repository exclusively for type definitions of: {', '.join(type_filters)}...")
    else:
        print("Scanning repository (no filters applied)...")

    context_string, count = generate_context(standard_filters, type_filters)
    
    if count > 0:
        pyperclip.copy(context_string)
        print(f"✅ Success! Copied {count} files to clipboard.")
    else:
        print("⚠️ No matching files found.")