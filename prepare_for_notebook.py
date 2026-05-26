import os

def consolidate_code(output_file="project_context.txt"):
    # Files/folders to ignore
    ignore = {'.git', '__pycache__', 'venv', '.ipynb_checkpoints', output_file}
    
    with open(output_file, 'w', encoding='utf-8') as outfile:
        for root, dirs, files in os.walk('.'):
            # Modify dirs in-place to skip ignored directories
            dirs[:] = [d for d in dirs if d not in ignore]
            
            for file in files:
                if file.endswith('.py') and file not in ignore:
                    file_path = os.path.join(root, file)
                    outfile.write(f"\n\n--- FILE: {file_path} ---\n\n")
                    with open(file_path, 'r', encoding='utf-8') as infile:
                        outfile.write(infile.read())
    
    print(f"Project consolidated into {output_file}")

if __name__ == "__main__":
    consolidate_code()