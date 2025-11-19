import os

def create_project_structure(base_dir="tiny_llm"):
    """
    Creates the basic project structure for a tiny LLM project.
    """
    structure = [
        "data/raw",
        "data/processed",
        "models",
        "scripts",
        "tinyllm",
    ]

    for path in structure:
        full_path = os.path.join(base_dir, path)
        os.makedirs(full_path, exist_ok=True)
        print(f"Created: {full_path}")

    # Create placeholder files
    open(os.path.join(base_dir, "tinyllm", "__init__.py"), "w").close()
    open(os.path.join(base_dir, "scripts", "train.py"), "w").close()
    open(os.path.join(base_dir, "scripts", "generate.py"), "w").close()

    print("\nProject initialized successfully.")

if __name__ == "__main__":
    create_project_structure()
