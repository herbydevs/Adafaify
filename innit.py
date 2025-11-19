import os

# Directories for tiny LLM project
DIRECTORIES = [
    'data/raw',
    'data/processed',
    'src/model',
    'src/tokenization',
    'src/training',
    'src/generation',
    'src/utils',
    'experiments',
    'notebooks',
    'checkpoints',
    'configs'
]

# Files to create with optional starter content
FILES = {
    'README.md': '# Tiny LLM Project\nThis project contains a tiny transformer-based language model built from scratch.\n',
    'requirements.txt': 'torch\ntqdm\n',
    'configs/tiny.yaml': 'model:\n  dim: 128\n  layers: 4\n  heads: 4\n  vocab_size: 100\n  seq_len: 128\n\ntraining:\n  lr: 0.0002\n  batch_size: 64\n  epochs: 5\n',
    'src/__init__.py': '',
    'src/model/__init__.py': '',
    'src/tokenization/__init__.py': '',
    'src/training/__init__.py': '',
    'src/generation/__init__.py': '',
    'src/utils/__init__.py': '',
}


def create_directories():
    for directory in DIRECTORIES:
        os.makedirs(directory, exist_ok=True)
        print(f"Created directory: {directory}")


def create_files():
    for filepath, content in FILES.items():
        with open(filepath, 'w') as f:
            f.write(content)
        print(f"Created file: {filepath}")


def main():
    create_directories()
    create_files()
    print("\nProject structure initialized successfully.")


if __name__ == '__main__':
    main()
