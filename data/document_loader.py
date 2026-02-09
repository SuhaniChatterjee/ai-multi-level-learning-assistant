import os


def load_documents(folder_path="data/raw_docs"):
    """
    Loads all .txt documents from a folder
    and returns a list of raw text strings.
    """
    documents = []

    for filename in os.listdir(folder_path):
        if filename.endswith(".txt"):
            file_path = os.path.join(folder_path, filename)
            with open(file_path, "r", encoding="utf-8") as f:
                documents.append(f.read())

    return documents
