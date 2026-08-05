import os

chunks = []

for filename in os.listdir("data"):
    if filename.endswith(".md"):
        text = open(f"data/{filename}").read()
        chunks.append({"source": filename, "text": text})

print(len(chunks), "chunks loaded from data folder.")
print(chunks[0])