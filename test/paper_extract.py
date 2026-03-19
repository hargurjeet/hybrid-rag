import json

input_file = "dataset/arxiv_10k.json"
output_file = "evaluation_seed_papers.json"

papers = []

with open(input_file, "r") as f:
    for i, line in enumerate(f):
        if i >= 50:
            break

        record = json.loads(line)

        papers.append({
            "paper_id": record["id"],
            "title": record["title"],
            "abstract": record["abstract"].strip()
        })

with open(output_file, "w") as f:
    json.dump(papers, f, indent=2)

print(f"Saved {len(papers)} papers to {output_file}")