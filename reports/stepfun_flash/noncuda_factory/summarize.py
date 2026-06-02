import json

def main():
    rows = []
    with open("bench.jsonl") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            rows.append(json.loads(line))

    rows.sort(key=lambda r: r["tps"], reverse=True)
    max_tps = rows[0]["tps"]

    header = "| config | tps | rel_to_best |"
    sep    = "|--------|-----|-------------|"
    lines  = [header, sep]

    for r in rows:
        rel = r["tps"] / max_tps
        lines.append(f"| {r['config']} | {r['tps']} | {rel:.3f} |")

    table = "\n".join(lines)
    print(table)

    with open("summary.md", "w") as f:
        f.write(table + "\n")

if __name__ == "__main__":
    main()
