from f1_podium.reporting import generate_reports, write_metrics


if __name__ == "__main__":
    for output in [*generate_reports(), write_metrics()]:
        print(f"Wrote {output}")

