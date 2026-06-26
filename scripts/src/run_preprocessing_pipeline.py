from hydrologic_pipeline.pipeline import run_pipeline


def main() -> None:
    artifacts = run_pipeline()
    print("Pipeline done")
    print(f"Raw rows: {len(artifacts.raw_flattened)}")
    for node_id, node_df in artifacts.cleaned_by_node.items():
        hourly_rows = len(artifacts.hourly_by_node[node_id])
        print(f"{node_id} cleaned rows: {len(node_df)}")
        print(f"{node_id} hourly rows: {hourly_rows}")


if __name__ == "__main__":
    main()
