#!/usr/bin/env python3
from __future__ import annotations

import json

from openclaw_algae.verification.stream_probe import run_live_stream_regression, write_regression_report


def main() -> None:
    results = run_live_stream_regression()
    for result in results:
        print(
            json.dumps(
                {
                    "model": result["model"],
                    "endpoint": result["endpoint"],
                    "class": result["class"],
                    "first_visible_s": result["first_visible_s"],
                    "visible_chunks": result["visible_chunks"],
                    "window_5_s": result["window_5_s"],
                    "error": result["error"],
                },
                ensure_ascii=False,
            ),
            flush=True,
        )
    summary: dict[str, int] = {}
    for item in results:
        summary[item["class"]] = summary.get(item["class"], 0) + 1
    out_path = write_regression_report(results)
    print("SUMMARY", json.dumps(summary, ensure_ascii=False, sort_keys=True))
    print("WROTE", out_path)


if __name__ == "__main__":
    main()
