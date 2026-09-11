#!/usr/bin/env python3
"""Monitor SSM (Mamba) pool metrics from a running SGLang server.

Polls /v1/loads and prints live SSM pool utilization and memory usage.
On exit (Ctrl+C), prints a summary with max memory and final cache hit rate.

Usage:
    python monitor_mamba.py --host localhost --port 30000 --interval 2
"""

import argparse
import csv
import json
import signal
import sys
import time
import urllib.request
import urllib.error


def fetch_loads(base_url: str) -> dict:
    url = f"{base_url}/v1/loads?include=all"
    req = urllib.request.Request(url)
    with urllib.request.urlopen(req, timeout=5) as resp:
        return json.loads(resp.read().decode())


def main():
    parser = argparse.ArgumentParser(description="Monitor SGLang Mamba pool metrics")
    parser.add_argument("--host", default="localhost")
    parser.add_argument("--port", type=int, default=30000)
    parser.add_argument("--interval", type=float, default=2.0, help="Poll interval in seconds")
    parser.add_argument(
        "--output", "-o", default="mamba_monitor.csv",
        help="CSV file to save polled metrics (default: mamba_monitor.csv)",
    )
    args = parser.parse_args()

    base_url = f"http://{args.host}:{args.port}"

    # Open CSV for writing
    csv_file = open(args.output, "w", newline="")
    csv_writer = csv.writer(csv_file)
    csv_writer.writerow([
        "timestamp", "elapsed_s", "dp_rank", "cache_hit_rate", "token_usage",
        "ssm_memory_gb", "ssm_slots_used", "ssm_slots_total", "ssm_utilization",
    ])
    start_time = None

    # Tracking state
    max_memory_gb = 0.0
    max_slots_used = 0
    max_utilization = 0.0
    sample_count = 0
    last_data = None
    stopped = False

    def print_summary(signum=None, frame=None):
        nonlocal stopped
        if stopped:
            return
        stopped = True
        print("\n" + "=" * 60)
        print("SUMMARY")
        print("=" * 60)
        print(f"  Samples collected:       {sample_count}")
        print(f"  Max SSM memory used:     {max_memory_gb:.3f} GB")
        print(f"  Max SSM slots used:      {max_slots_used}")
        print(f"  Max SSM utilization:     {max_utilization:.4f}")
        if last_data:
            dp = last_data if "cache_hit_rate" in last_data else last_data.get("dp_loads", [{}])[0]
            hit_rate = dp.get("cache_hit_rate", "N/A")
            print(f"  Final cache hit rate:    {hit_rate}")
            mamba = dp.get("mamba")
            if mamba:
                print(f"  Final SSM memory used:   {mamba['pool_memory_used_gb']:.3f} GB")
                print(f"  Final SSM slots used:    {mamba['pool_slots_used']} / {mamba['pool_slots_total']}")
                print(f"  Final SSM utilization:   {mamba['pool_utilization']:.4f}")
            else:
                print("  Final SSM metrics:       N/A (not a hybrid SSM model?)")
        print(f"  Metrics saved to:        {args.output}")
        print("=" * 60)
        csv_file.close()
        sys.exit(0)

    signal.signal(signal.SIGINT, print_summary)
    signal.signal(signal.SIGTERM, print_summary)

    print(f"Monitoring {base_url} every {args.interval}s (Ctrl+C to stop and show summary)")
    print("-" * 60)

    # Wait for server to be ready
    while not stopped:
        try:
            fetch_loads(base_url)
            break
        except (urllib.error.URLError, ConnectionRefusedError, OSError):
            print("Waiting for server...", end="\r")
            time.sleep(args.interval)

    while not stopped:
        try:
            data = fetch_loads(base_url)
            last_data = data

            # Handle both single-dp and multi-dp responses
            if "dp_loads" in data:
                dp_loads = data["dp_loads"]
            else:
                dp_loads = [data]

            for dp in dp_loads:
                sample_count += 1
                dp_rank = dp.get("dp_rank", 0)
                cache_hit_rate = dp.get("cache_hit_rate", 0.0)
                token_usage = dp.get("token_usage", 0.0)
                mamba = dp.get("mamba")

                parts = [
                    f"[{time.strftime('%H:%M:%S')}]",
                    f"dp={dp_rank}",
                    f"cache_hit={cache_hit_rate:.4f}",
                    f"token_usage={token_usage:.4f}",
                ]

                now = time.time()
                if start_time is None:
                    start_time = now

                if mamba:
                    mem_gb = mamba["pool_memory_used_gb"]
                    slots_used = mamba["pool_slots_used"]
                    slots_total = mamba["pool_slots_total"]
                    utilization = mamba["pool_utilization"]

                    max_memory_gb = max(max_memory_gb, mem_gb)
                    max_slots_used = max(max_slots_used, slots_used)
                    max_utilization = max(max_utilization, utilization)

                    parts += [
                        f"ssm_mem={mem_gb:.3f}GB",
                        f"ssm_slots={slots_used}/{slots_total}",
                        f"ssm_util={utilization:.4f}",
                    ]
                else:
                    mem_gb = slots_used = slots_total = utilization = ""

                print("  ".join(parts))

                csv_writer.writerow([
                    time.strftime("%Y-%m-%d %H:%M:%S"),
                    f"{now - start_time:.1f}",
                    dp_rank, cache_hit_rate, token_usage,
                    mem_gb, slots_used, slots_total, utilization,
                ])
                csv_file.flush()

        except (urllib.error.URLError, ConnectionRefusedError, OSError) as e:
            print(f"[{time.strftime('%H:%M:%S')}]  Connection error: {e}")
        except json.JSONDecodeError as e:
            print(f"[{time.strftime('%H:%M:%S')}]  Invalid response: {e}")

        time.sleep(args.interval)


if __name__ == "__main__":
    main()
