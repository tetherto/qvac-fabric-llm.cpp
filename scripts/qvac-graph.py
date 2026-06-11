from abc import ABC, abstractmethod
from io import StringIO
from itertools import product
from pathlib import Path
import argparse
import matplotlib.pyplot as plt
import pandas as pd
import re


def branch_sort_key(branch: str) -> tuple[int, int, str]:
    m = re.search(r'\d+', branch)
    if branch.startswith("upstream"):
        rank = 1
    elif branch.startswith("qvac"):
        rank = 2
    else:
        rank = 999

    return (int(m.group()) if m else 2**32, rank, branch)


class Report(ABC):

    @classmethod
    @abstractmethod
    def name(cls) -> str:
        pass

    @classmethod
    @abstractmethod
    def detect(cls, text: str) -> bool:
        pass

    @classmethod
    def plot(cls, df: pd.DataFrame, outpath: Path,
             groups: list[list[str]]) -> None:
        pass


class LlamaBenchReport(Report):

    @classmethod
    def name(cls) -> str:
        return "llama-bench"

    @classmethod
    def detect(cls, text: str) -> bool:
        if not '- **llama-bench**' in text or not '| Model ' in text:
            return False

        return True

    @classmethod
    def plot(cls, df: pd.DataFrame, outpath: Path, groups: list[list[str]]) -> None:
        branch_cols = sorted(
            df.columns[~df.columns.isin(["model", "metric", "device"])],
            key=branch_sort_key)

        for model in df["model"].unique():
            outfile = outpath / (model + ".pdf")
            print(f"Plotting graphs for model '{model}' -> {outfile}")
            model_df = df[df["model"] == model]

            metrics = model_df["metric"].unique()

            fig, axes = plt.subplots(len(groups) * len(metrics), 1, figsize=(8.27, 11.69), squeeze=False)
            fig.suptitle(f"{model} Inference", fontsize=16)

            for i, (group, metric) in enumerate(product(groups, metrics), start=1):
                ax = axes[i - 1, 0]
                metric_df = model_df[(model_df["metric"] == metric) & (model_df["device"].isin(group))]

                for device in metric_df["device"].unique():
                    device_df = metric_df[metric_df["device"] == device]

                    def clean_value(val):
                        if not isinstance(val, str):
                            return [0.0, 0.0]
                        else:
                            return [float(f) for f in val.strip().replace("±", " ").split()[:2]]

                    vals = device_df[branch_cols].fillna(0.0).values.flatten()
                    ys, yerrs = zip(*(clean_value(v) for v in vals))
                    if len(ys):
                        line = ax.plot(branch_cols, ys, marker="o", label=device)
                        ax.errorbar(branch_cols, ys, yerr=yerrs, fmt='none', capsize=5, color=line[0].get_color())

                if len(ax.get_lines()) > 0:
                    ax.set_title(f"{metric}")
                    ax.set_xlabel("Branch")
                    ax.set_ylabel("Tokens/sec")
                    ax.grid(True, linestyle="--", alpha=0.5)
                    ymax = max(line.get_ydata().max() for line in ax.get_lines())
                    ax.set_ylim(bottom=0, top=ymax * 1.1)
                    ax.legend()
                else:
                    ax.remove()

            outpath.mkdir(exist_ok=True)
            plt.tight_layout()
            plt.savefig(outfile)
            plt.close()

    def __init__(self, text: str, device: str):
        data = text[text.find("| Model "):]
        df = pd.read_csv(StringIO(data), sep="|", skiprows=[1])
        df = df.iloc[:, 1:-1]
        df.columns = df.columns.str.replace("t/s", "")
        df.columns = df.columns.str.replace("-vulkan", "")
        df.columns = df.columns.str.replace("-opencl", "").str.strip().str.lower()
        df = df.apply(lambda col: col.map(lambda x: x.strip() if isinstance(x, str) else x))
        df["device"] = device

        self.df = df


class LlamaFinetuneLoraReport(Report):

    @classmethod
    def name(cls) -> str:
        return "llama-finetune-lora"

    @classmethod
    def detect(cls, text: str) -> bool:
        if not '- **Loss tolerance' in text or not '| Model | ' in text or not 'epoch_s' in text:
            return False

        return True

    @classmethod
    def plot(cls, df: pd.DataFrame, outpath: Path, groups: list[list[str]]) -> None:
        branch_cols = sorted(
            df.columns[~df.columns.str.contains("model|loss|verdict|device")],
            key=branch_sort_key)

        for model in df["model"].unique():
            outfile = outpath / (model + ".pdf")
            print(f"Plotting graphs for model '{model}' -> {outfile}")
            model_df = df[df["model"] == model]

            fig, axes = plt.subplots(len(groups), 1, figsize=(8.27, 11.69), squeeze=False)
            fig.suptitle(f"{model} LoRA Finetuning", fontsize=16)

            for i, group in enumerate(groups, start=1):
                ax = axes[i - 1, 0]
                metric_df = model_df[model_df["device"].isin(group)]

                for device in metric_df["device"].unique():
                    device_df = metric_df[metric_df["device"] == device]
                    vals = device_df[branch_cols].fillna(0.0).values.flatten()
                    line = ax.plot(branch_cols, vals, marker="o", label=device)

                if len(ax.get_lines()) > 0:
                    ax.set_title("Epoch time")
                    ax.set_xlabel("Branch")
                    ax.set_ylabel("sec")
                    ax.grid(True, linestyle="--", alpha=0.5)
                    ymax = max(line.get_ydata().max() for line in ax.get_lines())
                    ax.set_ylim(bottom=0, top=ymax * 1.1)
                    ax.legend()
                else:
                    ax.remove()

            outpath.mkdir(exist_ok=True)
            plt.tight_layout()
            plt.savefig(outfile)
            plt.close()

    def __init__(self, text: str, device: str):
        data = text[text.find("| Model | "):]
        df = pd.read_csv(StringIO(data), sep="|", skiprows=[1])
        df = df.iloc[:, 1:-1]
        df.columns = df.columns.str.replace("epoch_s", "")
        df.columns = df.columns.str.replace("-vulkan", "")
        df.columns = df.columns.str.replace("-opencl", "").str.strip().str.lower()
        df = df.apply(lambda col: col.map(lambda x: x.strip() if isinstance(x, str) else x))
        df["device"] = device

        self.df = df


report_types = [LlamaBenchReport, LlamaFinetuneLoraReport]


def parse_report(text, device):
    for report_cls in report_types:
        if report_cls.detect(text):
            return report_cls(text, device)
    return None


def build_parser():
    p = argparse.ArgumentParser()

    p.add_argument("--reports",
                   "-r",
                   nargs="*",
                   required=True,
                   help="Paths to reports")
    p.add_argument("--devices",
                   "-d",
                   nargs="*",
                   action="append",
                   required=True,
                   help="Devices for each corresponding report (can be specified multiple times to group devices in graphs)")
    p.add_argument("--output-dir",
                   "-o",
                   required=True,
                   help="Directory in which to save the output graphs")
    p.add_argument("--models",
                   "-m",
                   nargs="*",
                   help="Models to produce graphs for (all models by default)")

    return p


def main() -> None:
    parser = build_parser()
    args = parser.parse_args()

    flat_devices = [item for sublist in args.devices for item in sublist] if args.devices else []
    if not args.reports or len(args.reports) != len(flat_devices):
        print(
            "Reports and devices must be provided and must have the same length."
        )
        return

    def read_file(path):
        with open(path, "r") as f:
            return f.read()

    report_texts = [read_file(path) for path in args.reports]
    reports = [
        parse_report(text, device)
        for text, device in zip(report_texts, flat_devices)
    ]

    for path, report in zip(args.reports, reports):
        if report is None:
            raise RuntimeError(
                f"Could not detect report type for file: {path}")

    if any(r.name() != reports[0].name() for r in reports):
        raise RuntimeError(
            f"All reports must be of the same type but got {[r.name() for r in reports]}"
        )

    combined_df = pd.concat([r.df for r in reports], ignore_index=True)
    if args.models:
        combined_df = combined_df[combined_df["model"].str.startswith(tuple(args.models), na=False)]
    reports[0].plot(combined_df, Path(args.output_dir), args.devices)


if __name__ == "__main__":
    try:
        main()
    except Exception as e:
        print(f"Error: {e}")
