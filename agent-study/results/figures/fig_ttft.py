"""Generate the paper's later-turn TTFT figure and its source-linked values."""

import json
from pathlib import Path

import matplotlib as mpl
from matplotlib.backends.backend_pdf import FigureCanvasPdf
from matplotlib.figure import Figure


OUTPUT = Path(__file__).resolve().parent
STUDY = OUTPUT.parent.parent
SESSIONS = (8, 16, 24, 32, 48)
SWEEPS = {
    "C1": "results/C1/sweep-20261126T202631.742740Z.json",
    "C2": "results/C2/sweep-20261127T072909.617020Z.json",
    "C3": "results/C3/sweep-20261128T181103.552840Z.json",
    "C4": "results/C4/sweep-20261128T184543.363766Z.json",
}
STYLES = {
    "C1": ("#333333", "o", "-"),
    "C2": ("#0072B2", "s", "--"),
    "C3": ("#008064", "^", "-."),
    "C4": ("#C45100", "D", ":"),
}


def main():
    records = []
    with mpl.rc_context({"font.size": 9, "pdf.fonttype": 42}):
        figure = Figure(figsize=(5, 3), layout="constrained")
        FigureCanvasPdf(figure)
        ax = figure.subplots()
        for config, sweep_path in SWEEPS.items():
            sources = json.loads((STUDY / sweep_path).read_text())["results"]
            values = []
            for sessions in SESSIONS:
                source = sources[str(sessions)]
                value = json.loads((STUDY / source).read_text())["later_p95"]
                records.append(dict(config=config, sessions=sessions, later_p95=value, source=source))
                values.append(value)
            color, marker, style = STYLES[config]
            ax.plot(SESSIONS, values, label=config, color=color, linestyle=style,
                    marker=marker, markersize=4, markerfacecolor="none", linewidth=1.4)

        ax.set(xlabel="Sessions", ylabel="Later-turn p95 TTFT, s", xticks=SESSIONS,
               ylim=(0, 56), yticks=range(0, 51, 10))
        ax.spines[["right", "top"]].set_visible(False)
        ax.set_axisbelow(True)
        ax.grid(axis="y", color="#dddddd", linewidth=0.5)
        ax.legend(loc="upper left", frameon=False, ncols=2)
        figure.savefig(OUTPUT / "fig_ttft.pdf", metadata={"CreationDate": None})

    (OUTPUT / "fig_ttft_values.json").write_text(json.dumps(records, indent=2) + "\n")


if __name__ == "__main__":
    main()
