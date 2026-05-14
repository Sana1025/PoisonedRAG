
import sys
import traceback

from poison_retrieval_experiment import run_experiment  

import matplotlib 
matplotlib.use("Qt5Agg")
from matplotlib.backends.backend_qt5agg import FigureCanvasQTAgg as FigureCanvas  
from matplotlib.figure import Figure  

from PyQt5.QtCore import Qt, QThread, pyqtSignal  
from PyQt5.QtGui import QColor, QFont  
from PyQt5.QtWidgets import (  
    QApplication, QHBoxLayout, QHeaderView, QLabel, QMainWindow, QPlainTextEdit,
    QPushButton, QStatusBar, QTabWidget, QTableWidget, QTableWidgetItem,
    QVBoxLayout, QWidget,
)


POISON_BG = QColor(255, 220, 220)


class ExperimentWorker(QThread):
    progress = pyqtSignal(str)
    finished_ok = pyqtSignal(dict)
    failed = pyqtSignal(str)

    def run(self):
        try:
            res = run_experiment(progress=self.progress.emit)
            self.finished_ok.emit(res)
        except Exception:
            self.failed.emit(traceback.format_exc())


class MainWindow(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("PoisonedRAG Toy Experiment")
        self.resize(1100, 750)

        central = QWidget()
        self.setCentralWidget(central)
        root = QVBoxLayout(central)

   
        top = QHBoxLayout()
        self.run_btn = QPushButton("Run Experiment")
        self.run_btn.clicked.connect(self.start_experiment)
        top.addWidget(self.run_btn)
        self.status_label = QLabel("Idle. Click Run.")
        top.addWidget(self.status_label, stretch=1)
        root.addLayout(top)

        self.setup_box = QPlainTextEdit()
        self.setup_box.setReadOnly(True)
        self.setup_box.setMaximumHeight(140)
        mono = QFont("Consolas")
        mono.setStyleHint(QFont.Monospace)
        self.setup_box.setFont(mono)
        root.addWidget(self.setup_box)

        self.tabs = QTabWidget()
        root.addWidget(self.tabs, stretch=1)

        self.summary_tbl = QTableWidget()
        self.f1_tbl = QTableWidget()
        self.per_query_tbl = QTableWidget()

        for tbl in (self.summary_tbl, self.f1_tbl, self.per_query_tbl):
            tbl.setAlternatingRowColors(True)
            tbl.horizontalHeader().setSectionResizeMode(QHeaderView.ResizeToContents)
            tbl.verticalHeader().setVisible(False)
            tbl.setEditTriggers(QTableWidget.NoEditTriggers)

        self.viz_widget = QWidget()
        viz_layout = QVBoxLayout(self.viz_widget)
        self.fig_topk = Figure(figsize=(8, 3.2), tight_layout=True)
        self.canvas_topk = FigureCanvas(self.fig_topk)
        self.fig_rank = Figure(figsize=(8, 3.8), tight_layout=True)
        self.canvas_rank = FigureCanvas(self.fig_rank)
        viz_layout.addWidget(self.canvas_topk)
        viz_layout.addWidget(self.canvas_rank)

        self.tabs.addTab(self.summary_tbl, "Summary (top-k hit rate)")
        self.tabs.addTab(self.f1_tbl, "Malicious retrieval F1 @ k=5")
        self.tabs.addTab(self.per_query_tbl, "Per-query top-5")
        self.tabs.addTab(self.viz_widget, "Visualization")

        self.setStatusBar(QStatusBar())

        self.worker = None

    def start_experiment(self):
        self.run_btn.setEnabled(False)
        self.status_label.setText("Starting...")
        self.setup_box.setPlainText("")
        for tbl in (self.summary_tbl, self.f1_tbl, self.per_query_tbl):
            tbl.clear()
            tbl.setRowCount(0)
            tbl.setColumnCount(0)
        self.fig_topk.clear()
        self.fig_rank.clear()
        self.canvas_topk.draw_idle()
        self.canvas_rank.draw_idle()

        self.worker = ExperimentWorker()
        self.worker.progress.connect(self.on_progress)
        self.worker.finished_ok.connect(self.on_done)
        self.worker.failed.connect(self.on_error)
        self.worker.start()

    def on_progress(self, msg):
        self.status_label.setText(msg)
        self.statusBar().showMessage(msg, 5000)

    def on_error(self, tb):
        self.run_btn.setEnabled(True)
        self.status_label.setText("Failed.")
        self.setup_box.setPlainText(tb)

    def on_done(self, res):
        self.status_label.setText("Done.")
        self.run_btn.setEnabled(True)

        # Setup pane
        setup_lines = [
            f"Target Q : {res['target_q']}",
            f"Target R : {res['target_r']}  (attacker-chosen; intentionally false)",
            f"Corpus   : {res['corpus_size']} docs ({res['clean_size']} clean + 1 poisoned at index {res['poison_idx']})",
            "",
            "Poisoned document (P = Q ⊕ I, with S=Q):",
            f"  {res['poisoned_doc']}",
        ]
        self.setup_box.setPlainText("\n".join(setup_lines))

        ks = res["ks"]
        sum_headers = ["group", "n"] + [f"top-{k}" for k in ks] + ["mean rank"]
        self.summary_tbl.setColumnCount(len(sum_headers))
        self.summary_tbl.setHorizontalHeaderLabels(sum_headers)
        self.summary_tbl.setRowCount(len(res["summary"]))
        for r, row in enumerate(res["summary"]):
            self.summary_tbl.setItem(r, 0, QTableWidgetItem(row["group"]))
            self.summary_tbl.setItem(r, 1, QTableWidgetItem(str(row["n"])))
            for c, k in enumerate(ks):
                v = row[f"top{k}"]
                item = QTableWidgetItem(f"{v:.2f}")
                if v >= 0.99:
                    item.setBackground(POISON_BG)
                self.summary_tbl.setItem(r, 2 + c, item)
            self.summary_tbl.setItem(
                r, 2 + len(ks),
                QTableWidgetItem(f"{row['mean_rank']:.1f}"),
            )

        f1_headers = ["group", "precision", "recall", "F1"]
        self.f1_tbl.setColumnCount(len(f1_headers))
        self.f1_tbl.setHorizontalHeaderLabels(f1_headers)
        self.f1_tbl.setRowCount(len(res["f1_table"]))
        for r, row in enumerate(res["f1_table"]):
            self.f1_tbl.setItem(r, 0, QTableWidgetItem(row["group"]))
            self.f1_tbl.setItem(r, 1, QTableWidgetItem(f"{row['precision']:.3f}"))
            self.f1_tbl.setItem(r, 2, QTableWidgetItem(f"{row['recall']:.3f}"))
            f1_item = QTableWidgetItem(f"{row['f1']:.3f}")
            if row["f1"] > 0:
                f1_item.setBackground(POISON_BG)
            self.f1_tbl.setItem(r, 3, f1_item)

        pq_headers = ["group", "query", "poison rank", "result rank",
                      "doc idx", "sim", "poison?", "snippet"]
        self.per_query_tbl.setColumnCount(len(pq_headers))
        self.per_query_tbl.setHorizontalHeaderLabels(pq_headers)
        rows = []
        for pq in res["per_query"]:
            for top in pq["top5"]:
                rows.append((pq, top))
        self.per_query_tbl.setRowCount(len(rows))
        for r, (pq, top) in enumerate(rows):
            cells = [
                pq["group"],
                pq["query"],
                str(pq["rank"]),
                str(top["rank"]),
                str(top["doc_idx"]),
                f"{top['sim']:.3f}",
                "YES" if top["is_poison"] else "",
                top["snippet"],
            ]
            for c, val in enumerate(cells):
                item = QTableWidgetItem(val)
                if top["is_poison"]:
                    item.setBackground(POISON_BG)
                self.per_query_tbl.setItem(r, c, item)
        self.per_query_tbl.horizontalHeader().setSectionResizeMode(
            len(pq_headers) - 1, QHeaderView.Stretch
        )

        self.render_plots(res)

    def render_plots(self, res):
        import numpy as np

        group_order = ["exact", "paraphrase", "unrelated"]
        group_colors = {
            "exact": "#1f77b4",
            "paraphrase": "#2ca02c",
            "unrelated": "#d62728",
        }
        ks = res["ks"]
        summary_by_group = {row["group"]: row for row in res["summary"]}

        ax1 = self.fig_topk.add_subplot(111)
        x = np.arange(len(group_order))
        width = 0.8 / len(ks)
        for i, k in enumerate(ks):
            values = [summary_by_group[g][f"top{k}"] for g in group_order]
            ax1.bar(x + i * width - 0.4 + width / 2, values, width, label=f"top-{k}")
        ax1.set_xticks(x)
        ax1.set_xticklabels(group_order)
        ax1.set_ylim(0, 1.05)
        ax1.set_ylabel("hit rate")
        ax1.set_title("Poisoned-document retrieval rate, by query group and k")
        ax1.legend(loc="center right", fontsize=9)
        ax1.grid(axis="y", alpha=0.3)
        self.canvas_topk.draw_idle()

        ax2 = self.fig_rank.add_subplot(111)
        rows = []
        for pq in res["per_query"]:
            rows.append((pq["group"], pq["query"], pq["rank"]))
        rows.sort(key=lambda r: (group_order.index(r[0]), r[2]))
        labels = [
            (q if len(q) <= 50 else q[:47] + "...") for (_, q, _) in rows
        ]
        ranks = [r[2] for r in rows]
        colors = [group_colors[r[0]] for r in rows]
        y_pos = np.arange(len(rows))
        ax2.barh(y_pos, ranks, color=colors)
        ax2.set_yticks(y_pos)
        ax2.set_yticklabels(labels, fontsize=8)
        ax2.invert_yaxis()
        ax2.set_xscale("log")
        ax2.set_xlim(0.9, max(ranks) * 1.4)
        ax2.set_xlabel("rank of poisoned doc (log scale; 1 = best for attacker)")
        ax2.set_title("Where the poison ranks for each query")
        ax2.axvline(5, color="grey", linestyle="--", linewidth=1, alpha=0.6)
        ax2.text(5, -0.5, "k=5", fontsize=8, color="grey", ha="center")
        for spine in ("top", "right"):
            ax2.spines[spine].set_visible(False)
        
        from matplotlib.patches import Patch
        handles = [Patch(facecolor=group_colors[g], label=g) for g in group_order]
        ax2.legend(handles=handles, loc="lower right", fontsize=9)
        self.canvas_rank.draw_idle()


def main():
    app = QApplication(sys.argv)
    w = MainWindow()
    w.show()
    sys.exit(app.exec_())


if __name__ == "__main__":
    main()
