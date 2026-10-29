"""
E-Commerce Sales Analyzer - Desktop GUI
---------------------------------------
GUI version of the menu-driven "E-Commerce Sales Analyzer" mini project.
Same stack as the report: pandas + NumPy + Matplotlib (GUI via Tkinter, which
ships with Python, so nothing extra to install).

Run:   python ecommerce_sales_analyzer_gui.py
Needs: pip install pandas numpy matplotlib
Data:  Ecommerce_Sales_500_Data.csv in the same folder (or use "Open CSV...").
"""

import os
import tkinter as tk
from tkinter import ttk, filedialog, messagebox

import numpy as np
import pandas as pd
from matplotlib.figure import Figure
from matplotlib.ticker import FuncFormatter
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg, NavigationToolbar2Tk

DEFAULT_CSV = "Ecommerce_Sales_500_Data.csv"
REQUIRED_COLS = [
    "Order_ID", "Order_Date", "Customer_ID", "Customer_Name", "Product",
    "Category", "Quantity", "Unit_Price", "Region", "Payment_Mode", "Order_Status",
]
REV = "Calculated_Total_Sales"

# ---- colours / fonts -------------------------------------------------------
BG = "#f4f6fb"
SIDEBAR = "#1f2a44"
SIDEBAR_HOVER = "#2f3d63"
ACCENT = "#3b6df0"
CARD = "#ffffff"
TEXT = "#1d2433"
MUTED = "#6b7590"
PALETTE = ["#3b6df0", "#f08c3b", "#2fb37a", "#e5484d", "#8e5cf0", "#17a2b8", "#d4a017"]
STATUS_COLORS = {"Delivered": "#2fb37a", "Pending": "#f0b429",
                 "Cancelled": "#e5484d", "Returned": "#8e5cf0"}


def inr(x):
    return f"\u20b9{x:,.2f}"


# =============================================================================
# DATA LAYER  (same logic as the notebook, but returns data instead of printing)
# =============================================================================
def load_and_clean(path):
    """Load CSV, capture the 'overview' info, then clean. Returns (df, info)."""
    raw = pd.read_csv(path)
    missing_cols = [c for c in REQUIRED_COLS if c not in raw.columns]
    if missing_cols:
        raise ValueError("CSV is missing required column(s): " + ", ".join(missing_cols))

    info = {
        "shape": raw.shape,
        "columns": raw.columns.tolist(),
        "head": raw.head(),
        "tail": raw.tail(),
        "dtypes": pd.DataFrame({"Column": raw.columns, "Data Type": raw.dtypes.astype(str).values}),
        "missing": pd.DataFrame({"Column": raw.columns, "Missing Values": raw.isnull().sum().values}),
        "rows_dropped": int(raw.isnull().any(axis=1).sum()),
    }
    df = raw.dropna().copy()
    df["Order_Date"] = pd.to_datetime(df["Order_Date"])
    df[REV] = df["Quantity"] * df["Unit_Price"]
    return df.reset_index(drop=True), info


# Each analysis returns: {"title", "kpis": [(label, value)], "tables": [(name, df)], "fig": Figure|None}
def _fig(w=7.5, h=4.6):
    f = Figure(figsize=(w, h), dpi=100)
    f.patch.set_facecolor("white")
    return f, f.add_subplot(111)


def _bar(ax, labels, values, color, title, xlabel, ylabel, rotate=0):
    bars = ax.bar(labels, values, color=color, label=ylabel)
    ax.set_title(title, fontsize=13, fontweight="bold")
    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
    ax.grid(axis="y", linestyle="--", alpha=0.5)
    ax.set_axisbelow(True)
    ax.legend()
    ax.tick_params(axis="x", rotation=rotate)
    ax.yaxis.set_major_formatter(FuncFormatter(lambda v, _: f"{v:,.0f}"))
    for b, v in zip(bars, values):
        ax.annotate(f"{v/1e5:.1f}L", (b.get_x() + b.get_width() / 2, b.get_height()),
                    ha="center", va="bottom", fontsize=8, color=MUTED)


def a_overview(df, info):
    r, c = info["shape"]
    return {
        "title": "Data Overview",
        "kpis": [("Rows (raw)", r), ("Columns", c),
                 ("Rows dropped (nulls)", info["rows_dropped"]), ("Rows after cleaning", len(df))],
        "tables": [("First 5 Records", info["head"]), ("Last 5 Records", info["tail"]),
                   ("Data Types", info["dtypes"]), ("Missing Values", info["missing"])],
        "fig": None,
    }


def a_basic(df, info):
    total_orders = df["Order_ID"].nunique()
    total_rev = df[REV].sum()
    s = df[REV].to_numpy()
    np_stats = pd.DataFrame({
        "Statistic": ["Mean", "Median", "Minimum", "Maximum", "Standard Deviation"],
        "Order Value": [inr(np.mean(s)), inr(np.median(s)), inr(np.min(s)),
                        inr(np.max(s)), inr(np.std(s))],
    })
    f, ax = _fig()
    ax.hist(s, bins=25, color=ACCENT, edgecolor="white", label="Orders")
    ax.axvline(np.mean(s), color="#e5484d", ls="--", label=f"Mean {inr(np.mean(s))}")
    ax.axvline(np.median(s), color="#2fb37a", ls="--", label=f"Median {inr(np.median(s))}")
    ax.set_title("Order Value Distribution", fontsize=13, fontweight="bold")
    ax.set_xlabel("Order value (\u20b9)")
    ax.set_ylabel("Number of orders")
    ax.legend()
    ax.grid(axis="y", ls="--", alpha=0.5)
    return {
        "title": "Basic Statistics",
        "kpis": [("Total Orders", f"{total_orders:,}"),
                 ("Quantity Sold", f"{df['Quantity'].sum():,}"),
                 ("Total Revenue", inr(total_rev)),
                 ("Avg Order Value", inr(total_rev / total_orders)),
                 ("Avg Product Price", inr(df["Unit_Price"].mean())),
                 ("Customers", f"{df['Customer_ID'].nunique():,}")],
        "tables": [("NumPy Statistics (Order Value)", np_stats)],
        "fig": f,
    }


def a_product(df, info):
    ps = df.groupby("Product").agg(Quantity=("Quantity", "sum"), Revenue=(REV, "sum")).reset_index()
    best_q = ps.loc[ps["Quantity"].idxmax()]
    best_r = ps.loc[ps["Revenue"].idxmax()]
    top10 = ps.nlargest(10, "Revenue")
    f, ax = _fig(8.5, 4.8)
    _bar(ax, top10["Product"], top10["Revenue"], "#6aa9f0",
         "Top 10 Products by Revenue", "Product", "Revenue (\u20b9)", rotate=40)
    f.tight_layout()
    return {
        "title": "Product-wise Analysis",
        "kpis": [("Best Seller (Qty)", f"{best_q['Product']} \u2022 {int(best_q['Quantity'])} units"),
                 ("Top Revenue Product", f"{best_r['Product']} \u2022 {inr(best_r['Revenue'])}")],
        "tables": [("Top 5 by Quantity", ps.nlargest(5, "Quantity")[["Product", "Quantity"]]),
                   ("Top 5 by Revenue", ps.nlargest(5, "Revenue")[["Product", "Revenue"]]),
                   ("All Products", ps.sort_values("Revenue", ascending=False))],
        "fig": f,
    }


def a_category(df, info):
    cs = df.groupby("Category").agg(Quantity=("Quantity", "sum"), Revenue=(REV, "sum")).reset_index()
    f, ax = _fig()
    _bar(ax, cs["Category"], cs["Revenue"], "#f08c6a", "Revenue by Category",
         "Category", "Revenue (\u20b9)", rotate=25)
    f.tight_layout()
    top = cs.loc[cs["Revenue"].idxmax()]
    share = top["Revenue"] / cs["Revenue"].sum() * 100
    return {
        "title": "Category-wise Analysis",
        "kpis": [("Top Category", str(top["Category"])), ("Its Revenue Share", f"{share:.1f}%")],
        "tables": [("Category Summary", cs)],
        "fig": f,
    }


def a_region(df, info):
    rs = df.groupby("Region").agg(Quantity=("Quantity", "sum"), Revenue=(REV, "sum")).reset_index()
    f, ax = _fig()
    _bar(ax, rs["Region"], rs["Revenue"], "#7cc98a", "Revenue by Region", "Region", "Revenue (\u20b9)")
    f.tight_layout()
    top = rs.loc[rs["Revenue"].idxmax()]
    return {
        "title": "Region-wise Analysis",
        "kpis": [("Top Region", str(top["Region"])), ("Its Revenue", inr(top["Revenue"]))],
        "tables": [("Region Summary", rs)],
        "fig": f,
    }


def a_monthly(df, info):
    d = df.copy()
    d["Month"] = d["Order_Date"].dt.to_period("M").astype(str)
    ms = d.groupby("Month").agg(Total_Orders=("Order_ID", "count"),
                                Quantity=("Quantity", "sum"), Revenue=(REV, "sum")).reset_index()
    f, ax = _fig(9, 4.8)
    ax.plot(ms["Month"], ms["Revenue"], marker="o", lw=2, color="#8e5cf0", label="Monthly Revenue")
    ax.set_title("Monthly Sales Trend", fontsize=13, fontweight="bold")
    ax.set_xlabel("Month")
    ax.set_ylabel("Revenue (\u20b9)")
    ax.tick_params(axis="x", rotation=45)
    ax.grid(True, ls="--", alpha=0.5)
    ax.legend()
    f.tight_layout()
    best = ms.loc[ms["Revenue"].idxmax()]
    return {
        "title": "Monthly Sales Trend",
        "kpis": [("Best Month", str(best["Month"])), ("Best Month Revenue", inr(best["Revenue"])),
                 ("Months Covered", len(ms))],
        "tables": [("Monthly Summary", ms)],
        "fig": f,
    }


def _pie(ax, values, labels, title, colors=None):
    ax.pie(values, labels=labels, autopct="%1.1f%%", startangle=140, colors=colors,
           wedgeprops={"edgecolor": "white"})
    ax.set_title(title, fontsize=13, fontweight="bold")
    ax.legend(labels, loc="center left", bbox_to_anchor=(1, 0.5))


def a_status(df, info):
    sc = df["Order_Status"].value_counts()
    total = len(df)
    canc = sc.get("Cancelled", 0) / total * 100
    ret = sc.get("Returned", 0) / total * 100
    f, ax = _fig(7.5, 4.8)
    _pie(ax, sc.values, sc.index, "Order Status Distribution",
         [STATUS_COLORS.get(s, "#999") for s in sc.index])
    f.tight_layout()
    tbl = sc.rename_axis("Order_Status").reset_index(name="Orders")
    tbl["Share %"] = (tbl["Orders"] / total * 100).round(2)
    return {
        "title": "Order Status & Return Rate",
        "kpis": [("Cancellation Rate", f"{canc:.2f}%"), ("Return Rate", f"{ret:.2f}%"),
                 ("Delivered", int(sc.get("Delivered", 0))), ("Pending", int(sc.get("Pending", 0)))],
        "tables": [("Status Counts", tbl)],
        "fig": f,
    }


def a_payment(df, info):
    ps = df.groupby("Payment_Mode").agg(Total_Orders=("Order_ID", "count"),
                                        Revenue=(REV, "sum")).reset_index()
    f, ax = _fig(7.5, 4.8)
    _pie(ax, ps["Total_Orders"], ps["Payment_Mode"], "Payment Mode Distribution (By Orders)", PALETTE)
    f.tight_layout()
    fav = ps.loc[ps["Total_Orders"].idxmax()]
    return {
        "title": "Payment Mode Analysis",
        "kpis": [("Most Used", str(fav["Payment_Mode"])), ("Its Orders", int(fav["Total_Orders"]))],
        "tables": [("Payment Summary", ps)],
        "fig": f,
    }


def a_customer(df, info):
    cs = df.groupby(["Customer_ID", "Customer_Name"]).agg(
        Orders=("Order_ID", "count"), Revenue=(REV, "sum")).reset_index()
    hs = cs.loc[cs["Revenue"].idxmax()]
    mf = cs.loc[cs["Orders"].idxmax()]
    top = cs.nlargest(10, "Revenue")
    f, ax = _fig(8.5, 4.8)
    _bar(ax, top["Customer_Name"], top["Revenue"], "#e0b04a",
         "Top 10 Customers by Spend", "Customer", "Revenue (\u20b9)", rotate=40)
    f.tight_layout()
    return {
        "title": "Customer-wise Analysis",
        "kpis": [("Highest Spender", f"{hs['Customer_Name']} ({hs['Customer_ID']}) \u2022 {inr(hs['Revenue'])}"),
                 ("Most Frequent", f"{mf['Customer_Name']} ({mf['Customer_ID']}) \u2022 {int(mf['Orders'])} orders")],
        "tables": [("Top 10 by Spend", top), ("Top 10 by Orders", cs.nlargest(10, "Orders")),
                   ("All Customers", cs.sort_values("Revenue", ascending=False))],
        "fig": f,
    }


MENU = [
    ("\U0001F4CB  Data Overview", a_overview),
    ("\U0001F4CA  Basic Statistics", a_basic),
    ("\U0001F6CD  Product-wise", a_product),
    ("\U0001F5C2  Category-wise", a_category),
    ("\U0001F30D  Region-wise", a_region),
    ("\U0001F4C8  Monthly Trend", a_monthly),
    ("\U0001F4E6  Order Status", a_status),
    ("\U0001F4B3  Payment Mode", a_payment),
    ("\U0001F465  Customer-wise", a_customer),
]


# =============================================================================
# GUI
# =============================================================================
class App(tk.Tk):
    def __init__(self):
        super().__init__()
        self.title("E-Commerce Sales Analyzer")
        self.geometry("1280x800")
        self.minsize(1000, 650)
        self.configure(bg=BG)

        self.raw_df = None
        self.info = None
        self.current = None          # (index, result dict)
        self.canvas = None
        self.toolbar = None
        self.menu_buttons = []
        self.current_tables = {}

        self._style()
        self._build()
        self.bind("<Control-o>", lambda e: self.open_csv())

        if os.path.exists(DEFAULT_CSV):
            self.load(DEFAULT_CSV)
        else:
            self._placeholder("Click  \u201cOpen CSV\u2026\u201d  to load your dataset\n(expected: " + DEFAULT_CSV + ")")

    # ---- styling -----------------------------------------------------------
    def _style(self):
        s = ttk.Style(self)
        try:
            s.theme_use("clam")
        except tk.TclError:
            pass
        s.configure("Treeview", rowheight=26, font=("Segoe UI", 10), background="white",
                    fieldbackground="white", foreground=TEXT, borderwidth=0)
        s.configure("Treeview.Heading", font=("Segoe UI", 10, "bold"), background="#e8ecf6",
                    foreground=TEXT, relief="flat")
        s.map("Treeview", background=[("selected", ACCENT)], foreground=[("selected", "white")])
        s.configure("TNotebook", background=BG, borderwidth=0)
        s.configure("TNotebook.Tab", padding=(14, 6), font=("Segoe UI", 10))
        s.configure("TCombobox", padding=3)

    # ---- layout ------------------------------------------------------------
    def _build(self):
        # sidebar
        side = tk.Frame(self, bg=SIDEBAR, width=220)
        side.pack(side="left", fill="y")
        side.pack_propagate(False)
        tk.Label(side, text="Sales\nAnalyzer", bg=SIDEBAR, fg="white",
                 font=("Segoe UI", 18, "bold"), justify="left").pack(anchor="w", padx=18, pady=(22, 2))
        tk.Label(side, text="E-Commerce Insights", bg=SIDEBAR, fg="#9fb0d9",
                 font=("Segoe UI", 9)).pack(anchor="w", padx=18, pady=(0, 18))

        for i, (label, _) in enumerate(MENU):
            b = tk.Label(side, text=label, bg=SIDEBAR, fg="white", anchor="w",
                         font=("Segoe UI", 11), padx=18, pady=9, cursor="hand2")
            b.pack(fill="x")
            b.bind("<Button-1>", lambda e, i=i: self.show(i))
            b.bind("<Enter>", lambda e, b=b: b.configure(bg=SIDEBAR_HOVER) if b["bg"] != ACCENT else None)
            b.bind("<Leave>", lambda e, b=b: b.configure(bg=SIDEBAR) if b["bg"] != ACCENT else None)
            self.menu_buttons.append(b)

        ex = tk.Label(side, text="\u23FB  Exit", bg=SIDEBAR, fg="#ff9aa2", anchor="w",
                      font=("Segoe UI", 11), padx=18, pady=9, cursor="hand2")
        ex.pack(side="bottom", fill="x", pady=(0, 12))
        ex.bind("<Button-1>", lambda e: self.destroy())

        # main area
        main = tk.Frame(self, bg=BG)
        main.pack(side="left", fill="both", expand=True)

        # top bar
        top = tk.Frame(main, bg=CARD, height=56)
        top.pack(fill="x")
        top.pack_propagate(False)
        tk.Button(top, text="\U0001F4C2 Open CSV\u2026", command=self.open_csv, bg=ACCENT, fg="white",
                  relief="flat", padx=12, pady=4, font=("Segoe UI", 10, "bold"),
                  activebackground="#2b55c9", activeforeground="white", cursor="hand2"
                  ).pack(side="left", padx=14, pady=10)
        self.file_lbl = tk.Label(top, text="No file loaded", bg=CARD, fg=MUTED, font=("Segoe UI", 9))
        self.file_lbl.pack(side="left")

        # filters (right side)
        self.filter_vars = {}
        fr = tk.Frame(top, bg=CARD)
        fr.pack(side="right", padx=12)
        for col, label in (("Category", "Category"), ("Region", "Region"), ("Order_Status", "Status")):
            tk.Label(fr, text=label, bg=CARD, fg=MUTED, font=("Segoe UI", 9)).pack(side="left", padx=(10, 3))
            v = tk.StringVar(value="All")
            cb = ttk.Combobox(fr, textvariable=v, values=["All"], state="readonly", width=13)
            cb.pack(side="left")
            cb.bind("<<ComboboxSelected>>", lambda e: self.refresh())
            self.filter_vars[col] = (v, cb)
        tk.Button(fr, text="Reset", command=self.reset_filters, relief="flat", bg="#e8ecf6", fg=TEXT,
                  padx=8, cursor="hand2").pack(side="left", padx=(10, 0))

        # title + kpi + content
        self.title_lbl = tk.Label(main, text="", bg=BG, fg=TEXT, font=("Segoe UI", 18, "bold"), anchor="w")
        self.title_lbl.pack(fill="x", padx=20, pady=(14, 4))
        self.kpi_frame = tk.Frame(main, bg=BG)
        self.kpi_frame.pack(fill="x", padx=14)
        self.content = tk.Frame(main, bg=BG)
        self.content.pack(fill="both", expand=True, padx=20, pady=(8, 6))

        self.status = tk.Label(main, text="", bg=BG, fg=MUTED, anchor="w", font=("Segoe UI", 9))
        self.status.pack(fill="x", padx=20, pady=(0, 8))

    # ---- data --------------------------------------------------------------
    def open_csv(self):
        path = filedialog.askopenfilename(title="Select sales CSV",
                                          filetypes=[("CSV files", "*.csv"), ("All files", "*.*")])
        if path:
            self.load(path)

    def load(self, path):
        try:
            self.raw_df, self.info = load_and_clean(path)
        except FileNotFoundError:
            messagebox.showerror("File not found", f"Could not find:\n{path}")
            return
        except Exception as e:  # bad format, missing columns, etc.
            messagebox.showerror("Could not load data", str(e))
            return
        self.file_lbl.config(text=f"{os.path.basename(path)}  \u2022  {len(self.raw_df)} orders")
        for col, (var, cb) in self.filter_vars.items():
            cb["values"] = ["All"] + sorted(self.raw_df[col].unique().tolist())
            var.set("All")
        self.show(self.current[0] if self.current else 1)

    def filtered(self):
        df = self.raw_df
        for col, (var, _) in self.filter_vars.items():
            if var.get() != "All":
                df = df[df[col] == var.get()]
        return df

    def reset_filters(self):
        for var, _ in self.filter_vars.values():
            var.set("All")
        self.refresh()

    def refresh(self):
        if self.current:
            self.show(self.current[0])

    # ---- rendering ---------------------------------------------------------
    def _placeholder(self, text):
        for w in self.content.winfo_children():
            w.destroy()
        tk.Label(self.content, text=text, bg=BG, fg=MUTED, font=("Segoe UI", 13)).pack(expand=True)

    def show(self, idx):
        if self.raw_df is None:
            messagebox.showinfo("No data", "Please open a CSV file first.")
            return
        df = self.filtered()
        if df.empty:
            self._placeholder("No orders match the selected filters.")
            return
        for i, b in enumerate(self.menu_buttons):
            b.configure(bg=ACCENT if i == idx else SIDEBAR)

        try:
            result = MENU[idx][1](df, self.info)
        except Exception as e:
            messagebox.showerror("Analysis error", f"{type(e).__name__}: {e}")
            return
        self.current = (idx, result)
        self.title_lbl.config(text=result["title"])
        self._render_kpis(result["kpis"])
        self._render_content(result)
        active = [f"{c}={v.get()}" for c, (v, _) in self.filter_vars.items() if v.get() != "All"]
        self.status.config(text=f"{len(df)} orders analysed" +
                           (f"  \u2022  filters: {', '.join(active)}" if active else ""))

    def _render_kpis(self, kpis):
        for w in self.kpi_frame.winfo_children():
            w.destroy()
        for i, (label, value) in enumerate(kpis):
            card = tk.Frame(self.kpi_frame, bg=CARD, highlightbackground="#e1e6f2", highlightthickness=1)
            card.pack(side="left", padx=6, pady=4, ipadx=8, ipady=4)
            tk.Label(card, text=label.upper(), bg=CARD, fg=MUTED, font=("Segoe UI", 8, "bold")).pack(anchor="w", padx=8, pady=(6, 0))
            tk.Label(card, text=str(value), bg=CARD, fg=TEXT, font=("Segoe UI", 12, "bold")).pack(anchor="w", padx=8, pady=(0, 6))

    def _render_content(self, result):
        for w in self.content.winfo_children():
            w.destroy()
        self.canvas = self.toolbar = None
        self.current_tables = {}
        nb = ttk.Notebook(self.content)
        nb.pack(fill="both", expand=True)

        if result["fig"] is not None:
            tab = tk.Frame(nb, bg="white")
            nb.add(tab, text="  Chart  ")
            self.canvas = FigureCanvasTkAgg(result["fig"], master=tab)
            self.canvas.draw()
            tb = tk.Frame(tab, bg="white")
            tb.pack(side="bottom", fill="x")
            self.toolbar = NavigationToolbar2Tk(self.canvas, tb)
            self.toolbar.update()
            self.canvas.get_tk_widget().pack(fill="both", expand=True)

        for name, tdf in result["tables"]:
            tab = tk.Frame(nb, bg="white")
            nb.add(tab, text=f"  {name}  ")
            self._table(tab, tdf)
            self.current_tables[name] = tdf

        bar = tk.Frame(self.content, bg=BG)
        bar.pack(fill="x", pady=(6, 0))
        tk.Button(bar, text="\u2B07 Export table (CSV)", command=lambda: self.export_table(nb),
                  relief="flat", bg="#e8ecf6", fg=TEXT, padx=10, cursor="hand2").pack(side="left")
        if result["fig"] is not None:
            tk.Button(bar, text="\U0001F5BC Save chart (PNG)", command=self.save_chart,
                      relief="flat", bg="#e8ecf6", fg=TEXT, padx=10, cursor="hand2").pack(side="left", padx=6)

    def _table(self, parent, tdf):
        cols = list(tdf.columns)
        tv = ttk.Treeview(parent, columns=cols, show="headings")
        ys = ttk.Scrollbar(parent, orient="vertical", command=tv.yview)
        xs = ttk.Scrollbar(parent, orient="horizontal", command=tv.xview)
        tv.configure(yscrollcommand=ys.set, xscrollcommand=xs.set)
        ys.pack(side="right", fill="y")
        xs.pack(side="bottom", fill="x")
        tv.pack(fill="both", expand=True)
        tv.tag_configure("odd", background="#f7f9fd")
        for c in cols:
            numeric = pd.api.types.is_numeric_dtype(tdf[c])
            tv.heading(c, text=c.replace("_", " "))
            tv.column(c, width=max(110, min(220, len(str(c)) * 11)), anchor="e" if numeric else "w")
        for n, row in enumerate(tdf.itertuples(index=False)):
            vals = []
            for c, v in zip(cols, row):
                if isinstance(v, pd.Timestamp):
                    v = v.strftime("%Y-%m-%d")
                elif isinstance(v, (float, np.floating)):
                    v = f"{v:,.2f}"
                elif isinstance(v, (int, np.integer)) and not str(c).endswith("ID"):
                    v = f"{v:,}"
                vals.append(v)
            tv.insert("", "end", values=vals, tags=("odd",) if n % 2 else ())

    # ---- export ------------------------------------------------------------
    def export_table(self, nb):
        label = nb.tab(nb.select(), "text").strip()
        if label not in self.current_tables:
            messagebox.showinfo("Export", "Select a table tab first.")
            return
        path = filedialog.asksaveasfilename(defaultextension=".csv", initialfile=label.replace(" ", "_") + ".csv",
                                            filetypes=[("CSV", "*.csv")])
        if path:
            self.current_tables[label].to_csv(path, index=False)
            self.status.config(text=f"Saved {path}")

    def save_chart(self):
        if not self.current or self.current[1]["fig"] is None:
            return
        path = filedialog.asksaveasfilename(defaultextension=".png",
                                            initialfile=self.current[1]["title"].replace(" ", "_") + ".png",
                                            filetypes=[("PNG image", "*.png")])
        if path:
            self.current[1]["fig"].savefig(path, dpi=200, bbox_inches="tight")
            self.status.config(text=f"Saved {path}")


if __name__ == "__main__":
    App().mainloop()