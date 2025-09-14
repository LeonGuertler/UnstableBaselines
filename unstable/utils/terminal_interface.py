import ray, asyncio, psutil, time, re, pynvml
from typing import Dict, Any, Tuple
import collections
from collections import deque
from itertools import zip_longest

from rich import box
from rich.text import Text
from rich.live import Live
from rich.table import Table
from rich.style import Style
from rich.color import Color
from rich.panel import Panel
from rich.layout import Layout
from rich.console import Console, Group

# TODO track the learner info here as well maybe

_HIST_BARS = " ▁▂▃▄▅▆▇█"
_UID_FIELD = 22
PALETTE = {"ok": "#A9B665", "warn": "#E0C06F", "crit": "#EA6962"}

def _bar(pct: float, width: int) -> str: return "█" * int(pct / 100 * width)
def _trim_uid(uid: str, width: int = _UID_FIELD) -> str: return uid if len(uid) <= width else f"…{uid[-width-2:]}"#f"{uid[:width//2-1]}…{uid[-(width//2-1)-3:]}"
def _scaled_hist(hist: deque[float|int], width: int) -> str:
    if not hist: return " " * width
    step = max(1, len(hist) // width); buckets = [list(hist)[i:i+step] for i in range(0, len(hist), step)]; result = ""
    for bucket in buckets[:width]:
        if not bucket: result += " "; continue
        level = int(((sum(bucket) / len(bucket)) - min(hist)) / (max(hist) - min(hist) if max(hist) != min(hist) else 1e-5) * (len(_HIST_BARS) - 1))
        result += _HIST_BARS[level]
    return result.ljust(width)

class TerminalInterface:
    def __init__(self, tracker, buffer):
        self.tracker, self.buffer = tracker, buffer; self.console = Console(color_system="truecolor") 
        self._gpu_stats, self._general_stats, self._tracker_stats = None, None, None
        self._hist: dict[str, deque[float|int]] = collections.defaultdict(lambda: deque(maxlen=128))
        self._max_tok_s: int = 1_000; pynvml.nvmlInit(); self.pynvml=pynvml; self.gpu_count=pynvml.nvmlDeviceGetCount()

        # Initialize panel attributes with placeholder panels
        self._gpu_panel = Panel(Text("waiting …"), title="GPU", box=box.SQUARE); self._base_stats_panel = Panel(Text("waiting …"), title="Collection Stats.", box=box.SQUARE)
        self._ts_panel = Panel(Text("waiting …"), title="TrueSkill", box=box.DOUBLE); self._heatmap_panel = Panel(Text("waiting …"), title="Match Frequencies", box=box.SQUARE)
        self._exploration_panel = Panel(Text("Not Implemented"), title="Exploration", box=box.DOUBLE)

    async def _system_stats(self) -> Dict[str, Any]:
        gpus = []
        if self.pynvml:
            for gid in range(self.gpu_count):
                h=self.pynvml.nvmlDeviceGetHandleByIndex(gid); power=self.pynvml.nvmlDeviceGetPowerUsage(h)/1000.0; limit=self.pynvml.nvmlDeviceGetEnforcedPowerLimit(h)/1000.0; m=self.pynvml.nvmlDeviceGetMemoryInfo(h)
                gpus.append({"id": gid, "used": m.used/1e9, "total": m.total/1e9, "mem_pct": (m.used/1e9)/(m.total/1e9)*100, "power": power, "limit": limit, "power_pct": power/limit*100 if limit else 0.0})
        return gpus

    async def _fetch_loop(self, interval: float = 2.0):
        while True:
            try:
                self._gpu_stats=await self._system_stats(); self._buffer_size=await self.buffer.size.remote(); self._tracker_stats=await self.tracker.get_interface_info.remote() # Fetch all stats
                self._gpu_panel = self._gpu(); self._base_stats_panel = self._base_stats(); self._ts_panel = self._ts(); self._heatmap_panel = self._heatmap(); self._exploration_panel = self._exploration() # Update all panels with new data
            except Exception as e: self.console.log(f"[red]stat-fetch error: {e}")
            await asyncio.sleep(interval)

    def _colour_for_util(self, pct: float) -> str: return PALETTE["ok"] if pct >= 75 else PALETTE["warn"] if pct >= 40 else PALETTE["crit"]
    def _gpu(self) -> Panel:
        if not self._gpu_stats or not self._tracker_stats: return Panel(Text("waiting …"), title="GPU", box=box.SQUARE)
        gpu_panels = []; bar_w = max(10, int(self.console.size.width * 0.48 - 20))
        for gpu_d in self._gpu_stats:
            tok_s = self._tracker_stats["gpu_tok_s"].get(gpu_d["id"], 0); self._max_tok_s = self._max_tok_s if tok_s<self._max_tok_s else tok_s
            tok_pct = tok_s/self._max_tok_s * 100; role = "Actor" if tok_s and tok_s > 0 else "Learner"
            line1 = Text.assemble(("TOK ", "dim"), Text(_bar(tok_pct, bar_w),            style=self._colour_for_util(tok_pct)),             f" {tok_s:5.0f} tok/s")
            line2 = Text.assemble(("PWR ", "dim"), Text(_bar(gpu_d['power_pct'], bar_w), style=self._colour_for_util(gpu_d['power_pct'])),  f" {gpu_d['power_pct']:5.1f}%")
            line3 = Text.assemble(("MEM ", "dim"), Text(_bar(gpu_d['mem_pct'], bar_w),   style=self._colour_for_util(gpu_d['mem_pct'])),    f" {gpu_d['mem_pct']:5.1f}%")
            gpu_panels.append(Panel(Group(line1, line2, line3), title=f"GPU{gpu_d['id']} - {role}", box=box.SQUARE, padding=(0, 1)))
        
        # build a 2-column grid (rows = ceil(N/2))
        tbl = Table.grid(expand=True, padding=0); tbl.add_column(ratio=1); tbl.add_column(ratio=1)
        for a, b in zip_longest(gpu_panels[0::2], gpu_panels[1::2], fillvalue=Panel("\n\n")): tbl.add_row(a, b) # filler for odd counts
        return Panel(tbl, title="GPU Performance", box=box.DOUBLE)

    def _base_stats(self) -> Panel:
        if not self._tracker_stats: return Panel(Text("waiting …"), title="Collection Stats.", box=box.SQUARE)
        try:
            self._hist["format_success"].append(self._tracker_stats["Format Success Rate - correct_answer_format"]*100); self._hist["inv_move_rate"].append(self._tracker_stats["Format Success Rate - invalid_move"]*100)
            self._hist["game_len"].append(self._tracker_stats["Game Length"]); self._hist["buffer_size"].append(self._buffer_size); bar_w = max(10, int(self.console.size.width * 0.48 - 35))
            line1 = Text.assemble(("Format Success: ", "dim"), Text(_scaled_hist(self._hist["format_success"], bar_w),  style="dim"), f" {self._hist['format_success'][-1]:5.2f}%")
            line2 = Text.assemble(("Inv. Move Rate: ", "dim"), Text(_scaled_hist(self._hist["inv_move_rate"], bar_w),   style="dim"), f" {self._hist['inv_move_rate'][-1]:5.2f}%")
            line3 = Text.assemble(("Game Length:    ", "dim"), Text(_scaled_hist(self._hist["game_len"], bar_w),        style="dim"), f" {self._hist['game_len'][-1]:5.2f} turns")
            line4 = Text.assemble(("Buffer Size:    ", "dim"), Text(_scaled_hist(self._hist["buffer_size"], bar_w),     style="dim"), f" {self._hist['buffer_size'][-1]:5.0f} samples")
            return Panel(Group(line1, line2, line3, line4), title=f"Collection Stats.", box=box.SQUARE, padding=(0, 1))
        except Exception as exc:
            return Panel(Text(f"Exception: {exc}"), title="Collection Stats.", box=box.SQUARE)
    
    def _ts(self) -> Panel:
        if not self._tracker_stats or not self._tracker_stats.get("TS"): return Panel(Text("waiting …"), title="TrueSkill", box=box.DOUBLE)
        bar_field = max(10, int(self.console.size.width * 0.45)-50); max_rows = max(3, int(self.console.size.height * 0.35))
        entries = [(uid, v["rating"].mu, v["rating"].sigma) for uid, v in self._tracker_stats["TS"].items()]
        entries.sort(key=lambda x: x[1], reverse=True); entries = entries[:max_rows]
        self._ts_idx: dict[str, int] = {}   # reset each draw
        for idx, (uid, _, _) in enumerate(entries): self._ts_idx[uid] = idx
        max_mu = max(mu for _, mu, _ in entries); bar_lines = []
        for uid, mu, sigma in entries:
            bar = "█" * int((mu/max_mu if max_mu else 0.0) * bar_field); bar_text = f"{bar:<{bar_field}}"  # pad to fixed width
            bar_lines.append(Text.assemble((f"[{self._ts_idx[uid]:<3}] {_trim_uid(uid):<25}", "bold"), "  ", Text(bar_text, style="cyan"), f"  μ {mu:6.2f}  σ {sigma:4.2f}"))
        title = f"TrueSkill (μ, σ)  – {len(self._tracker_stats['TS'])} models"
        return Panel(Group(*bar_lines), title=title, box=box.SQUARE, padding=(0, 1))

    def _heatmap(self) -> Panel: 
        if not self._tracker_stats or not self._tracker_stats.get("match_counts") or not getattr(self, "_ts_idx", None): return Panel(Text("waiting …"), title="Match Frequencies", box=box.SQUARE)
        uid_by_row = sorted(self._ts_idx, key=self._ts_idx.__getitem__); max_rows = max(3, int(self.console.size.height * 0.35)); uid_by_row = uid_by_row[:max_rows]
        def _cnt(a: str, b: str) -> int: return self._tracker_stats["match_counts"].get(tuple(sorted((a, b))), 0)
        max_cnt = max((_cnt(a, b) for a in uid_by_row for b in uid_by_row), default=1); light_rgb = (0, 43, 54); dark_rgb  = (255,  87,  34) # TODO somehow set light_rgb to be the users terminal color
        def _bg_style(pct: float) -> Style: return Style(bgcolor=Color.from_rgb(int(light_rgb[0]+pct*(dark_rgb[0]-light_rgb[0])), int(light_rgb[1]+pct*(dark_rgb[1]-light_rgb[1])), int(light_rgb[2]+pct*(dark_rgb[2]-light_rgb[2]))))
        tbl = Table.grid(padding=(0, 1)); tbl.add_row(""); header = [""] + [Text(f"{self._ts_idx[u]:>2}", style="bold") for u in uid_by_row]; tbl.add_row(*header)
        for ua in uid_by_row:
            row_cells = [Text(f"{self._ts_idx[ua]:>2}", style="bold")]
            for ub in uid_by_row:
                c = _cnt(ua, ub); row_cells.append(Text(f"{c:3}", style=_bg_style(c / max_cnt if max_cnt else 0.0)))
            tbl.add_row(*row_cells)
        return Panel(tbl, title="Match Counts", box=box.SQUARE)

    def _exploration(self) -> Panel: return Panel(Text("Not Implemented"), title="Exploration", box=box.DOUBLE) 
    async def run(self):
        layout = Layout(); layout.split_column(Layout(name="grid"), Layout(name="gpu", size=5*((self.gpu_count+self.gpu_count%2)//2)+2))
        layout["grid"].split_row(Layout(name="col1"), Layout(name="col2")); layout["col1"].split_column(Layout(name="exploration",ratio=1), Layout(name="heatmap", ratio=2))
        layout["col2"].split_column(Layout(name="ts"), Layout(name="bs", size=6)); asyncio.create_task(self._fetch_loop())  # Start background fetcher
        with Live(layout, console=self.console, auto_refresh=False) as live:
            while True:
                layout["gpu"].update(self._gpu_panel); layout["bs"].update(self._base_stats_panel); layout["ts"].update(self._ts_panel)
                layout["heatmap"].update(self._heatmap_panel); layout["exploration"].update(self._exploration_panel)
                live.refresh()
                await asyncio.sleep(2)
