from __future__ import annotations
from typing import List, cast
from datetime import datetime
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
from shiny import Inputs, Outputs, Session, reactive, render, ui

from .utils import epley_1rm, epley_training_max, BIG3, REQUIRED_TABS
from .repos import Repo, repo_factory

# Set matplotlib style for professional look
plt.style.use('seaborn-v0_8-darkgrid')
plt.rcParams['figure.facecolor'] = 'white'
plt.rcParams['axes.facecolor'] = '#f8f9fa'
plt.rcParams['axes.edgecolor'] = '#dee2e6'
plt.rcParams['grid.color'] = '#dee2e6'
plt.rcParams['grid.alpha'] = 0.5
plt.rcParams['font.size'] = 10
plt.rcParams['axes.labelsize'] = 11
plt.rcParams['axes.titlesize'] = 12
plt.rcParams['legend.fontsize'] = 9
plt.rcParams['xtick.labelsize'] = 9
plt.rcParams['ytick.labelsize'] = 9


def server(input: Inputs, output: Outputs, session: Session):
    repo = reactive.value(cast(Repo, None))

    def get_repo():
        r = repo.get()
        return cast(Repo, r) if r is not None and hasattr(r, 'read_df') else None

    @reactive.effect
    def _init_repo():
        repo.set(repo_factory())

    # Dataframes
    @reactive.calc
    def lifts_df():
        r = get_repo()
        df = r.read_df("Lifts") if r else pd.DataFrame(columns=REQUIRED_TABS["Lifts"])
        if not df.empty and all(c in df.columns for c in ["weight_kg", "reps"]):
            # Compute 1RM and Training Max (90%) with 1-decimal rounding
            df = df.copy()
            def _safe_1rm(row):
                try:
                    return epley_1rm(float(row.get("weight_kg", 0) or 0), int(row.get("reps", 1) or 1))
                except Exception:
                    return float("nan")
            def _safe_tm(row):
                try:
                    return epley_training_max(float(row.get("weight_kg", 0) or 0), int(row.get("reps", 1) or 1))
                except Exception:
                    return float("nan")
            df["1rm"] = df.apply(_safe_1rm, axis=1)
            df["tm"] = df.apply(_safe_tm, axis=1)
        return df

    @reactive.calc
    def bw_df():
        r = get_repo()
        return r.read_df("Bodyweight") if r else pd.DataFrame(columns=REQUIRED_TABS["Bodyweight"])

    @reactive.calc
    def meas_df():
        r = get_repo()
        return r.read_df("Measurements") if r else pd.DataFrame(columns=REQUIRED_TABS["Measurements"])

    # Helper to populate selects
    def _choices_from_df(df: pd.DataFrame, label_cols: List[str]):
        if df.empty:
            return {}
        dfx = df.sort_values("date", ascending=False).head(50)
        out = {}
        for _, r in dfx.iterrows():
            parts: List[str] = []
            for c in label_cols:
                v = r.get(c, "")
                if pd.isna(v):
                    parts.append("")
                else:
                    parts.append(str(v))
            label = " | ".join([str(r.get("date", ""))] + parts + [f"#{str(r.get('id',''))[:6]}"])
            out[label] = r.get("id")
        return out

    @reactive.effect
    def _refresh_picks():
        ui.update_selectize("lift_pick", choices=_choices_from_df(lifts_df(), ["exercise", "weight_kg", "reps"]))
        ui.update_selectize("bw_pick", choices=_choices_from_df(bw_df(), ["weight_kg"]))
        ui.update_selectize("m_pick", choices=_choices_from_df(meas_df(), ["chest_cm", "waist_cm", "biceps_cm"]))

    # Add handlers
    @reactive.effect
    @reactive.event(input.btn_add_lift)
    def _add_lift():
        r = get_repo()
        if not r:
            print("[DEBUG] No repo available for add_lift")
            return
        payload = {
            "date": input.lift_date(),
            "exercise": input.lift_ex(),
            "weight_kg": input.lift_weight(),
            "reps": input.lift_reps(),
            "notes": input.lift_notes(),
        }
        print(f"[DEBUG] add_lift payload: {payload}")
        r.append("Lifts", payload)
        ui.notification_show("Lift added.")

    @reactive.effect
    @reactive.event(input.btn_add_bw)
    def _add_bw():
        r = get_repo()
        if not r:
            print("[DEBUG] No repo available for add_bw")
            return
        # Compute time string
        if input.bw_use_now():
            tstr = datetime.now().strftime("%H:%M")
        else:
            try:
                h = int(str(input.bw_hour()))
                m = int(str(input.bw_min()))
            except Exception:
                h, m = 0, 0
            tstr = f"{h:02d}:{m:02d}"
        payload = {
            "date": input.bw_date(),
            "time": tstr,
            "weight_kg": input.bw_weight(),
            "notes": input.bw_notes(),
        }
        print(f"[DEBUG] add_bw payload: {payload}")
        r.append("Bodyweight", payload)
        ui.notification_show("Bodyweight added.")

    @reactive.effect
    @reactive.event(input.btn_add_meas)
    def _add_meas():
        r = get_repo()
        if not r:
            print("[DEBUG] No repo available for add_meas")
            return
        payload = {
            "date": input.m_date(),
            "neck_cm": (None if input.m_neck_missing() else input.m_neck()),
            "shoulder_cm": (None if input.m_shoulder_missing() else input.m_shoulder()),
            "chest_cm": (None if input.m_chest_missing() else input.m_chest()),
            "waist_cm": (None if input.m_waist_missing() else input.m_waist()),
            "biceps_cm": (None if input.m_biceps_missing() else input.m_biceps()),
            "thigh_cm": (None if input.m_thigh_missing() else input.m_thigh()),
            "calf_cm": (None if input.m_calf_missing() else input.m_calf()),
        }
        print(f"[DEBUG] add_meas payload: {payload}")
        r.append("Measurements", payload)
        ui.notification_show("Measurements added.")

    @reactive.effect
    @reactive.event(input.btn_test)
    def _test_event():
        print("[DEBUG] test button clicked")

    # Load / Save / Delete — Lifts
    @reactive.effect
    @reactive.event(input.btn_load_lift)
    def _load_lift():
        sel = input.lift_pick()
        r = lifts_df().loc[lifts_df()["id"] == sel]
        if r.empty:
            ui.notification_show("Pick a lift to load", type="warning"); return
        row = r.iloc[0]
        session.send_input_message("lift_date", {"value": row["date"]})
        session.send_input_message("lift_ex", {"value": row.get("exercise")})
        session.send_input_message("lift_weight", {"value": float(row.get("weight_kg") or 0)})
        session.send_input_message("lift_reps", {"value": int(row.get("reps") or 0)})
        session.send_input_message("lift_notes", {"value": str(row.get("notes") or "")})

    @reactive.effect
    @reactive.event(input.btn_save_lift)
    def _edit_lift():
        r = get_repo()
        if not r: return
        sel = input.lift_pick()
        if not sel:
            ui.notification_show("Pick a lift to save", type="warning"); return
        r.update("Lifts", sel, {
            "date": input.lift_date(),
            "exercise": input.lift_ex(),
            "weight_kg": input.lift_weight(),
            "reps": input.lift_reps(),
            "notes": input.lift_notes(),
        })
        session.send_input_message("btn_save_lift", {})
        ui.notification_show("Lift updated.")

    @reactive.effect
    @reactive.event(input.btn_del_lift)
    def _delete_lift():
        r = get_repo()
        if not r: return
        sel = input.lift_pick()
        if not sel:
            ui.notification_show("Pick a lift to delete", type="warning"); return
        r.delete("Lifts", sel)
        ui.notification_show("Lift deleted.")
        session.send_input_message("delete_lift", {})

    # Load / Save / Delete — Bodyweight
    @reactive.effect
    @reactive.event(input.btn_load_bw)
    def _load_bw():
        sel = input.bw_pick()
        r = bw_df().loc[bw_df()["id"] == sel]
        if r.empty:
            ui.notification_show("Pick a record to load", type="warning"); return
        row = r.iloc[0]
        session.send_input_message("bw_date", {"value": row["date"]})
        session.send_input_message("bw_weight", {"value": float(row.get("weight_kg") or 0)})
        session.send_input_message("bw_notes", {"value": str(row.get("notes") or "")})
        # Populate time controls
        t = row.get("time")
        if t and isinstance(t, str) and ":" in t:
            hh, mm = t.split(":", 1)
            session.send_input_message("bw_use_now", {"value": False})
            session.send_input_message("bw_hour", {"value": f"{int(hh):02d}"})
            session.send_input_message("bw_min", {"value": f"{int(mm):02d}"})
        else:
            session.send_input_message("bw_use_now", {"value": True})

    @reactive.effect
    @reactive.event(input.btn_save_bw)
    def _edit_bw():
        r = get_repo()
        if not r: return
        sel = input.bw_pick()
        if not sel:
            ui.notification_show("Pick a bodyweight record to save", type="warning"); return
        if input.bw_use_now():
            tstr = datetime.now().strftime("%H:%M")
        else:
            try:
                h = int(str(input.bw_hour()))
                m = int(str(input.bw_min()))
            except Exception:
                h, m = 0, 0
            tstr = f"{h:02d}:{m:02d}"
        r.update("Bodyweight", sel, {"date": input.bw_date(), "time": tstr, "weight_kg": input.bw_weight(), "notes": input.bw_notes()})
        session.send_input_message("btn_save_bw", {})
        ui.notification_show("Bodyweight updated.")

    @reactive.effect
    @reactive.event(input.btn_del_bw)
    def _delete_bw():
        r = get_repo()
        if not r: return
        sel = input.bw_pick()
        print(f"[DEBUG] delete_bw raw selection: {sel!r}")
        if not sel:
            ui.notification_show("Pick a record to delete", type="warning"); return

        # Normalize selection shapes (selectize can sometimes return a dict or list)
        norm = None
        if isinstance(sel, dict):
            # sometimes comes as {'value': '<id>'}
            norm = sel.get("value") or sel.get("id")
        elif isinstance(sel, (list, tuple)):
            norm = sel[0] if sel else None
        else:
            norm = sel

        # If norm looks like a label (not found in df ids), try to map via choices
        bw_choices = _choices_from_df(bw_df(), ["weight_kg"]) if bw_df() is not None else {}
        # If norm matches a label, map to id
        if norm in bw_choices:
            norm = bw_choices.get(norm)

        print(f"[DEBUG] delete_bw normalized id: {norm!r}")

        if not norm:
            ui.notification_show("Could not determine record id to delete", type="warning"); return

        try:
            r.delete("Bodyweight", norm)
        except Exception as e:
            print(f"[ERROR] delete_bw failed: {e}")
            ui.notification_show(f"Delete failed: {e}", type="error")
            return

        ui.notification_show("Bodyweight deleted.")
        session.send_input_message("delete_bw", {})

    # Load / Save / Delete — Measurements
    @reactive.effect
    @reactive.event(input.btn_load_meas)
    def _load_meas():
        sel = input.m_pick()
        r = meas_df().loc[meas_df()["id"] == sel]
        if r.empty:
            ui.notification_show("Pick a record to load", type="warning"); return
        row = r.iloc[0]
        session.send_input_message("m_date", {"value": row["date"]})
        for fld, key, miss in [
            ("m_neck","neck_cm","m_neck_missing"),
            ("m_shoulder","shoulder_cm","m_shoulder_missing"),
            ("m_chest", "chest_cm","m_chest_missing"),
            ("m_waist","waist_cm","m_waist_missing"),
            ("m_biceps","biceps_cm","m_biceps_missing"),
            ("m_thigh","thigh_cm","m_thigh_missing"),
            ("m_calf","calf_cm","m_calf_missing"),
        ]:
            val = row.get(key)
            if pd.isna(val) or val is None or val == "":
                session.send_input_message(miss, {"value": True})
                session.send_input_message(fld, {"value": 0})
            else:
                session.send_input_message(miss, {"value": False})
                session.send_input_message(fld, {"value": float(val)})
            # Notes were removed from measurements; skip sending notes to UI

    @reactive.effect
    @reactive.event(input.btn_save_meas)
    def _edit_meas():
        r = get_repo()
        if not r: return
        sel = input.m_pick()
        if not sel:
            ui.notification_show("Pick a measurement to save", type="warning"); return
        r.update("Measurements", sel, {
            "date": input.m_date(),
            "neck_cm": (None if input.m_neck_missing() else input.m_neck()),
            "shoulder_cm": (None if input.m_shoulder_missing() else input.m_shoulder()),
            "chest_cm": (None if input.m_chest_missing() else input.m_chest()),
            "waist_cm": (None if input.m_waist_missing() else input.m_waist()),
            "biceps_cm": (None if input.m_biceps_missing() else input.m_biceps()),
            "thigh_cm": (None if input.m_thigh_missing() else input.m_thigh()),
            "calf_cm": (None if input.m_calf_missing() else input.m_calf()),
        })
        session.send_input_message("btn_save_meas", {})
        ui.notification_show("Measurements updated.")

    @reactive.effect
    @reactive.event(input.btn_del_meas)
    def _delete_meas():
        r = get_repo()
        if not r: return
        sel = input.m_pick()
        if not sel:
            ui.notification_show("Pick a measurement to delete", type="warning"); return
        r.delete("Measurements", sel)
        ui.notification_show("Measurements deleted.")
        session.send_input_message("delete_meas", {})

    # Charts & tables
    @output
    @render.plot
    def plot_1rm():
        d = lifts_df()
        if d.empty:
            fig, ax = plt.subplots(figsize=(10, 5))
            ax.text(0.5, 0.5, 'No lifts recorded yet\nAdd your first lift!', 
                   ha='center', va='center', fontsize=14, color='#6c757d')
            ax.set_xlim(0, 1)
            ax.set_ylim(0, 1)
            ax.axis('off')
            return fig
        chosen = input.exercise_filter() or BIG3
        d = d[d["exercise"].isin(chosen)].sort_values("date")
        if d.empty:
            fig, ax = plt.subplots(figsize=(10, 5))
            ax.text(0.5, 0.5, 'No data for selected exercises', 
                   ha='center', va='center', fontsize=14, color='#6c757d')
            ax.set_xlim(0, 1)
            ax.set_ylim(0, 1)
            ax.axis('off')
            return fig
        
        fig, ax = plt.subplots(figsize=(10, 5))
        ycol = 'tm' if input.rm_mode_tm() else '1rm'
        colors = {'Squat': '#0d6efd', 'Bench Press': '#dc3545', 'Deadlift': '#198754'}
        
        for exercise_name, grp in d.groupby(['exercise']):
            # Handle tuple from groupby
            ex_str = exercise_name[0] if isinstance(exercise_name, tuple) else exercise_name
            color = colors.get(ex_str, '#6c757d')
            ax.plot(grp['date'], grp[ycol], marker='o', linewidth=2.5, 
                   markersize=8, label=ex_str, color=color, alpha=0.9)
        
        ax.set_title("Training Max (90%)" if input.rm_mode_tm() else "Estimated 1RM", 
                    fontweight='bold', pad=15)
        ax.set_xlabel("Date", fontweight='bold')
        ax.set_ylabel("Training Max (kg)" if input.rm_mode_tm() else "1RM (kg)", fontweight='bold')
        ax.legend(title="Exercise", loc='best', framealpha=0.9)
        ax.grid(True, alpha=0.3)
        plt.tight_layout()
        return fig

    @output
    @render.plot
    def plot_bw():
        d = bw_df().sort_values("date")
        if d.empty:
            fig, ax = plt.subplots(figsize=(8, 5))
            ax.text(0.5, 0.5, 'No bodyweight recorded yet\nAdd your first measurement!', 
                   ha='center', va='center', fontsize=14, color='#6c757d')
            ax.set_xlim(0, 1)
            ax.set_ylim(0, 1)
            ax.axis('off')
            return fig
        
        fig, ax = plt.subplots(figsize=(8, 5))
        ax.plot(d["date"], d["weight_kg"], marker='o', linewidth=2.5, 
               markersize=8, color='#198754', alpha=0.9)
        ax.fill_between(d["date"], d["weight_kg"], alpha=0.15, color='#198754')
        ax.set_title("Bodyweight Tracking", fontweight='bold', pad=15)
        ax.set_xlabel("Date", fontweight='bold')
        ax.set_ylabel("Weight (kg)", fontweight='bold')
        ax.grid(True, alpha=0.3)
        plt.tight_layout()
        return fig

    @output
    @render.plot
    def plot_meas():
        d = meas_df().sort_values("date")
        if d.empty:
            fig, ax = plt.subplots(figsize=(10, 5))
            ax.text(0.5, 0.5, 'No measurements recorded yet\nAdd your first measurements!', 
                   ha='center', va='center', fontsize=14, color='#6c757d')
            ax.set_xlim(0, 1)
            ax.set_ylim(0, 1)
            ax.axis('off')
            return fig
        cols = [c for c in ["neck_cm","shoulder_cm","chest_cm","waist_cm","biceps_cm","thigh_cm","calf_cm"] if c in d.columns]
        if not cols:
            fig, ax = plt.subplots(figsize=(10, 5))
            ax.text(0.5, 0.5, 'No measurement columns available', 
                   ha='center', va='center', fontsize=14, color='#6c757d')
            ax.set_xlim(0, 1)
            ax.set_ylim(0, 1)
            ax.axis('off')
            return fig
        
        fig, ax = plt.subplots(figsize=(10, 5))
        color_map = {
            "neck_cm": "#0d6efd",
            "shoulder_cm": "#6610f2",
            "chest_cm": "#198754",
            "waist_cm": "#dc3545",
            "biceps_cm": "#fd7e14",
            "thigh_cm": "#0dcaf0",
            "calf_cm": "#d63384",
        }
        for col in cols:
            if d[col].notnull().any():
                ax.plot(d["date"], d[col], marker='o', linewidth=2.5, markersize=8,
                       label=col.replace('_cm','').capitalize(), 
                       color=color_map.get(col, None), alpha=0.9)
        ax.set_title("Body Measurements Progress", fontweight='bold', pad=15)
        ax.set_xlabel("Date", fontweight='bold')
        ax.set_ylabel("Measurements (cm)", fontweight='bold')
        ax.legend(title="Body Part", loc='best', framealpha=0.9, ncol=2)
        ax.grid(True, alpha=0.3)
        plt.tight_layout()
        return fig

    @output
    @render.data_frame
    def tbl_lifts():
        d = lifts_df()
        if d.empty:
            import pandas as pd
            from .utils import REQUIRED_TABS
            return pd.DataFrame(columns=REQUIRED_TABS["Lifts"]+["1rm","tm"])  # type: ignore
        cols = [c for c in ["date","exercise","weight_kg","reps","1rm","tm","notes","id"] if c in d.columns]
        return d.sort_values("date", ascending=False).head(20)[cols]

    @output
    @render.data_frame
    def tbl_bw():
        d = bw_df()
        if d.empty:
            import pandas as pd
            from .utils import REQUIRED_TABS
            return pd.DataFrame(columns=REQUIRED_TABS["Bodyweight"])  # type: ignore
        return d.sort_values("date", ascending=False).head(20)

    @output
    @render.data_frame
    def tbl_meas():
        d = meas_df()
        if d.empty:
            import pandas as pd
            from .utils import REQUIRED_TABS
            return pd.DataFrame(columns=REQUIRED_TABS["Measurements"])  # type: ignore
        # Only expose the tracked measurement columns (drop hips_cm, arm_cm, notes if present)
        cols = [c for c in ["date","neck_cm","shoulder_cm","chest_cm","waist_cm","biceps_cm","thigh_cm","calf_cm","id"] if c in d.columns]
        return d.sort_values("date", ascending=False).head(20)[cols]

