from __future__ import annotations

from datetime import datetime
from typing import List, cast

import numpy as np
import pandas as pd
import plotly.graph_objects as go
from shiny import Inputs, Outputs, Session, reactive, render, ui
from shinywidgets import render_plotly

from .repos import Repo, repo_factory
from .utils import BIG3, REQUIRED_TABS, epley_1rm, epley_training_max


def server(input: Inputs, output: Outputs, session: Session):
    repo = reactive.value(cast(Repo, None))

    def get_repo():
        r = repo.get()
        return cast(Repo, r) if r is not None and hasattr(r, 'read_df') else None

    def _normalize_decimal(value) -> float:
        """Convert numeric input to float, accepting both comma and period as decimal separator."""
        if value is None:
            return 0.0
        if isinstance(value, (int, float)):
            return float(value)
        # Convert string, replacing comma with period
        try:
            return float(str(value).replace(',', '.'))
        except (ValueError, AttributeError):
            return 0.0

    @reactive.effect
    def _init_repo():
        repo.set(repo_factory())
        # Debug: print Bodyweight data from both sources at startup
        try:
            r = repo.get()
            if r and hasattr(r, 'read_df'):
                df = r.read_df("Bodyweight")
                print(f"[DEBUG] Bodyweight df rows: {0 if df is None else len(df)}")
                if df is not None and not df.empty:
                    print("[DEBUG] Bodyweight df sample:\n" + df.head(10).to_string(index=False))
                else:
                    print("[DEBUG] Bodyweight df is empty")
                # If using CombinedRepo, also print raw Google Sheet dataframe for verification
                try:
                    if hasattr(r, 'secondary') and r.secondary is not None:
                        gdf = r.secondary.read_df("Bodyweight")
                        if gdf is None or gdf.empty:
                            print("[DEBUG] Google Sheet Bodyweight is empty or not readable")
                        else:
                            print("[DEBUG] Google Sheet Bodyweight full df:\n" + gdf.to_string(index=False))
                    else:
                        print("[DEBUG] No Sheets secondary repo attached")
                except Exception as ge:
                    print(f"[DEBUG] Failed to read Google Sheet Bodyweight df: {ge}")
        except Exception as e:
            print(f"[DEBUG] Failed to read Bodyweight df: {e}")

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

    @reactive.calc
    def inbody_df():
        r = get_repo()
        df = r.read_df("InBody") if r else pd.DataFrame(columns=REQUIRED_TABS["InBody"])
        if df.empty:
            return df
        
        # Sort by date
        df = df.sort_values("date").copy()

        # Ensure all expected numeric columns exist; fill missing with NaN
        numeric_cols = [
            "inbody_score", "weight_kg", "skeletal_muscle_kg_total", "body_fat_kg_total",
            "body_fat_percent", "visceral_fat_level", "bmr_kcal",
            "muscle_right_arm_kg", "muscle_left_arm_kg", "muscle_trunk_kg",
            "muscle_right_leg_kg", "muscle_left_leg_kg",
            "fat_right_arm_kg", "fat_left_arm_kg", "fat_trunk_kg",
            "fat_right_leg_kg", "fat_left_leg_kg"
        ]
        for col in numeric_cols:
            if col not in df.columns:
                df[col] = float("nan")
            df[col] = pd.to_numeric(df[col], errors='coerce')

        # Compute derived metrics safely
        # Arm asymmetry
        df['arm_avg'] = (df['muscle_right_arm_kg'] + df['muscle_left_arm_kg']) / 2
        df['arm_asym_pct'] = ((df['muscle_right_arm_kg'] - df['muscle_left_arm_kg']) / df['arm_avg'].replace(0, pd.NA)) * 100

        # Leg asymmetry
        df['leg_avg'] = (df['muscle_right_leg_kg'] + df['muscle_left_leg_kg']) / 2
        df['leg_asym_pct'] = ((df['muscle_right_leg_kg'] - df['muscle_left_leg_kg']) / df['leg_avg'].replace(0, pd.NA)) * 100

        # Trunk muscle share
        df['trunk_muscle_share_pct'] = (df['muscle_trunk_kg'] / df['skeletal_muscle_kg_total'].replace(0, pd.NA)) * 100

        # Upper/lower ratio
        lower = (df['muscle_right_leg_kg'] + df['muscle_left_leg_kg']).replace(0, pd.NA)
        df['upper_lower_ratio'] = (
            df['muscle_right_arm_kg'] + df['muscle_left_arm_kg'] + df['muscle_trunk_kg']
        ) / lower

        # Trunk to limb fat ratio
        limb_fat = (
            df['fat_right_arm_kg'] + df['fat_left_arm_kg'] +
            df['fat_right_leg_kg'] + df['fat_left_leg_kg']
        ).replace(0, pd.NA)
        df['trunk_to_limb_fat_ratio'] = df['fat_trunk_kg'] / limb_fat

        # Month-to-month deltas
        df['delta_skeletal_muscle_kg'] = df['skeletal_muscle_kg_total'].diff()
        df['delta_body_fat_kg'] = df['body_fat_kg_total'].diff()
        df['delta_weight_kg'] = df['weight_kg'].diff()

        return df

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

    # Helper to normalize selectize selections (id, dict, list, or label)
    def _normalize_pick(sel, df: pd.DataFrame, label_cols: List[str]):
        norm = None
        if isinstance(sel, dict):
            norm = sel.get("value") or sel.get("id")
        elif isinstance(sel, (list, tuple)):
            norm = sel[0] if sel else None
        else:
            norm = sel
        # Map label to id if needed
        choices = _choices_from_df(df, label_cols) if df is not None else {}
        if norm in choices:
            norm = choices.get(norm)
        return norm

    @reactive.effect
    def _refresh_picks():
        ui.update_selectize("lift_pick", choices=_choices_from_df(lifts_df(), ["exercise", "weight_kg", "reps"]))
        ui.update_selectize("bw_pick", choices=_choices_from_df(bw_df(), ["weight_kg"]))
        ui.update_selectize("m_pick", choices=_choices_from_df(meas_df(), ["chest_cm", "waist_cm", "biceps_cm"]))
        ui.update_selectize("ib_pick", choices=_choices_from_df(inbody_df(), ["inbody_score", "weight_kg"]))

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
            "weight_kg": _normalize_decimal(input.lift_weight()),
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
            "weight_kg": _normalize_decimal(input.bw_weight()),
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
            "weight_kg": _normalize_decimal(input.m_weight()),
            "neck_cm": (None if input.m_neck_missing() else _normalize_decimal(input.m_neck())),
            "shoulder_cm": (None if input.m_shoulder_missing() else _normalize_decimal(input.m_shoulder())),
            "chest_cm": (None if input.m_chest_missing() else _normalize_decimal(input.m_chest())),
            "waist_cm": (None if input.m_waist_missing() else _normalize_decimal(input.m_waist())),
            "biceps_cm": (None if input.m_biceps_missing() else _normalize_decimal(input.m_biceps())),
            "thigh_cm": (None if input.m_thigh_missing() else _normalize_decimal(input.m_thigh())),
            "calf_cm": (None if input.m_calf_missing() else _normalize_decimal(input.m_calf())),
        }
        print(f"[DEBUG] add_meas payload: {payload}")
        r.append("Measurements", payload)
        ui.notification_show("Measurements added.")

    @reactive.effect
    @reactive.event(input.btn_test)
    def _test_event():
        print("[DEBUG] test button clicked")

    # Helper functions for common patterns
    def _empty_figure(message: str = "No data available"):
        """Create empty Plotly figure with message"""
        fig = go.Figure()
        fig.update_layout(
            template='plotly_white',
            autosize=True,
            margin=dict(l=50, r=50, t=50, b=50),
            annotations=[dict(
                text=message,
                showarrow=False,
                x=0.5, y=0.5,
                xref="paper", yref="paper",
                font=dict(color="#6c757d", size=16)
            )]
        )
        return fig

    def _render_stat_with_delta(value, delta=None, unit="kg", size="2rem"):
        """Helper to render stat value with optional delta indicator"""
        if value is None:
            return ui.span("—", style=f"font-size: {size};")
        value_str = f"{value:.1f} {unit}"
        if delta and delta != 0:
            arrow = "↑" if delta > 0 else "↓"
            color = "green" if delta > 0 else "red"
            return ui.HTML(
                f'<span style="font-size: {size};">{value_str}</span><br>'
                f'<span style="color: {color}; font-size: 1rem;">{arrow} {abs(delta):.1f}</span>'
            )
        return ui.span(value_str, style=f"font-size: {size};")

    def _get_exercise_stat(exercise_name: str):
        """Helper to get latest 1RM stat for an exercise with delta"""
        d = lifts_df()
        ex_data = d[d["exercise"] == exercise_name].sort_values("date", ascending=False)
        if ex_data.empty:
            return _render_stat_with_delta(None)
        latest = ex_data.iloc[0]['1rm']
        delta = None
        if len(ex_data) > 1:
            delta = latest - ex_data.iloc[1]['1rm']
        return _render_stat_with_delta(latest, delta)

    def _get_inbody_stat(column: str, decimals: int = 1, unit: str = ""):
        """Helper to get latest InBody stat value"""
        d = inbody_df()
        if d.empty:
            return ui.span("—", style="font-size: 2rem;")
        latest = d.iloc[-1]
        value = latest.get(column, 0)
        formatted = f"{value:.{decimals}f}{unit}"
        return ui.span(formatted, style="font-size: 2rem;")

    # Load / Save / Delete — Lifts
    loaded_lift_info = reactive.value("")

    @output
    @render.ui
    def lift_loaded_info():
        info = loaded_lift_info.get()
        if not info:
            return ui.div()
        return ui.div(
            ui.HTML(info),
            style="padding: 10px; margin: 10px 0; background: #e7f3ff; border-left: 4px solid #0d6efd; border-radius: 4px; font-size: 0.9rem;"
        )

    @reactive.effect
    @reactive.event(input.btn_load_lift)
    def _load_lift():
        raw = input.lift_pick()
        sel = _normalize_pick(raw, lifts_df(), ["exercise", "weight_kg", "reps"])
        r = lifts_df().loc[lifts_df()["id"] == sel]
        if r.empty:
            ui.notification_show("Pick a lift to load", type="warning")
            return
        row = r.iloc[0]
        session.send_input_message("lift_date", {"value": str(row["date"])})
        session.send_input_message("lift_ex", {"value": row.get("exercise")})
        session.send_input_message("lift_weight", {"value": float(row.get("weight_kg") or 0)})
        session.send_input_message("lift_reps", {"value": int(row.get("reps") or 0)})
        session.send_input_message("lift_notes", {"value": str(row.get("notes") or "")})
        loaded_lift_info.set(
            f"<strong>✓ Loaded:</strong> {row.get('exercise')} • "
            f"{row.get('weight_kg'):.1f} kg × {row.get('reps')} reps • "
            f"{row['date']} • <code>#{str(sel)[:8]}</code>"
        )

    @reactive.effect
    @reactive.event(input.btn_save_lift)
    def _edit_lift():
        r = get_repo()
        if not r:
            return
        sel = _normalize_pick(input.lift_pick(), lifts_df(), ["exercise", "weight_kg", "reps"])
        if not sel:
            ui.notification_show("Pick a lift to save", type="warning")
            return
        r.update("Lifts", sel, {
            "date": input.lift_date(),
            "exercise": input.lift_ex(),
            "weight_kg": _normalize_decimal(input.lift_weight()),
            "reps": input.lift_reps(),
            "notes": input.lift_notes(),
        })
        ui.notification_show("Lift updated.")

    @reactive.effect
    @reactive.event(input.btn_del_lift)
    def _delete_lift():
        r = get_repo()
        if not r:
            return
        sel = _normalize_pick(input.lift_pick(), lifts_df(), ["exercise", "weight_kg", "reps"])
        if not sel:
            ui.notification_show("Pick a lift to delete", type="warning")
            return
        r.delete("Lifts", sel)
        ui.notification_show("Lift deleted.")

    # Load / Save / Delete — Bodyweight
    loaded_bw_info = reactive.value("")

    @output
    @render.ui
    def bw_loaded_info():
        info = loaded_bw_info.get()
        if not info:
            return ui.div()
        return ui.div(
            ui.HTML(info),
            style="padding: 10px; margin: 10px 0; background: #d1f2e8; border-left: 4px solid #198754; border-radius: 4px; font-size: 0.9rem;"
        )

    @reactive.effect
    @reactive.event(input.btn_load_bw)
    def _load_bw():
        raw = input.bw_pick()
        sel = _normalize_pick(raw, bw_df(), ["weight_kg"])
        r = bw_df().loc[bw_df()["id"] == sel]
        if r.empty:
            ui.notification_show("Pick a record to load", type="warning")
            return
        row = r.iloc[0]
        session.send_input_message("bw_date", {"value": str(row["date"])})
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
        loaded_bw_info.set(
            f"<strong>✓ Loaded:</strong> {row.get('weight_kg'):.1f} kg • "
            f"{row['date']} {row.get('time', '')} • <code>#{str(sel)[:8]}</code>"
        )

    @reactive.effect
    @reactive.event(input.btn_save_bw)
    def _edit_bw():
        r = get_repo()
        if not r:
            return
        sel = _normalize_pick(input.bw_pick(), bw_df(), ["weight_kg"])
        if not sel:
            ui.notification_show("Pick a bodyweight record to save", type="warning")
            return
        if input.bw_use_now():
            tstr = datetime.now().strftime("%H:%M")
        else:
            try:
                h = int(str(input.bw_hour()))
                m = int(str(input.bw_min()))
            except Exception:
                h, m = 0, 0
            tstr = f"{h:02d}:{m:02d}"
        r.update("Bodyweight", sel, {
            "date": input.bw_date(),
            "time": tstr,
            "weight_kg": _normalize_decimal(input.bw_weight()),
            "notes": input.bw_notes(),
        })
        ui.notification_show("Bodyweight updated.")

    @reactive.effect
    @reactive.event(input.btn_del_bw)
    def _delete_bw():
        r = get_repo()
        if not r:
            return
        raw = input.bw_pick()
        norm = _normalize_pick(raw, bw_df(), ["weight_kg"])
        if not norm:
            ui.notification_show("Could not determine record id to delete", type="warning")
            
            return
        try:
            r.delete("Bodyweight", norm)
        except Exception as e:
            ui.notification_show(f"Delete failed: {e}", type="error")
            return
        ui.notification_show("Bodyweight deleted.")

    # Load / Save / Delete — Measurements
    loaded_meas_info = reactive.value("")

    @output
    @render.ui
    def meas_loaded_info():
        info = loaded_meas_info.get()
        if not info:
            return ui.div()
        return ui.div(
            ui.HTML(info),
            style="padding: 10px; margin: 10px 0; background: #cff4fc; border-left: 4px solid #0dcaf0; border-radius: 4px; font-size: 0.9rem;"
        )

    @reactive.effect
    @reactive.event(input.btn_load_meas)
    def _load_meas():
        raw = input.m_pick()
        sel = _normalize_pick(raw, meas_df(), ["chest_cm", "waist_cm", "biceps_cm"])
        r = meas_df().loc[meas_df()["id"] == sel]
        if r.empty:
            ui.notification_show("Pick a record to load", type="warning")
            return
        row = r.iloc[0]
        session.send_input_message("m_date", {"value": str(row["date"])})
        session.send_input_message("m_weight", {"value": float(row.get("weight_kg") or 0)})
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
        # Build summary of loaded measurements
        parts = []
        for col in ["chest_cm", "waist_cm", "biceps_cm", "thigh_cm"]:
            val = row.get(col)
            if pd.notna(val) and val:
                parts.append(f"{col.replace('_cm','').title()}: {val:.1f}cm")
        summary = ", ".join(parts[:3]) if parts else "Multiple measurements"
        loaded_meas_info.set(
            f"<strong>✓ Loaded:</strong> {summary} • "
            f"{row['date']} • <code>#{str(sel)[:8]}</code>"
        )

    @reactive.effect
    @reactive.event(input.btn_save_meas)
    def _edit_meas():
        r = get_repo()
        if not r: 
            return
        sel = _normalize_pick(input.m_pick(), meas_df(), ["chest_cm", "waist_cm", "biceps_cm"])
        if not sel:
            ui.notification_show("Pick a measurement to save", type="warning")
            return
        r.update("Measurements", sel, {
            "date": input.m_date(),
            "weight_kg": _normalize_decimal(input.m_weight()),
            "neck_cm": (None if input.m_neck_missing() else _normalize_decimal(input.m_neck())),
            "shoulder_cm": (None if input.m_shoulder_missing() else _normalize_decimal(input.m_shoulder())),
            "chest_cm": (None if input.m_chest_missing() else _normalize_decimal(input.m_chest())),
            "waist_cm": (None if input.m_waist_missing() else _normalize_decimal(input.m_waist())),
            "biceps_cm": (None if input.m_biceps_missing() else _normalize_decimal(input.m_biceps())),
            "thigh_cm": (None if input.m_thigh_missing() else _normalize_decimal(input.m_thigh())),
            "calf_cm": (None if input.m_calf_missing() else _normalize_decimal(input.m_calf())),
        })
        ui.notification_show("Measurements updated.")

    @reactive.effect
    @reactive.event(input.btn_del_meas)
    def _delete_meas():
        r = get_repo()
        if not r:
            return
        sel = _normalize_pick(input.m_pick(), meas_df(), ["chest_cm", "waist_cm", "biceps_cm"])
        if not sel:
            ui.notification_show("Pick a measurement to delete", type="warning")
            return
        r.delete("Measurements", sel)
        ui.notification_show("Measurements deleted.")

    # Load / Save / Delete — InBody
    loaded_ib_info = reactive.value("")

    @output
    @render.ui
    def ib_loaded_info():
        info = loaded_ib_info.get()
        if not info:
            return ui.div()
        return ui.div(
            ui.HTML(info),
            style="padding: 10px; margin: 10px 0; background: #fff3cd; border-left: 4px solid #ffc107; border-radius: 4px; font-size: 0.9rem;"
        )

    @reactive.effect
    @reactive.event(input.btn_add_ib)
    def _add_inbody():
        r = get_repo()
        if not r:
            print("[DEBUG] No repo available for add_inbody")
            return
        payload = {
            "date": input.ib_date(),
            "inbody_score": _normalize_decimal(input.ib_score()),
            "weight_kg": _normalize_decimal(input.ib_weight()),
            "skeletal_muscle_kg_total": _normalize_decimal(input.ib_muscle_total()),
            "body_fat_kg_total": _normalize_decimal(input.ib_fat_total()),
            "body_fat_percent": _normalize_decimal(input.ib_fat_pct()),
            "visceral_fat_level": _normalize_decimal(input.ib_visceral()),
            "bmr_kcal": _normalize_decimal(input.ib_bmr()),
            "muscle_right_arm_kg": _normalize_decimal(input.ib_muscle_rarm()),
            "muscle_left_arm_kg": _normalize_decimal(input.ib_muscle_larm()),
            "muscle_trunk_kg": _normalize_decimal(input.ib_muscle_trunk()),
            "muscle_right_leg_kg": _normalize_decimal(input.ib_muscle_rleg()),
            "muscle_left_leg_kg": _normalize_decimal(input.ib_muscle_lleg()),
            "fat_right_arm_kg": _normalize_decimal(input.ib_fat_rarm()),
            "fat_left_arm_kg": _normalize_decimal(input.ib_fat_larm()),
            "fat_trunk_kg": _normalize_decimal(input.ib_fat_trunk()),
            "fat_right_leg_kg": _normalize_decimal(input.ib_fat_rleg()),
            "fat_left_leg_kg": _normalize_decimal(input.ib_fat_lleg()),
            "notes": input.ib_notes(),
        }
        print(f"[DEBUG] add_inbody payload: {payload}")
        r.append("InBody", payload)
        ui.notification_show("InBody added.")

    @reactive.effect
    @reactive.event(input.btn_load_ib)
    def _load_inbody():
        sel = input.ib_pick()
        norm = _normalize_pick(sel, inbody_df(), ["inbody_score", "weight_kg"])
        r = inbody_df().loc[inbody_df()["id"] == norm]
        if r.empty:
            ui.notification_show("Pick an InBody entry to load", type="warning")
            return
        row = r.iloc[0]
        session.send_input_message("ib_date", {"value": str(row["date"])})
        session.send_input_message("ib_score", {"value": float(row.get("inbody_score") or 0)})
        session.send_input_message("ib_weight", {"value": float(row.get("weight_kg") or 0)})
        session.send_input_message("ib_muscle_total", {"value": float(row.get("skeletal_muscle_kg_total") or 0)})
        session.send_input_message("ib_fat_total", {"value": float(row.get("body_fat_kg_total") or 0)})
        session.send_input_message("ib_fat_pct", {"value": float(row.get("body_fat_percent") or 0)})
        session.send_input_message("ib_visceral", {"value": float(row.get("visceral_fat_level") or 0)})
        session.send_input_message("ib_bmr", {"value": float(row.get("bmr_kcal") or 0)})
        session.send_input_message("ib_muscle_rarm", {"value": float(row.get("muscle_right_arm_kg") or 0)})
        session.send_input_message("ib_muscle_larm", {"value": float(row.get("muscle_left_arm_kg") or 0)})
        session.send_input_message("ib_muscle_trunk", {"value": float(row.get("muscle_trunk_kg") or 0)})
        session.send_input_message("ib_muscle_rleg", {"value": float(row.get("muscle_right_leg_kg") or 0)})
        session.send_input_message("ib_muscle_lleg", {"value": float(row.get("muscle_left_leg_kg") or 0)})
        session.send_input_message("ib_fat_rarm", {"value": float(row.get("fat_right_arm_kg") or 0)})
        session.send_input_message("ib_fat_larm", {"value": float(row.get("fat_left_arm_kg") or 0)})
        session.send_input_message("ib_fat_trunk", {"value": float(row.get("fat_trunk_kg") or 0)})
        session.send_input_message("ib_fat_rleg", {"value": float(row.get("fat_right_leg_kg") or 0)})
        session.send_input_message("ib_fat_lleg", {"value": float(row.get("fat_left_leg_kg") or 0)})
        session.send_input_message("ib_notes", {"value": str(row.get("notes") or "")})
        loaded_ib_info.set(
            f"<strong>✓ Loaded:</strong> InBody Score {row.get('inbody_score'):.0f} • "
            f"{row.get('weight_kg'):.1f} kg • Muscle {row.get('skeletal_muscle_kg_total'):.1f} kg • "
            f"Fat {row.get('body_fat_percent'):.1f}% • {row['date']} • <code>#{str(norm)[:8]}</code>"
        )

    @reactive.effect
    @reactive.event(input.btn_save_ib)
    def _edit_inbody():
        r = get_repo()
        if not r: 
            return
        sel = input.ib_pick()
        if not sel:
            ui.notification_show("Pick an InBody entry to save", type="warning")
            return
        r.update("InBody", sel, {
            "date": input.ib_date(),
            "inbody_score": _normalize_decimal(input.ib_score()),
            "weight_kg": _normalize_decimal(input.ib_weight()),
            "skeletal_muscle_kg_total": _normalize_decimal(input.ib_muscle_total()),
            "body_fat_kg_total": _normalize_decimal(input.ib_fat_total()),
            "body_fat_percent": _normalize_decimal(input.ib_fat_pct()),
            "visceral_fat_level": _normalize_decimal(input.ib_visceral()),
            "bmr_kcal": _normalize_decimal(input.ib_bmr()),
            "muscle_right_arm_kg": _normalize_decimal(input.ib_muscle_rarm()),
            "muscle_left_arm_kg": _normalize_decimal(input.ib_muscle_larm()),
            "muscle_trunk_kg": _normalize_decimal(input.ib_muscle_trunk()),
            "muscle_right_leg_kg": _normalize_decimal(input.ib_muscle_rleg()),
            "muscle_left_leg_kg": _normalize_decimal(input.ib_muscle_lleg()),
            "fat_right_arm_kg": _normalize_decimal(input.ib_fat_rarm()),
            "fat_left_arm_kg": _normalize_decimal(input.ib_fat_larm()),
            "fat_trunk_kg": _normalize_decimal(input.ib_fat_trunk()),
            "fat_right_leg_kg": _normalize_decimal(input.ib_fat_rleg()),
            "fat_left_leg_kg": _normalize_decimal(input.ib_fat_lleg()),
            "notes": input.ib_notes(),
        })
        ui.notification_show("InBody updated.")

    @reactive.effect
    @reactive.event(input.btn_del_ib)
    def _delete_inbody():
        r = get_repo()
        if not r: 
            return
        sel = input.ib_pick()
        norm = _normalize_pick(sel, inbody_df(), ["inbody_score", "weight_kg"])
        if not norm:
            ui.notification_show("Could not determine record id to delete", type="warning")
            return
        try:
            r.delete("InBody", norm)
        except Exception as e:
            ui.notification_show(f"Delete failed: {e}", type="error")
            return
        ui.notification_show("InBody deleted.")

    # Summary statistics
    @output
    @render.ui
    def stat_squat():
        return _get_exercise_stat("Squat")

    @output
    @render.ui
    def stat_bench():
        return _get_exercise_stat("Bench")

    @output
    @render.ui
    def stat_deadlift():
        return _get_exercise_stat("Deadlift")

    @output
    @render.ui
    def stat_bodyweight():
        d = bw_df().sort_values("date", ascending=False)
        if d.empty:
            return _render_stat_with_delta(None)
        latest = d.iloc[0]['weight_kg']
        delta = None
        if len(d) > 1:
            delta = latest - d.iloc[1]['weight_kg']
        
        # Custom rendering for bodyweight with specific colors
        if delta and delta != 0:
            arrow = "↑" if delta > 0 else "↓"
            color = "#d63384" if delta > 0 else "#0dcaf0"
            return ui.HTML(
                f'<div style="display:flex; flex-direction:column; align-items:center;">'
                f'<span style="font-size: 2rem; font-weight:600;">{latest:.1f} kg</span>'
                f'<span style="margin-top:4px; color:{color}; font-size:1rem; font-weight:600;">{arrow} {abs(delta):.1f}</span>'
                f'</div>'
            )
        return ui.span(f"{latest:.1f} kg", style="font-size: 2rem;")

    # Charts & tables
    @output
    @render_plotly
    def plot_1rm():
        d = lifts_df()
        if d.empty:
            return _empty_figure("No lifts recorded yet")
        chosen = input.exercise_filter() or BIG3
        d = d[d["exercise"].isin(chosen)].sort_values("date")
        if d.empty:
            return _empty_figure("No data for selected exercises")
        
        ycol = 'tm' if input.rm_mode_tm() else '1rm'
        colors = {'Squat': '#0d6efd', 'Bench': '#dc3545', 'Deadlift': '#198754'}
        
        fig = go.Figure()
        for exercise_name, grp in d.groupby(['exercise']):
            ex_str = exercise_name[0] if isinstance(exercise_name, tuple) else exercise_name
            color = colors.get(ex_str, '#6c757d')
            fig.add_trace(go.Scatter(
                x=grp['date'],
                y=grp[ycol],
                mode='lines+markers',
                name=ex_str,
                line=dict(color=color, width=3),
                marker=dict(size=8, color=color),
                hovertemplate='<b>%{fullData.name}</b><br>Date: %{x}<br>Weight: %{y:.1f} kg<extra></extra>'
            ))
        
        title = "Training Max (90%)" if input.rm_mode_tm() else "Estimated 1RM"
        ylabel = "Training Max (kg)" if input.rm_mode_tm() else "1RM (kg)"
        
        fig.update_layout(
            title=dict(text=title, font=dict(size=16, weight='bold')),
            xaxis_title="Date",
            yaxis_title=ylabel,
            hovermode='closest',
            template='plotly_white',
            autosize=True,
            margin=dict(l=50, r=50, t=50, b=50),
            legend=dict(title=dict(text="Exercise"), orientation='v', yanchor='top', y=1, xanchor='left', x=1.02)
        )
        return fig

    @output
    @render_plotly
    def plot_bw():
        d = bw_df().sort_values("date")
        if d.empty:
            return _empty_figure("No bodyweight recorded yet")
        
        fig = go.Figure()
        fig.add_trace(go.Scatter(
            x=d["date"],
            y=d["weight_kg"],
            mode='lines+markers',
            name='Bodyweight',
            line=dict(color='#198754', width=3),
            marker=dict(size=8, color='#198754'),
            fill='tozeroy',
            fillcolor='rgba(25, 135, 84, 0.15)',
            hovertemplate='Date: %{x}<br>Weight: %{y:.1f} kg<extra></extra>'
        ))
        
        fig.update_layout(
            title=dict(text="Bodyweight Tracking", font=dict(size=16, weight='bold')),
            xaxis_title="Date",
            yaxis_title="Weight (kg)",
            hovermode='closest',
            template='plotly_white',
            autosize=True,
            margin=dict(l=50, r=50, t=50, b=50),
            showlegend=False
        )
        return fig

    @output
    @render_plotly
    def plot_meas():
        d = meas_df().sort_values("date")
        if d.empty:
            return _empty_figure("No measurements recorded yet")
        cols = [c for c in ["neck_cm","shoulder_cm","chest_cm","waist_cm","biceps_cm","thigh_cm","calf_cm"] if c in d.columns]
        if not cols:
            return _empty_figure("No measurement columns available")
        
        fig = go.Figure()
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
                fig.add_trace(go.Scatter(
                    x=d["date"],
                    y=d[col],
                    mode='lines+markers',
                    name=col.replace('_cm','').capitalize(),
                    line=dict(color=color_map.get(col, '#6c757d'), width=3),
                    marker=dict(size=8, color=color_map.get(col, '#6c757d')),
                    hovertemplate='<b>%{fullData.name}</b><br>Date: %{x}<br>Size: %{y:.1f} cm<extra></extra>'
                ))
        
        fig.update_layout(
            title=dict(text="Body Measurements Progress", font=dict(size=16, weight='bold')),
            xaxis_title="Date",
            yaxis_title="Measurements (cm)",
            hovermode='closest',
            template='plotly_white',
            autosize=True,
            margin=dict(l=50, r=50, t=50, b=50),
            legend=dict(title=dict(text="Body Part"), orientation='v', yanchor='top', y=1, xanchor='left', x=1.02)
        )
        return fig

    # InBody simple stat boxes - using helper for consistency
    @output
    @render.ui
    def ib_stat_score():
        return _get_inbody_stat('inbody_score', decimals=0)

    @output
    @render.ui
    def ib_stat_muscle():
        return _get_inbody_stat('skeletal_muscle_kg_total', unit=' kg')

    @output
    @render.ui
    def ib_stat_fatkg():
        return _get_inbody_stat('body_fat_kg_total', unit=' kg')

    @output
    @render.ui
    def ib_stat_fatpct():
        return _get_inbody_stat('body_fat_percent', unit='%')

    @output
    @render.ui
    def ib_stat_visceral():
        return _get_inbody_stat('visceral_fat_level', decimals=0)

    @output
    @render.ui
    def ib_stat_bmr():
        return _get_inbody_stat('bmr_kcal', decimals=0)

    @output
    @render.ui
    def ib_progress_stats():
        d = inbody_df()
        if d.empty or len(d) < 2:
            return ui.HTML(
                "<div style='padding:20px; text-align:center; color:#6c757d;'>"
                "<p style='font-size:1.1rem;'>📊 Progress statistics will appear here once you have multiple InBody measurements.</p>"
                "</div>"
            )
        
        latest = d.iloc[-1]
        first = d.iloc[0]
        prev = d.iloc[-2] if len(d) >= 2 else first
        
        # Calculate changes
        muscle_change = latest.get('skeletal_muscle_kg_total', 0) - first.get('skeletal_muscle_kg_total', 0)
        fat_change = latest.get('body_fat_kg_total', 0) - first.get('body_fat_kg_total', 0)
        weight_change = latest.get('weight_kg', 0) - first.get('weight_kg', 0)
        score_change = latest.get('inbody_score', 0) - first.get('inbody_score', 0)
        
        muscle_recent = latest.get('skeletal_muscle_kg_total', 0) - prev.get('skeletal_muscle_kg_total', 0)
        fat_recent = latest.get('body_fat_kg_total', 0) - prev.get('body_fat_kg_total', 0)
        
        def format_change(value, suffix='kg', decimals=1):
            if abs(value) < 0.01:
                return f"0.0 {suffix}"
            arrow = "↑" if value > 0 else "↓"
            color = "#198754" if value > 0 else "#dc3545"
            if suffix == '%' or suffix == 'score':
                color = "#198754" if value > 0 else "#dc3545"
            if suffix == 'kg' and 'fat' in locals():
                color = "#dc3545" if value > 0 else "#198754"  # For fat, down is good
            return f"<span style='color:{color};'>{arrow} {abs(value):.{decimals}f} {suffix}</span>"
        
        days_tracked = (pd.to_datetime(latest['date']) - pd.to_datetime(first['date'])).days
        measurements = len(d)
        
        html = f"""
        <div style='padding:20px; font-family:system-ui;'>
            <div style='margin-bottom:25px;'>
                <h5 style='color:#6c757d; font-size:0.9rem; margin-bottom:10px; text-transform:uppercase;'>📅 Tracking Overview</h5>
                <div style='display:grid; grid-template-columns:1fr 1fr; gap:15px;'>
                    <div>
                        <div style='font-size:0.85rem; color:#6c757d;'>Total Days</div>
                        <div style='font-size:1.5rem; font-weight:bold;'>{days_tracked}</div>
                    </div>
                    <div>
                        <div style='font-size:0.85rem; color:#6c757d;'>Measurements</div>
                        <div style='font-size:1.5rem; font-weight:bold;'>{measurements}</div>
                    </div>
                </div>
            </div>
            
            <div style='margin-bottom:25px;'>
                <h5 style='color:#6c757d; font-size:0.9rem; margin-bottom:10px; text-transform:uppercase;'>📈 Overall Progress</h5>
                <div style='background:#f8f9fa; padding:15px; border-radius:8px;'>
                    <div style='margin-bottom:12px;'>
                        <div style='font-size:0.85rem; color:#6c757d;'>💪 Muscle Mass</div>
                        <div style='font-size:1.3rem;'>{format_change(muscle_change)}</div>
                    </div>
                    <div style='margin-bottom:12px;'>
                        <div style='font-size:0.85rem; color:#6c757d;'>🔥 Body Fat</div>
                        <div style='font-size:1.3rem;'>{format_change(-fat_change)}</div>
                    </div>
                    <div style='margin-bottom:12px;'>
                        <div style='font-size:0.85rem; color:#6c757d;'>⚖️ Body Weight</div>
                        <div style='font-size:1.3rem;'>{format_change(weight_change)}</div>
                    </div>
                    <div>
                        <div style='font-size:0.85rem; color:#6c757d;'>📊 InBody Score</div>
                        <div style='font-size:1.3rem;'>{format_change(score_change, 'pts', 0)}</div>
                    </div>
                </div>
            </div>
            
            <div>
                <h5 style='color:#6c757d; font-size:0.9rem; margin-bottom:10px; text-transform:uppercase;'>🎯 Recent Change</h5>
                <div style='background:#e7f3ff; padding:15px; border-radius:8px; border-left:4px solid #0d6efd;'>
                    <div style='font-size:0.85rem; color:#6c757d; margin-bottom:8px;'>Since last measurement:</div>
                    <div style='display:grid; grid-template-columns:1fr 1fr; gap:10px;'>
                        <div>
                            <div style='font-size:0.8rem; color:#6c757d;'>Muscle</div>
                            <div style='font-size:1.1rem;'>{format_change(muscle_recent)}</div>
                        </div>
                        <div>
                            <div style='font-size:0.8rem; color:#6c757d;'>Fat</div>
                            <div style='font-size:1.1rem;'>{format_change(-fat_recent)}</div>
                        </div>
                    </div>
                </div>
            </div>
            
            <div style='margin-top:25px; padding-top:15px; border-top:1px solid #dee2e6;'>
                <div style='font-size:0.75rem; color:#6c757d; text-align:center;'>
                    Latest: {latest['date']} • First: {first['date']}
                </div>
            </div>
        </div>
        """
        return ui.HTML(html)


    @output
    @render_plotly
    def inbody_trend_main():
        d = inbody_df()
        if d.empty:
            return _empty_figure("No InBody data")
        
        fig = go.Figure()
        fig.add_trace(go.Scatter(
            x=d["date"], y=d["skeletal_muscle_kg_total"],
            mode='lines+markers', name='Skeletal Muscle',
            line=dict(color='#198754', width=3), marker=dict(size=8, color='#198754'),
            hovertemplate='Date: %{x}<br>Muscle: %{y:.1f} kg<extra></extra>'
        ))
        fig.add_trace(go.Scatter(
            x=d["date"], y=d["body_fat_kg_total"],
            mode='lines+markers', name='Body Fat',
            line=dict(color='#dc3545', width=3), marker=dict(size=8, color='#dc3545'),
            hovertemplate='Date: %{x}<br>Fat: %{y:.1f} kg<extra></extra>'
        ))
        fig.add_trace(go.Scatter(
            x=d["date"], y=d["body_fat_percent"],
            mode='lines+markers', name='Body Fat %',
            line=dict(color='#fd7e14', width=3, dash='dot'), marker=dict(size=8, color='#fd7e14'),
            yaxis='y2',
            hovertemplate='Date: %{x}<br>Fat: %{y:.1f} %<extra></extra>'
        ))
        
        fig.update_layout(
            title=dict(text="InBody Composition Trends", font=dict(size=18, weight='bold')),
            xaxis_title="Date",
            yaxis_title="Mass (kg)",
            yaxis2=dict(title="Body Fat %", overlaying='y', side='right'),
            hovermode='closest',
            template='plotly_white',
            autosize=True,
            margin=dict(t=40, r=20, b=40, l=50),
            legend=dict(orientation='h', yanchor='bottom', y=1.02, xanchor='center', x=0.5)
        )
        return fig

    @output
    @render_plotly
    def inbody_visceral_fat():
        d = inbody_df()
        if d.empty:
            return _empty_figure("No InBody data")
        
        fig = go.Figure()
        fig.add_trace(go.Scatter(
            x=d["date"], y=d["visceral_fat_level"],
            mode='lines+markers', name='Visceral Fat Level',
            line=dict(color='#dc3545', width=3), marker=dict(size=8, color='#dc3545'),
            fill='tozeroy', fillcolor='rgba(220, 53, 69, 0.15)',
            hovertemplate='Date: %{x}<br>Level: %{y:.0f}<extra></extra>'
        ))
        
        fig.update_layout(
            title=dict(text="Visceral Fat Level", font=dict(size=18, weight='bold')),
            xaxis_title="Date",
            yaxis_title="Level",
            hovermode='closest',
            template='plotly_white',
            autosize=True,
            margin=dict(t=40, r=20, b=40, l=50),
            showlegend=False
        )
        return fig

    @output
    @render_plotly
    def inbody_segmental_muscle():
        d = inbody_df()
        if d.empty:
            return _empty_figure("No InBody data")
        
        latest = d.iloc[-1]
        segments = ['Right Arm', 'Left Arm', 'Trunk', 'Right Leg', 'Left Leg']
        values = [
            latest.get('muscle_right_arm_kg', 0),
            latest.get('muscle_left_arm_kg', 0),
            latest.get('muscle_trunk_kg', 0),
            latest.get('muscle_right_leg_kg', 0),
            latest.get('muscle_left_leg_kg', 0)
        ]
        colors = ['#0d6efd', '#6610f2', '#198754', '#0dcaf0', '#d63384']
        
        fig = go.Figure(data=[go.Bar(
            x=segments, y=values,
            marker_color=colors,
            text=[f"{v:.1f} kg" for v in values],
            textposition='outside',
            hovertemplate='%{x}<br>%{y:.1f} kg<extra></extra>'
        )])
        
        fig.update_layout(
            title=dict(text="Segmental Muscle Mass (Latest)", font=dict(size=18, weight='bold')),
            xaxis_title="Body Segment",
            yaxis_title="Muscle Mass (kg)",
            template='plotly_white',
            autosize=True,
            margin=dict(l=50, r=50, t=50, b=50),
            showlegend=False
        )
        return fig

    @output
    @render_plotly
    def inbody_segmental_fat():
        d = inbody_df()
        if d.empty:
            return _empty_figure("No InBody data")
        
        latest = d.iloc[-1]
        segments = ['Right Arm', 'Left Arm', 'Trunk', 'Right Leg', 'Left Leg']
        values = [
            latest.get('fat_right_arm_kg', 0),
            latest.get('fat_left_arm_kg', 0),
            latest.get('fat_trunk_kg', 0),
            latest.get('fat_right_leg_kg', 0),
            latest.get('fat_left_leg_kg', 0)
        ]
        colors = ['#fd7e14', '#ffc107', '#dc3545', '#e83e8c', '#6f42c1']
        
        fig = go.Figure(data=[go.Bar(
            x=segments, y=values,
            marker_color=colors,
            text=[f"{v:.1f} kg" for v in values],
            textposition='outside',
            hovertemplate='%{x}<br>%{y:.1f} kg<extra></extra>'
        )])
        
        fig.update_layout(
            title=dict(text="Segmental Fat Mass (Latest)", font=dict(size=18, weight='bold')),
            xaxis_title="Body Segment",
            yaxis_title="Fat Mass (kg)",
            template='plotly_white',
            autosize=True,
            margin=dict(l=50, r=50, t=50, b=50),
            showlegend=False
        )
        return fig

    @output
    @render_plotly
    def inbody_asymmetry():
        d = inbody_df()
        if d.empty:
            return _empty_figure("No InBody data")
        
        fig = go.Figure()
        fig.add_trace(go.Scatter(
            x=d["date"], y=d["arm_asym_pct"],
            mode='lines+markers', name='Arm Asymmetry',
            line=dict(color='#0d6efd', width=3), marker=dict(size=8, color='#0d6efd'),
            hovertemplate='Date: %{x}<br>Arm: %{y:.1f}%<extra></extra>'
        ))
        fig.add_trace(go.Scatter(
            x=d["date"], y=d["leg_asym_pct"],
            mode='lines+markers', name='Leg Asymmetry',
            line=dict(color='#198754', width=3), marker=dict(size=8, color='#198754'),
            hovertemplate='Date: %{x}<br>Leg: %{y:.1f}%<extra></extra>'
        ))
        
        # Add ±5% tolerance bands
        fig.add_hline(y=5, line_dash="dash", line_color="rgba(255,0,0,0.3)", annotation_text="±5%")
        fig.add_hline(y=-5, line_dash="dash", line_color="rgba(255,0,0,0.3)")
        fig.add_hline(y=0, line_dash="dot", line_color="rgba(0,0,0,0.2)")
        
        fig.update_layout(
            title=dict(text="Muscle Asymmetry (±5% tolerance)", font=dict(size=18, weight='bold')),
            xaxis_title="Date",
            yaxis_title="Asymmetry (%)",
            hovermode='closest',
            template='plotly_white',
            autosize=True,
            margin=dict(t=40, r=20, b=40, l=50),
            legend=dict(orientation='h', yanchor='bottom', y=1.02, xanchor='center', x=0.5)
        )
        return fig

    @output
    @render_plotly
    def inbody_bmr_trend():
        d = inbody_df()
        if d.empty:
            return _empty_figure("No InBody data")
        
        fig = go.Figure()
        
        # BMR trend
        fig.add_trace(go.Scatter(
            x=d["date"], y=d["bmr_kcal"],
            mode='lines+markers', name='BMR (kcal/day)',
            line=dict(color='#198754', width=3), marker=dict(size=8, color='#198754'),
            fill='tozeroy', fillcolor='rgba(25, 135, 84, 0.15)',
            hovertemplate='Date: %{x}<br>BMR: %{y:.0f} kcal<extra></extra>'
        ))
        
        # InBody Score on secondary axis
        fig.add_trace(go.Scatter(
            x=d["date"], y=d["inbody_score"],
            mode='lines+markers', name='InBody Score',
            line=dict(color='#0d6efd', width=3, dash='dot'), marker=dict(size=8, color='#0d6efd'),
            yaxis='y2',
            hovertemplate='Date: %{x}<br>Score: %{y:.0f}<extra></extra>'
        ))
        
        fig.update_layout(
            title=dict(text="Basal Metabolic Rate & InBody Score", font=dict(size=18, weight='bold')),
            xaxis_title="Date",
            yaxis_title="BMR (kcal/day)",
            yaxis2=dict(title="InBody Score", overlaying='y', side='right', range=[0, 100]),
            hovermode='closest',
            template='plotly_white',
            autosize=True,
            margin=dict(t=40, r=20, b=40, l=50),
            legend=dict(orientation='h', yanchor='bottom', y=1.02, xanchor='center', x=0.5)
        )
        return fig

    @output
    @render_plotly
    def inbody_ratios():
        d = inbody_df()
        if d.empty:
            return _empty_figure("No InBody data")
        
        fig = go.Figure()
        
        # Upper/Lower muscle ratio
        fig.add_trace(go.Scatter(
            x=d["date"], y=d["upper_lower_ratio"],
            mode='lines+markers', name='Upper/Lower Ratio',
            line=dict(color='#0d6efd', width=3), marker=dict(size=8, color='#0d6efd'),
            hovertemplate='Date: %{x}<br>Ratio: %{y:.2f}<extra></extra>'
        ))
        
        # Trunk muscle share percentage
        fig.add_trace(go.Scatter(
            x=d["date"], y=d["trunk_muscle_share_pct"],
            mode='lines+markers', name='Trunk Muscle %',
            line=dict(color='#198754', width=3), marker=dict(size=8, color='#198754'),
            yaxis='y2',
            hovertemplate='Date: %{x}<br>Trunk: %{y:.1f}%<extra></extra>'
        ))
        
        # Trunk to limb fat ratio
        fig.add_trace(go.Scatter(
            x=d["date"], y=d["trunk_to_limb_fat_ratio"],
            mode='lines+markers', name='Trunk/Limb Fat',
            line=dict(color='#dc3545', width=3, dash='dot'), marker=dict(size=8, color='#dc3545'),
            hovertemplate='Date: %{x}<br>Fat Ratio: %{y:.2f}<extra></extra>'
        ))
        
        fig.update_layout(
            title=dict(text="Body Composition Ratios", font=dict(size=18, weight='bold')),
            xaxis_title="Date",
            yaxis_title="Ratio",
            yaxis2=dict(title="Trunk Muscle %", overlaying='y', side='right'),
            hovermode='closest',
            template='plotly_white',
            autosize=True,
            margin=dict(l=50, r=50, t=50, b=50),
            legend=dict(orientation='h', yanchor='bottom', y=1.02, xanchor='center', x=0.5)
        )
        return fig

    @output
    @render_plotly
    def inbody_3d_phase():
        """3D scatter showing body composition phase space (Muscle vs Fat vs Weight)"""
        d = inbody_df()
        if d.empty or len(d) < 3:
            return _empty_figure("Need at least 3 measurements for 3D visualization")
        
        # Create color scale based on date progression
        dates_numeric = pd.to_datetime(d['date']).astype(int) / 10**9
        
        fig = go.Figure(data=[go.Scatter3d(
            x=d['skeletal_muscle_kg_total'],
            y=d['body_fat_kg_total'],
            z=d['weight_kg'],
            mode='markers+lines',
            marker=dict(
                size=8,
                color=dates_numeric,
                colorscale='Viridis',
                showscale=True,
                colorbar=dict(title="Time →", ticktext=['Start', 'Latest'], tickvals=[dates_numeric.min(), dates_numeric.max()]),
                line=dict(color='white', width=1)
            ),
            line=dict(color='rgba(100,100,100,0.3)', width=2),
            text=[f"Date: {row['date']}<br>Muscle: {row['skeletal_muscle_kg_total']:.1f}kg<br>Fat: {row['body_fat_kg_total']:.1f}kg<br>Weight: {row['weight_kg']:.1f}kg" 
                  for _, row in d.iterrows()],
            hovertemplate='%{text}<extra></extra>'
        )])
        
        fig.update_layout(
            title=dict(text="3D Body Composition Journey", font=dict(size=16, weight='bold')),
            scene=dict(
                xaxis=dict(title='Muscle Mass (kg)', backgroundcolor="rgb(230, 230,230)", gridcolor="white"),
                yaxis=dict(title='Body Fat (kg)', backgroundcolor="rgb(230, 230,230)", gridcolor="white"),
                zaxis=dict(title='Body Weight (kg)', backgroundcolor="rgb(230, 230,230)", gridcolor="white"),
            ),
            template='plotly_white',
            autosize=True,
            margin=dict(l=0, r=0, t=40, b=0)
        )
        return fig

    @output
    @render_plotly
    def inbody_correlation_matrix():
        """Correlation heatmap of all InBody metrics"""
        d = inbody_df()
        if d.empty or len(d) < 3:
            return _empty_figure("Need at least 3 measurements for correlation analysis")
        
        # Select numeric columns for correlation
        numeric_cols = [
            'skeletal_muscle_kg_total', 'body_fat_kg_total', 'body_fat_percent',
            'visceral_fat_level', 'bmr_kcal', 'inbody_score', 'weight_kg',
            'upper_lower_ratio', 'trunk_muscle_share_pct', 'trunk_to_limb_fat_ratio'
        ]
        available_cols = [c for c in numeric_cols if c in d.columns]
        
        if len(available_cols) < 3:
            return _empty_figure("Insufficient metrics for correlation")
        
        corr_matrix = d[available_cols].corr()
        
        # Shorten labels
        labels = {
            'skeletal_muscle_kg_total': 'Muscle',
            'body_fat_kg_total': 'Fat Mass',
            'body_fat_percent': 'Fat %',
            'visceral_fat_level': 'Visceral',
            'bmr_kcal': 'BMR',
            'inbody_score': 'IB Score',
            'weight_kg': 'Weight',
            'upper_lower_ratio': 'Up/Low',
            'trunk_muscle_share_pct': 'Trunk %',
            'trunk_to_limb_fat_ratio': 'Trunk/Limb'
        }
        display_labels = [labels.get(c, c) for c in available_cols]
        
        fig = go.Figure(data=go.Heatmap(
            z=corr_matrix.values,
            x=display_labels,
            y=display_labels,
            colorscale='RdBu',
            zmid=0,
            text=np.round(corr_matrix.values, 2),
            texttemplate='%{text}',
            textfont={"size": 10},
            colorbar=dict(title="Correlation")
        ))
        
        fig.update_layout(
            title=dict(text="Metrics Correlation Matrix", font=dict(size=16, weight='bold')),
            xaxis=dict(side='bottom'),
            yaxis=dict(autorange='reversed'),
            template='plotly_white',
            autosize=True,
            margin=dict(l=80, r=50, t=60, b=80)
        )
        return fig

    @output
    @render_plotly
    def inbody_velocity_chart():
        """Rate of change (velocity) for key metrics"""
        d = inbody_df()
        if d.empty or len(d) < 2:
            return _empty_figure("Need at least 2 measurements")
        
        # Calculate daily rates of change
        d = d.sort_values('date').copy()
        d['days_since_first'] = (pd.to_datetime(d['date']) - pd.to_datetime(d['date'].iloc[0])).dt.days
        
        fig = go.Figure()
        
        if 'delta_skeletal_muscle_kg' in d.columns:
            d['muscle_velocity'] = d['delta_skeletal_muscle_kg'] / d['days_since_first'].diff().fillna(1)
            fig.add_trace(go.Scatter(
                x=d['date'], y=d['muscle_velocity'],
                mode='lines+markers', name='Muscle Δ/day',
                line=dict(color='#198754', width=2), marker=dict(size=6),
                hovertemplate='%{x}<br>%{y:+.3f} kg/day<extra></extra>'
            ))
        
        if 'delta_body_fat_kg' in d.columns:
            d['fat_velocity'] = d['delta_body_fat_kg'] / d['days_since_first'].diff().fillna(1)
            fig.add_trace(go.Scatter(
                x=d['date'], y=d['fat_velocity'],
                mode='lines+markers', name='Fat Δ/day',
                line=dict(color='#dc3545', width=2), marker=dict(size=6),
                hovertemplate='%{x}<br>%{y:+.3f} kg/day<extra></extra>'
            ))
        
        fig.add_hline(y=0, line_dash="dot", line_color="rgba(0,0,0,0.3)")
        
        fig.update_layout(
            title=dict(text="Daily Rate of Change", font=dict(size=16, weight='bold')),
            xaxis_title="Date",
            yaxis_title="Change Rate (kg/day)",
            hovermode='x unified',
            template='plotly_white',
            autosize=True,
            margin=dict(l=50, r=20, t=50, b=40),
            legend=dict(orientation='h', yanchor='bottom', y=1.02, xanchor='center', x=0.5)
        )
        return fig

    @output
    @render_plotly
    def inbody_recomp_efficiency():
        """Body recomposition efficiency: muscle gained per fat lost"""
        d = inbody_df()
        if d.empty or len(d) < 2:
            return _empty_figure("Need at least 2 measurements")
        
        d = d.sort_values('date').copy()
        
        # Calculate cumulative changes from baseline
        baseline = d.iloc[0]
        d['cumulative_muscle_gain'] = d['skeletal_muscle_kg_total'] - baseline['skeletal_muscle_kg_total']
        d['cumulative_fat_loss'] = -(d['body_fat_kg_total'] - baseline['body_fat_kg_total'])
        
        # Recomp ratio (positive means good: gaining muscle AND losing fat)
        d['recomp_ratio'] = np.where(
            d['cumulative_fat_loss'] != 0,
            d['cumulative_muscle_gain'] / d['cumulative_fat_loss'].abs(),
            0
        )
        
        fig = go.Figure()
        
        # Scatter plot with color gradient
        fig.add_trace(go.Scatter(
            x=d['cumulative_fat_loss'],
            y=d['cumulative_muscle_gain'],
            mode='markers+lines',
            marker=dict(
                size=12,
                color=d['recomp_ratio'],
                colorscale='RdYlGn',
                showscale=True,
                colorbar=dict(title="Efficiency"),
                line=dict(color='white', width=1)
            ),
            line=dict(color='rgba(100,100,100,0.3)', width=2),
            text=[f"Date: {row['date']}<br>Fat Loss: {row['cumulative_fat_loss']:+.1f}kg<br>Muscle Gain: {row['cumulative_muscle_gain']:+.1f}kg<br>Ratio: {row['recomp_ratio']:.2f}" 
                  for _, row in d.iterrows()],
            hovertemplate='%{text}<extra></extra>'
        ))
        
        # Add ideal 1:1 line
        max_val = max(d['cumulative_fat_loss'].max(), d['cumulative_muscle_gain'].max())
        fig.add_trace(go.Scatter(
            x=[0, max_val], y=[0, max_val],
            mode='lines',
            line=dict(color='rgba(0,0,0,0.2)', width=1, dash='dash'),
            name='1:1 ratio',
            hoverinfo='skip'
        ))
        
        fig.update_layout(
            title=dict(text="Recomposition Efficiency Map", font=dict(size=16, weight='bold')),
            xaxis=dict(title="Cumulative Fat Loss (kg)", zeroline=True),
            yaxis=dict(title="Cumulative Muscle Gain (kg)", zeroline=True),
            template='plotly_white',
            autosize=True,
            margin=dict(l=50, r=20, t=50, b=40),
            showlegend=True
        )
        return fig

    @output
    @render_plotly
    def inbody_radar_chart():
        """Radar chart comparing muscle vs fat distribution across body segments"""
        d = inbody_df()
        if d.empty:
            return _empty_figure("No InBody data")
        
        latest = d.iloc[-1]
        
        categories = ['Right Arm', 'Left Arm', 'Trunk', 'Right Leg', 'Left Leg']
        
        muscle_values = [
            latest.get('muscle_right_arm_kg', 0),
            latest.get('muscle_left_arm_kg', 0),
            latest.get('muscle_trunk_kg', 0),
            latest.get('muscle_right_leg_kg', 0),
            latest.get('muscle_left_leg_kg', 0)
        ]
        
        fat_values = [
            latest.get('fat_right_arm_kg', 0),
            latest.get('fat_left_arm_kg', 0),
            latest.get('fat_trunk_kg', 0),
            latest.get('fat_right_leg_kg', 0),
            latest.get('fat_left_leg_kg', 0)
        ]
        
        # Normalize to percentages of total
        total_muscle = sum(muscle_values)
        total_fat = sum(fat_values)
        
        muscle_pct = [v/total_muscle*100 if total_muscle > 0 else 0 for v in muscle_values]
        fat_pct = [v/total_fat*100 if total_fat > 0 else 0 for v in fat_values]
        
        fig = go.Figure()
        
        fig.add_trace(go.Scatterpolar(
            r=muscle_pct + [muscle_pct[0]],
            theta=categories + [categories[0]],
            fill='toself',
            name='Muscle %',
            line=dict(color='#198754', width=2),
            fillcolor='rgba(25, 135, 84, 0.2)'
        ))
        
        fig.add_trace(go.Scatterpolar(
            r=fat_pct + [fat_pct[0]],
            theta=categories + [categories[0]],
            fill='toself',
            name='Fat %',
            line=dict(color='#dc3545', width=2),
            fillcolor='rgba(220, 53, 69, 0.2)'
        ))
        
        fig.update_layout(
            polar=dict(
                radialaxis=dict(visible=True, range=[0, max(max(muscle_pct), max(fat_pct)) * 1.1])
            ),
            title=dict(text="Segmental Distribution Radar", font=dict(size=16, weight='bold')),
            template='plotly_white',
            autosize=True,
            margin=dict(l=80, r=80, t=60, b=40),
            showlegend=True,
            legend=dict(orientation='h', yanchor='bottom', y=-0.15, xanchor='center', x=0.5)
        )
        return fig

    @output
    @render.ui
    def ib_advanced_metrics():
        """Display advanced calculated metrics"""
        d = inbody_df()
        if d.empty:
            return ui.HTML("<div style='padding:20px; text-align:center; color:#6c757d;'>No data available</div>")
        
        latest = d.iloc[-1]
        
        # Calculate advanced metrics
        ffmi = latest.get('skeletal_muscle_kg_total', 0) / ((latest.get('weight_kg', 70) / 100) ** 2) if latest.get('weight_kg', 0) > 0 else 0
        
        muscle_to_fat_ratio = latest.get('skeletal_muscle_kg_total', 0) / latest.get('body_fat_kg_total', 1) if latest.get('body_fat_kg_total', 0) > 0 else 0
        
        # Estimated TDEE (BMR * activity factor, assuming moderate)
        tdee = latest.get('bmr_kcal', 0) * 1.55
        
        # Muscle quality score (muscle mass relative to body weight)
        muscle_quality = (latest.get('skeletal_muscle_kg_total', 0) / latest.get('weight_kg', 1) * 100) if latest.get('weight_kg', 0) > 0 else 0
        
        html = f"""
        <div style='padding:20px; font-family:system-ui;'>
            <div style='display:grid; grid-template-columns:1fr 1fr; gap:20px;'>
                <div style='background:#e7f3ff; padding:15px; border-radius:8px; border-left:4px solid #0d6efd;'>
                    <div style='font-size:0.75rem; color:#6c757d; text-transform:uppercase; margin-bottom:5px;'>Fat-Free Mass Index</div>
                    <div style='font-size:1.8rem; font-weight:bold; color:#0d6efd;'>{ffmi:.1f}</div>
                    <div style='font-size:0.7rem; color:#6c757d; margin-top:5px;'>kg/m² (Normal: 16-20)</div>
                </div>
                
                <div style='background:#d1f5e8; padding:15px; border-radius:8px; border-left:4px solid #198754;'>
                    <div style='font-size:0.75rem; color:#6c757d; text-transform:uppercase; margin-bottom:5px;'>Muscle/Fat Ratio</div>
                    <div style='font-size:1.8rem; font-weight:bold; color:#198754;'>{muscle_to_fat_ratio:.2f}</div>
                    <div style='font-size:0.7rem; color:#6c757d; margin-top:5px;'>Higher is better</div>
                </div>
                
                <div style='background:#fff3cd; padding:15px; border-radius:8px; border-left:4px solid #ffc107;'>
                    <div style='font-size:0.75rem; color:#6c757d; text-transform:uppercase; margin-bottom:5px;'>Est. TDEE</div>
                    <div style='font-size:1.8rem; font-weight:bold; color:#856404;'>{tdee:.0f}</div>
                    <div style='font-size:0.7rem; color:#6c757d; margin-top:5px;'>kcal/day (moderate activity)</div>
                </div>
                
                <div style='background:#e2e3e5; padding:15px; border-radius:8px; border-left:4px solid #6c757d;'>
                    <div style='font-size:0.75rem; color:#6c757d; text-transform:uppercase; margin-bottom:5px;'>Muscle Quality</div>
                    <div style='font-size:1.8rem; font-weight:bold; color:#495057;'>{muscle_quality:.1f}%</div>
                    <div style='font-size:0.7rem; color:#6c757d; margin-top:5px;'>Muscle as % of weight</div>
                </div>
            </div>
            
            <div style='margin-top:20px; padding:15px; background:#f8f9fa; border-radius:8px;'>
                <h6 style='margin:0 0 10px 0; font-size:0.9rem; color:#495057;'>📊 Key Insights</h6>
                <ul style='margin:0; padding-left:20px; font-size:0.85rem; color:#6c757d;'>
                    <li>Upper/Lower Ratio: <strong>{latest.get('upper_lower_ratio', 0):.2f}</strong> (ideal: 0.65-0.75)</li>
                    <li>Trunk Muscle Share: <strong>{latest.get('trunk_muscle_share_pct', 0):.1f}%</strong></li>
                    <li>Asymmetry Status: <strong>{"✓ Balanced" if abs(latest.get('arm_asym_pct', 0)) < 5 and abs(latest.get('leg_asym_pct', 0)) < 5 else "⚠ Check distribution"}</strong></li>
                    <li>Visceral Fat: <strong>{"✓ Healthy" if latest.get('visceral_fat_level', 0) < 10 else "⚠ Monitor closely"}</strong> (Level {latest.get('visceral_fat_level', 0):.0f})</li>
                </ul>
            </div>
        </div>
        """
        return ui.HTML(html)

    @output
    @render_plotly
    def inbody_percentile_chart():
        """Show personal progress percentiles (comparing current to personal history)"""
        d = inbody_df()
        if d.empty or len(d) < 3:
            return _empty_figure("Need at least 3 measurements for percentile analysis")
        
        latest = d.iloc[-1]
        
        metrics = {
            'InBody Score': ('inbody_score', True),
            'Muscle Mass': ('skeletal_muscle_kg_total', True),
            'Body Fat %': ('body_fat_percent', False),
            'Visceral Fat': ('visceral_fat_level', False),
            'BMR': ('bmr_kcal', True)
        }
        
        labels = []
        percentiles = []
        colors = []
        
        for label, (col, higher_better) in metrics.items():
            if col in d.columns:
                value = latest.get(col, 0)
                all_values = d[col].dropna()
                
                if len(all_values) > 1:
                    percentile = (all_values < value).sum() / len(all_values) * 100 if higher_better else (all_values > value).sum() / len(all_values) * 100
                    
                    labels.append(label)
                    percentiles.append(percentile)
                    
                    # Color based on performance
                    if percentile >= 75:
                        colors.append('#198754')  # green
                    elif percentile >= 50:
                        colors.append('#ffc107')  # yellow
                    elif percentile >= 25:
                        colors.append('#fd7e14')  # orange
                    else:
                        colors.append('#dc3545')  # red
        
        fig = go.Figure(data=[go.Bar(
            x=percentiles,
            y=labels,
            orientation='h',
            marker=dict(color=colors),
            text=[f"{p:.0f}%" for p in percentiles],
            textposition='outside',
            hovertemplate='%{y}: %{x:.1f}th percentile<extra></extra>'
        )])
        
        fig.add_vline(x=50, line_dash="dash", line_color="rgba(0,0,0,0.3)", annotation_text="Median")
        fig.add_vline(x=75, line_dash="dot", line_color="rgba(25,135,84,0.3)", annotation_text="75th")
        
        fig.update_layout(
            title=dict(text="Current Performance vs Personal History", font=dict(size=16, weight='bold')),
            xaxis=dict(title="Percentile Rank (%)", range=[0, 105]),
            yaxis=dict(title=""),
            template='plotly_white',
            autosize=True,
            margin=dict(l=120, r=20, t=50, b=40),
            showlegend=False
        )
        return fig

    @output
    @render.data_frame
    def tbl_inbody():
        d = inbody_df()
        if d.empty:
            from .utils import REQUIRED_TABS
            return pd.DataFrame(columns=REQUIRED_TABS["InBody"])
        
        # Sort descending and pick display columns
        d = d.sort_values("date", ascending=False).head(20).copy()
        
        cols = [
            "date", "inbody_score", "weight_kg", 
            "skeletal_muscle_kg_total", "body_fat_kg_total", "body_fat_percent",
            "visceral_fat_level", "bmr_kcal",
            "arm_asym_pct", "leg_asym_pct", "trunk_muscle_share_pct",
            "upper_lower_ratio", "trunk_to_limb_fat_ratio",
            "delta_skeletal_muscle_kg", "delta_body_fat_kg", "delta_weight_kg",
            "notes"
        ]
        available_cols = [c for c in cols if c in d.columns]
        return d[available_cols]

    @output
    @render.data_frame
    def tbl_lifts():
        d = lifts_df()
        if d.empty:
            from .utils import REQUIRED_TABS
            return pd.DataFrame(columns=REQUIRED_TABS["Lifts"]+["1rm","tm"])  # type: ignore
        
        # Sort and compute deltas
        d = d.sort_values("date", ascending=False).head(20).copy()
        
        # Compute change from previous entry for same exercise
        d['Δ 1RM'] = ''
        d['Δ Weight'] = ''
        for ex in d['exercise'].unique():
            mask = d['exercise'] == ex
            ex_data = d[mask].sort_values('date', ascending=True)
            if len(ex_data) > 1:
                rm_diff = ex_data['1rm'].diff()
                weight_diff = ex_data['weight_kg'].diff()
                for idx, (rm_d, w_d) in zip(ex_data.index, zip(rm_diff, weight_diff)):
                    if pd.notna(rm_d) and rm_d != 0:
                        arrow = '↑' if rm_d > 0 else '↓'
                        d.at[idx, 'Δ 1RM'] = f"{arrow} {abs(rm_d):.1f}"
                    if pd.notna(w_d) and w_d != 0:
                        arrow = '↑' if w_d > 0 else '↓'
                        d.at[idx, 'Δ Weight'] = f"{arrow} {abs(w_d):.1f}"
        
        cols = [c for c in ["date","exercise","weight_kg","Δ Weight","reps","1rm","Δ 1RM","tm","notes"] if c in d.columns or c in ['Δ 1RM', 'Δ Weight']]
        return d[cols]

    @output
    @render.data_frame
    def tbl_bw():
        d = bw_df()
        if d.empty:
            from .utils import REQUIRED_TABS
            return pd.DataFrame(columns=REQUIRED_TABS["Bodyweight"])  # type: ignore
        
        # Sort and compute deltas
        d = d.sort_values("date", ascending=False).head(20).copy()
        d_sorted = d.sort_values("date", ascending=True)
        
        # Compute change from previous entry
        d['Δ Weight'] = ''
        weight_diff = d_sorted['weight_kg'].diff()
        for idx, w_d in zip(d_sorted.index, weight_diff):
            if pd.notna(w_d) and w_d != 0:
                arrow = '↑' if w_d > 0 else '↓'
                d.at[idx, 'Δ Weight'] = f"{arrow} {abs(w_d):.1f}"
        
        cols = [c for c in ["date","weight_kg","Δ Weight","time","notes"] if c in d.columns or c == 'Δ Weight']
        return d[cols]

    @output
    @render.data_frame
    def tbl_meas():
        d = meas_df()
        if d.empty:
            from .utils import REQUIRED_TABS
            return pd.DataFrame(columns=REQUIRED_TABS["Measurements"])  # type: ignore
        
        # Sort and compute deltas
        d = d.sort_values("date", ascending=False).head(20).copy()
        d_sorted = d.sort_values("date", ascending=True)
        
        # Compute changes for each measurement
        meas_cols = ["neck_cm","shoulder_cm","chest_cm","waist_cm","biceps_cm","thigh_cm","calf_cm"]
        for col in meas_cols:
            if col in d.columns:
                delta_col = f"Δ {col.replace('_cm','').capitalize()}"
                d[delta_col] = ''
                diff = d_sorted[col].diff()
                for idx, val_d in zip(d_sorted.index, diff):
                    if pd.notna(val_d) and val_d != 0:
                        arrow = '↑' if val_d > 0 else '↓'
                        d.at[idx, delta_col] = f"{arrow} {abs(val_d):.1f}"
        
        # Build column list with deltas interspersed
        result_cols = ["date", "weight_kg"]
        for col in meas_cols:
            if col in d.columns:
                result_cols.append(col)
                delta_col = f"Δ {col.replace('_cm','').capitalize()}"
                if delta_col in d.columns:
                    result_cols.append(delta_col)
        
        return d[result_cols]

