from __future__ import annotations

from datetime import date, datetime
from typing import Any, cast

from shiny import ui
from shinywidgets import output_widget

from .utils import BIG3

_NOW = datetime.now()
_HOUR_CHOICES = [f"{i:02d}" for i in range(24)]
_MIN_CHOICES = [f"{i:02d}" for i in range(60)]

app_ui = ui.page_navbar(
    ui.nav_panel(
        "📊 Dashboard",
        ui.tags.style(
            ".vb-center{ text-align:center; }\n"
            ".vb-center .value-box-title, .vb-center .value-box-value{ text-align:center; width:100%; }\n"
            ".vb-center .value-box-grid{ justify-content:center; }\n"
        ),
        # Summary stats row
        ui.layout_columns(
            ui.value_box(
                "Latest Squat 1RM",
                ui.output_ui("stat_squat"),
                showcase=ui.span("🏋️", style="font-size: 3rem;"),
                theme="primary",
                class_="vb-center"
            ),
            ui.value_box(
                "Latest Bench 1RM",
                ui.output_ui("stat_bench"),
                showcase=ui.span("💪", style="font-size: 3rem;"),
                theme="danger",
                class_="vb-center"
            ),
            ui.value_box(
                "Latest Deadlift 1RM",
                ui.output_ui("stat_deadlift"),
                showcase=ui.span("🔥", style="font-size: 3rem;"),
                theme="success",
                class_="vb-center"
            ),
            ui.value_box(
                "Current Bodyweight",
                ui.output_ui("stat_bodyweight"),
                showcase=ui.span("⚖️", style="font-size: 3rem;"),
                theme="info",
                class_="vb-center"
            ),
            col_widths=cast(Any, {"lg": [3, 3, 3, 3]}),
        ),
        ui.layout_columns(
            ui.card(
                ui.card_header(
                    ui.h4("💪 Strength Progress", class_="mb-0"),
                    class_="bg-primary text-white"
                ),
                ui.layout_sidebar(
                    ui.sidebar(
                        ui.input_selectize(
                            "exercise_filter",
                            "Exercise Selection",
                            BIG3,
                            selected=BIG3,
                            multiple=True,
                            width="100%"
                        ),
                        ui.input_checkbox("rm_mode_tm", "Training Max (90%)", False),
                        width="220px",
                        bg="#f8f9fa"
                    ),
                    output_widget("plot_1rm"),
                ),
            ),
            ui.card(
                ui.card_header(
                    ui.h4("⚖️ Bodyweight Tracking", class_="mb-0"),
                    class_="bg-success text-white"
                ),
                output_widget("plot_bw"),
            ),
            col_widths=cast(Any, {"lg": [7, 5]}),
        ),
        ui.layout_columns(
            ui.card(
                ui.card_header(
                    ui.h4("📏 Body Measurements", class_="mb-0"),
                    class_="bg-info text-white"
                ),
                output_widget("plot_meas"),
            ),
            ui.card(
                ui.card_header(
                    ui.h4("📋 Recent Entries", class_="mb-0"),
                    class_="bg-secondary text-white"
                ),
                ui.navset_pill(
                    ui.nav_panel("Lifts", ui.output_data_frame("tbl_lifts")),
                    ui.nav_panel("Bodyweight", ui.output_data_frame("tbl_bw")),
                    ui.nav_panel("Measurements", ui.output_data_frame("tbl_meas")),
                ),
            ),
            col_widths=cast(Any, {"lg": [7, 5]}),
        ),
    ),
    ui.nav_panel(
        "🔬 InBody",
        ui.tags.style(
            ".vb-center{ text-align:center; }\n"
            ".vb-center .value-box-title, .vb-center .value-box-value{ text-align:center; width:100%; }\n"
            ".vb-center .value-box-grid{ justify-content:center; }\n"
            ".ib-h .shiny-widget-output, .ib-h .js-plotly-plot { height: 100% !important; }\n"
        ),
        # Summary Statistics Row
        ui.layout_columns(
            ui.value_box(
                "InBody Score",
                ui.output_ui("ib_stat_score"),
                showcase=ui.span("📈", style="font-size: 3rem;"),
                theme="info",
                class_="vb-center"
            ),
            ui.value_box(
                "Skeletal Muscle",
                ui.output_ui("ib_stat_muscle"),
                showcase=ui.span("💪", style="font-size: 3rem;"),
                theme="success",
                class_="vb-center"
            ),
            ui.value_box(
                "Body Fat (kg)",
                ui.output_ui("ib_stat_fatkg"),
                showcase=ui.span("🔥", style="font-size: 3rem;"),
                theme="danger",
                class_="vb-center"
            ),
            ui.value_box(
                "Body Fat %",
                ui.output_ui("ib_stat_fatpct"),
                showcase=ui.span("⚖️", style="font-size: 3rem;"),
                theme="secondary",
                class_="vb-center"
            ),
            ui.value_box(
                "Visceral Fat",
                ui.output_ui("ib_stat_visceral"),
                showcase=ui.span("🫀", style="font-size: 3rem;"),
                theme="warning",
                class_="vb-center"
            ),
            ui.value_box(
                "BMR (kcal/day)",
                ui.output_ui("ib_stat_bmr"),
                showcase=ui.span("⚡", style="font-size: 3rem;"),
                theme="primary",
                class_="vb-center"
            ),
            col_widths=cast(Any, {"lg": [2, 2, 2, 2, 2, 2]}),
        ),
        
        # Main Composition Trends + Stats Panel
        ui.layout_columns(
            ui.card(
                ui.card_header(ui.h4("📊 Composition Trends Over Time", class_="mb-0"), class_="bg-info text-white"),
                ui.div(output_widget("inbody_trend_main", height="500px", fill=False), style="height:500px;", class_="ib-h"),
                fill=False,
            ),
            ui.card(
                ui.card_header(ui.h4("📈 Progress Statistics", class_="mb-0"), class_="bg-secondary text-white"),
                ui.output_ui("ib_progress_stats"),
                fill=False,
            ),
            col_widths=cast(Any, {"lg": [8, 4]}),
        ),
        
        # Segmental Analysis Row
        ui.layout_columns(
            ui.card(
                ui.card_header(ui.h4("💪 Segmental Muscle Distribution", class_="mb-0"), class_="bg-success text-white"),
                ui.div(output_widget("inbody_segmental_muscle", height="450px", fill=False), style="height:450px;", class_="ib-h"),
                fill=False,
            ),
            ui.card(
                ui.card_header(ui.h4("🔥 Segmental Fat Distribution", class_="mb-0"), class_="bg-danger text-white"),
                ui.div(output_widget("inbody_segmental_fat", height="450px", fill=False), style="height:450px;", class_="ib-h"),
                fill=False,
            ),
            col_widths=cast(Any, {"lg": [6, 6]}),
        ),
        
        # BMR & Score + Body Composition Ratios
        ui.layout_columns(
            ui.card(
                ui.card_header(ui.h4("⚡ Basal Metabolic Rate & InBody Score", class_="mb-0"), class_="bg-success text-white"),
                ui.div(output_widget("inbody_bmr_trend", height="480px", fill=False), style="height:480px;", class_="ib-h"),
                fill=False,
            ),
            ui.card(
                ui.card_header(ui.h4("📐 Body Composition Ratios", class_="mb-0"), class_="bg-warning text-dark"),
                ui.div(output_widget("inbody_ratios", height="480px", fill=False), style="height:480px;", class_="ib-h"),
                fill=False,
            ),
            col_widths=cast(Any, {"lg": [6, 6]}),
        ),
        
        # Health Metrics Row
        ui.layout_columns(
            ui.card(
                ui.card_header(ui.h4("🫀 Visceral Fat Level Tracking", class_="mb-0"), class_="bg-danger text-white"),
                ui.div(output_widget("inbody_visceral_fat", height="450px", fill=False), style="height:450px;", class_="ib-h"),
                fill=False,
            ),
            ui.card(
                ui.card_header(ui.h4("⚖️ Muscle Asymmetry Analysis", class_="mb-0"), class_="bg-primary text-white"),
                ui.div(output_widget("inbody_asymmetry", height="450px", fill=False), style="height:450px;", class_="ib-h"),
                fill=False,
            ),
            col_widths=cast(Any, {"lg": [6, 6]}),
        ),
        
        # Advanced Analytics Row - 3D Body Composition Phase + Correlation Matrix
        ui.layout_columns(
            ui.card(
                ui.card_header(ui.h4("🧬 3D Body Composition Phase Space", class_="mb-0"), class_="bg-info text-white"),
                ui.div(output_widget("inbody_3d_phase", height="520px", fill=False), style="height:520px;", class_="ib-h"),
                fill=False,
            ),
            ui.card(
                ui.card_header(ui.h4("🔗 Metrics Correlation Heatmap", class_="mb-0"), class_="bg-secondary text-white"),
                ui.div(output_widget("inbody_correlation_matrix", height="520px", fill=False), style="height:520px;", class_="ib-h"),
                fill=False,
            ),
            col_widths=cast(Any, {"lg": [6, 6]}),
        ),
        
        # Trends & Predictions Row
        ui.layout_columns(
            ui.card(
                ui.card_header(ui.h4("📉 Rate of Change Analysis", class_="mb-0"), class_="bg-warning text-dark"),
                ui.div(output_widget("inbody_velocity_chart", height="450px", fill=False), style="height:450px;", class_="ib-h"),
                fill=False,
            ),
            ui.card(
                ui.card_header(ui.h4("🎯 Body Recomposition Efficiency", class_="mb-0"), class_="bg-success text-white"),
                ui.div(output_widget("inbody_recomp_efficiency", height="450px", fill=False), style="height:450px;", class_="ib-h"),
                fill=False,
            ),
            col_widths=cast(Any, {"lg": [6, 6]}),
        ),
        
        # Advanced Segmental Analysis
        ui.layout_columns(
            ui.card(
                ui.card_header(ui.h4("🗺️ Muscle-Fat Distribution Radar", class_="mb-0"), class_="bg-primary text-white"),
                ui.div(output_widget("inbody_radar_chart", height="500px", fill=False), style="height:500px;", class_="ib-h"),
                fill=False,
            ),
            ui.card(
                ui.card_header(ui.h4("📊 Advanced Metrics Dashboard", class_="mb-0"), class_="bg-dark text-white"),
                ui.output_ui("ib_advanced_metrics"),
                fill=False,
            ),
            col_widths=cast(Any, {"lg": [7, 5]}),
        ),
        
        # Percentile Rankings & Benchmarks
        ui.layout_columns(
            ui.card(
                ui.card_header(ui.h4("🏆 Personal Progress Percentiles", class_="mb-0"), class_="bg-info text-white"),
                ui.div(output_widget("inbody_percentile_chart", height="400px", fill=False), style="height:400px;", class_="ib-h"),
                fill=False,
            ),
            col_widths=cast(Any, {"lg": [12]}),
        ),
        
        # Data Table at Bottom
        ui.layout_columns(
            ui.card(
                ui.card_header(ui.h4("📋 Complete InBody Records", class_="mb-0"), class_="bg-secondary text-white"),
                ui.output_data_frame("tbl_inbody"),
                fill=False,
            ),
            col_widths=cast(Any, {"lg": [12]}),
        ),
        
    ),
    ui.nav_panel(
        "Add / Edit",
        ui.layout_columns(
            ui.card(
                ui.card_header(
                    ui.h4("💪 Lifts", class_="mb-0"),
                    class_="bg-primary text-white"
                ),
                ui.navset_pill(
                    ui.nav_panel(
                        "Add New",
                        ui.layout_column_wrap(
                            ui.input_date("lift_date", "Date", value=date.today()),
                            ui.input_select("lift_ex", "Exercise", BIG3),
                            width=1/2,
                        ),
                        ui.layout_column_wrap(
                            ui.input_numeric("lift_weight", "Weight (kg)", 50, min=0, step=2.5),
                            ui.input_numeric("lift_reps", "Reps", 5, min=1, step=1),
                            width=1/2,
                        ),
                        ui.input_text_area("lift_notes", "Notes", "", rows=2),
                        ui.input_action_button("btn_add_lift", "Add Lift", class_="btn-primary w-100"),
                    ),
                    ui.nav_panel(
                        "Edit",
                        ui.input_selectize("lift_pick", "Select Entry", choices=[], width="100%"),
                        ui.output_ui("lift_loaded_info"),
                        ui.layout_column_wrap(
                            ui.input_action_button("btn_load_lift", "Load", class_="btn-info"),
                            ui.input_action_button("btn_save_lift", "Save", class_="btn-success"),
                            ui.input_action_button("btn_del_lift", "Delete", class_="btn-danger"),
                            width=1/3,
                        ),
                    ),
                ),
            ),
            ui.card(
                ui.card_header(
                    ui.h4("⚖️ Bodyweight", class_="mb-0"),
                    class_="bg-success text-white"
                ),
                ui.navset_pill(
                    ui.nav_panel(
                        "Add New",
                        ui.input_date("bw_date", "Date", value=date.today()),
                        ui.input_checkbox("bw_use_now", "Use current time", True),
                        ui.layout_column_wrap(
                            ui.input_select("bw_hour", "Hour", _HOUR_CHOICES, selected=f"{_NOW.hour:02d}"),
                            ui.input_select("bw_min", "Min", _MIN_CHOICES, selected=f"{_NOW.minute:02d}"),
                            width=1/2,
                        ),
                        ui.input_numeric("bw_weight", "Weight (kg)", 80, min=0, step=0.1),
                        ui.input_text_area("bw_notes", "Notes", "", rows=2),
                        ui.input_action_button("btn_add_bw", "Add Entry", class_="btn-success w-100"),
                    ),
                    ui.nav_panel(
                        "Edit",
                        ui.input_selectize("bw_pick", "Select Entry", choices=[], width="100%"),
                        ui.output_ui("bw_loaded_info"),
                        ui.layout_column_wrap(
                            ui.input_action_button("btn_load_bw", "Load", class_="btn-info"),
                            ui.input_action_button("btn_save_bw", "Save", class_="btn-success"),
                            ui.input_action_button("btn_del_bw", "Delete", class_="btn-danger"),
                            width=1/3,
                        ),
                    ),
                ),
            ),
            ui.card(
                ui.card_header(
                    ui.h4("📏 Measurements", class_="mb-0"),
                    class_="bg-info text-white"
                ),
                ui.navset_pill(
                    ui.nav_panel(
                        "Add New",
                        ui.input_date("m_date", "Date", value=date.today()),
                        ui.input_numeric("m_weight", "Weight (kg)", 80, min=0, step=0.1),
                        ui.layout_column_wrap(
                            ui.div(
                                ui.input_numeric("m_neck", "Neck", 0, min=0, step=0.1),
                                ui.input_checkbox("m_neck_missing", "Skip", False),
                            ),
                            ui.div(
                                ui.input_numeric("m_shoulder", "Shoulder", 0, min=0, step=0.1),
                                ui.input_checkbox("m_shoulder_missing", "Skip", False),
                            ),
                            ui.div(
                                ui.input_numeric("m_chest", "Chest", 0, min=0, step=0.1),
                                ui.input_checkbox("m_chest_missing", "Skip", False),
                            ),
                            ui.div(
                                ui.input_numeric("m_waist", "Waist", 0, min=0, step=0.1),
                                ui.input_checkbox("m_waist_missing", "Skip", False),
                            ),
                            width=1/2,
                        ),
                        ui.layout_column_wrap(
                            ui.div(
                                ui.input_numeric("m_biceps", "Biceps", 0, min=0, step=0.1),
                                ui.input_checkbox("m_biceps_missing", "Skip", False),
                            ),
                            ui.div(
                                ui.input_numeric("m_thigh", "Thigh", 0, min=0, step=0.1),
                                ui.input_checkbox("m_thigh_missing", "Skip", False),
                            ),
                            ui.div(
                                ui.input_numeric("m_calf", "Calf", 0, min=0, step=0.1),
                                ui.input_checkbox("m_calf_missing", "Skip", False),
                            ),
                            width=1/2,
                        ),
                        ui.input_action_button("btn_add_meas", "Add Measurements", class_="btn-info w-100"),
                    ),
                    ui.nav_panel(
                        "Edit",
                        ui.input_selectize("m_pick", "Select Entry", choices=[], width="100%"),
                        ui.output_ui("meas_loaded_info"),
                        ui.layout_column_wrap(
                            ui.input_action_button("btn_load_meas", "Load", class_="btn-info"),
                            ui.input_action_button("btn_save_meas", "Save", class_="btn-success"),
                            ui.input_action_button("btn_del_meas", "Delete", class_="btn-danger"),
                            width=1/3,
                        ),
                    ),
                ),
            ),
            col_widths=cast(Any, {"lg": [4, 4, 4]}),
        ),
        ui.layout_columns(
            ui.card(
                ui.card_header(
                    ui.h4("🔬 InBody", class_="mb-0"),
                    class_="bg-warning text-dark"
                ),
                ui.navset_pill(
                    ui.nav_panel(
                        "Add New",
                        ui.input_date("ib_date", "Date", value=date.today()),
                        ui.layout_column_wrap(
                            ui.input_numeric("ib_score", "InBody Score", 80, min=0, max=100, step=1),
                            ui.input_numeric("ib_weight", "Weight (kg)", 80, min=0, step=0.1),
                            width=1/2,
                        ),
                        ui.tags.h6("Body Composition", style="margin-top:15px; margin-bottom:10px;"),
                        ui.layout_column_wrap(
                            ui.input_numeric("ib_muscle_total", "Skeletal Muscle (kg)", 35, min=0, step=0.1),
                            ui.input_numeric("ib_fat_total", "Body Fat (kg)", 15, min=0, step=0.1),
                            ui.input_numeric("ib_fat_pct", "Body Fat (%)", 18, min=0, max=100, step=0.1),
                            ui.input_numeric("ib_visceral", "Visceral Fat Level", 8, min=0, step=1),
                            ui.input_numeric("ib_bmr", "BMR (kcal)", 1800, min=0, step=1),
                            width=1/3,
                        ),
                        ui.tags.h6("Segmental Muscle (kg)", style="margin-top:15px; margin-bottom:10px;"),
                        ui.layout_column_wrap(
                            ui.input_numeric("ib_muscle_rarm", "Right Arm", 3.5, min=0, step=0.1),
                            ui.input_numeric("ib_muscle_larm", "Left Arm", 3.5, min=0, step=0.1),
                            ui.input_numeric("ib_muscle_trunk", "Trunk", 22, min=0, step=0.1),
                            ui.input_numeric("ib_muscle_rleg", "Right Leg", 8, min=0, step=0.1),
                            ui.input_numeric("ib_muscle_lleg", "Left Leg", 8, min=0, step=0.1),
                            width=1/3,
                        ),
                        ui.tags.h6("Segmental Fat (kg)", style="margin-top:15px; margin-bottom:10px;"),
                        ui.layout_column_wrap(
                            ui.input_numeric("ib_fat_rarm", "Right Arm", 1.0, min=0, step=0.1),
                            ui.input_numeric("ib_fat_larm", "Left Arm", 1.0, min=0, step=0.1),
                            ui.input_numeric("ib_fat_trunk", "Trunk", 8, min=0, step=0.1),
                            ui.input_numeric("ib_fat_rleg", "Right Leg", 2.5, min=0, step=0.1),
                            ui.input_numeric("ib_fat_lleg", "Left Leg", 2.5, min=0, step=0.1),
                            width=1/3,
                        ),
                        ui.input_text_area("ib_notes", "Notes", "", rows=2),
                        ui.input_action_button("btn_add_ib", "Add InBody", class_="btn-warning w-100"),
                    ),
                    ui.nav_panel(
                        "Edit",
                        ui.input_selectize("ib_pick", "Select Entry", choices=[], width="100%"),
                        ui.output_ui("ib_loaded_info"),
                        ui.layout_column_wrap(
                            ui.input_action_button("btn_load_ib", "Load", class_="btn-info"),
                            ui.input_action_button("btn_save_ib", "Save", class_="btn-success"),
                            ui.input_action_button("btn_del_ib", "Delete", class_="btn-danger"),
                            width=1/3,
                        ),
                    ),
                ),
            ),
            col_widths=cast(Any, {"lg": [12]}),
        ),
    ),
    ui.nav_spacer(),
    ui.nav_menu(
        "ℹ️ Info",
        ui.nav_control(
            ui.a("GitHub", href="https://github.com", target="_blank", class_="nav-link")
        ),
        align="right",
    ),
    title="🏋️ Gym Tracker Pro",
    #fillable=True,
    theme=ui.Theme("cosmo"),
)
