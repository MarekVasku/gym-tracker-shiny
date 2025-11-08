from __future__ import annotations
from datetime import date, datetime
from shiny import ui
from .utils import BIG3

_NOW = datetime.now()
_HOUR_CHOICES = [f"{i:02d}" for i in range(24)]
_MIN_CHOICES = [f"{i:02d}" for i in range(60)]

app_ui = ui.page_navbar(
    ui.nav_panel(
        "📊 Dashboard",
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
                            multiple=True,
                            width="100%"
                        ),
                        ui.input_checkbox("rm_mode_tm", "Training Max (90%)", False),
                        width="220px",
                        bg="#f8f9fa"
                    ),
                    ui.output_plot("plot_1rm", height="400px"),
                ),
            ),
            ui.card(
                ui.card_header(
                    ui.h4("⚖️ Bodyweight Tracking", class_="mb-0"),
                    class_="bg-success text-white"
                ),
                ui.output_plot("plot_bw", height="400px"),
            ),
            col_widths={"lg": [7, 5]},
        ),
        ui.layout_columns(
            ui.card(
                ui.card_header(
                    ui.h4("📏 Body Measurements", class_="mb-0"),
                    class_="bg-info text-white"
                ),
                ui.output_plot("plot_meas", height="380px"),
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
            col_widths={"lg": [7, 5]},
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
                        ui.layout_column_wrap(
                            ui.input_action_button("btn_load_meas", "Load", class_="btn-info"),
                            ui.input_action_button("btn_save_meas", "Save", class_="btn-success"),
                            ui.input_action_button("btn_del_meas", "Delete", class_="btn-danger"),
                            width=1/3,
                        ),
                    ),
                ),
            ),
            col_widths={"lg": [4, 4, 4]},
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
    fillable=True,
    theme=ui.Theme("cosmo"),
)
