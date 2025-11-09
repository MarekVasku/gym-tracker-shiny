import dotenv
from shiny import App

from gymtracker.server import server
from gymtracker.ui import app_ui

dotenv.load_dotenv()

app = App(app_ui, server)