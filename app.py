import dotenv; dotenv.load_dotenv()
from shiny import App
from gymtracker.ui import app_ui
from gymtracker.server import server

app = App(app_ui, server)