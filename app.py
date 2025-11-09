import dotenv
from shiny import App
from gymtracker.ui import app_ui
from gymtracker.server import server

dotenv.load_dotenv()

app = App(app_ui, server)