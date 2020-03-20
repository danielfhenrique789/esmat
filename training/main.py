import logging
from flask import Flask
from flask_restful import reqparse, abort, Api, Resource
from route import get_routes
from sys_lib.config_maneger import ConfigManager
import sys_lib.utils as utils
from sys_lib.context import Context
from configs.api_config import api_config

app = Flask(__name__)
api = Api(app)
logging.basicConfig(format='%(asctime)s - %(levelname)s:%(message)s', level=logging.DEBUG)
config = ConfigManager(api_config)

parser = reqparse.RequestParser()

ctx = Context(Resource, parser, config, utils, logging)

##
## Actually setup the Api resource routing here
##
routes_args = get_routes(ctx)
for route_args in routes_args:
    ctx.logging.info("Setting route: " + route_args["endpoint"])
    api.add_resource(route_args["class"], route_args["endpoint"])

if __name__ == '__main__':
    app.run(debug=config.debug_mode, host=config.host, port=config.port)