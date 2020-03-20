class ConfigManager:

    def __init__(context, endpoints):
        context.endpoints = endpoints
        context.debug_mode = True
        context.host = '0.0.0.0'
        context.port = '5000'