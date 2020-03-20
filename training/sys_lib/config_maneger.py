class ConfigManager:

    def __init__(context, api_config):
        context.flow_name = api_config["flow_name"]
        context.service_name = api_config["service_name"]
        context.debug_mode = api_config["debug_mode"]
        context.host = api_config["host"]
        context.port = api_config["port"]