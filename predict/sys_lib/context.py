class Context:
    def __init__(ctx, resource, parser, config, utils, logging):
        ctx.resource = resource
        ctx.parser = parser
        ctx.config = config
        ctx.utils = utils
        ctx.logging = logging