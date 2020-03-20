class Context:
    def __init__(ctx, resource, parser, config, logging):
        ctx.resource = resource
        ctx.parser = parser
        ctx.config = config
        ctx.logging = logging