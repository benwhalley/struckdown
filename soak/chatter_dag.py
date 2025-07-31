from chatter import chatter
from chatter.return_type_models import ACTION_LOOKUP


def chatter_dag(**kwargs):
    """
    # example usage to return Codes
    chatter_dag(multipart_prompt="Software as a service (SaaS /sæs/[1]) is a cloud computing service model where the provider offers use of application software to a client and manages all needed physical and software resources.[2] SaaS is usually accessed via a web application. Unlike other software delivery models, it separates 'the possession and ownership of software from its use'.[3] SaaS use began around 2000, and by 2023 was the main form of software application deployment.\n\n What is the theme of this text [[Codes:code]]")
    """
    from soak.models import Code, Theme, Themes, CodeList  # canonical import

    action_lookup = ACTION_LOOKUP.copy()
    action_lookup.update(
        {
            "theme": Theme,
            "code": Code,
            "themes": Themes,
            "codes": CodeList,
        }
    )
    action_lookup = dict(action_lookup)
    return chatter(**kwargs, action_lookup=action_lookup)
