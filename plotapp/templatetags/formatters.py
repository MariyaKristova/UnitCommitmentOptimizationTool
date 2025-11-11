from django import template
register = template.Library()

@register.filter
def space_thousands(value):
    try:
        s = f"{value:,.2f}"
        return " ".join(s.split(","))
    except:
        return value
