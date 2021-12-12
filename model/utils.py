# -- coding: utf-8 --

def construct_feed_dict(features1, features2, month, day, hour, labels, placeholders):
    """Construct feed dictionary."""
    feed_dict = dict()
    feed_dict.update({placeholders['month']: month})
    feed_dict.update({placeholders['day']: day})
    feed_dict.update({placeholders['hour']: hour})
    feed_dict.update({placeholders['labels']: labels})
    feed_dict.update({placeholders['features1']: features1})
    feed_dict.update({placeholders['features2']: features2})
    return feed_dict