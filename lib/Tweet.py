__author__ = 'Daniel Palomino'

class Tweet:

    id = 0
    user = ''
    date = ''
    lang = ''
    content = ''
    polarity = ''

    def __init__(self, tweet_id, user, date, lang, content, polarity):
        self.id = tweet_id
        self.user = user
        self.date = date
        self.lang = lang
        self.content = content
        self.polarity = polarity
