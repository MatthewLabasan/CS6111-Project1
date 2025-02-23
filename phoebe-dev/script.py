"""
    https://github.com/googleapis/google-api-python-client/blob/main/samples/customsearch/main.py
"""

import pprint

from googleapiclient.discovery import build


def main():
    # Build a service object for interacting with the API. Visit
    # the Google APIs Console <http://code.google.com/apis/console>
    # to get an API key for your own application.
    service = build(
        "customsearch", "v1", developerKey="748ba10aef8b944fa"
    )

    res = (
        service.cse()
        .list(
            q="per se",
            cx="AIzaSyASDjQb1R4dXsi5a49c9UBFtXB8jVNjmD0",
        )
        .execute()
    )
    pprint.pprint(res)


if __name__ == "__main__":
    main()