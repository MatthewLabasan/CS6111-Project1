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
        "customsearch", "v1", developerKey="xx"
    )

    res = (
        service.cse()
        .list(
            q="per se",
            cx="xx",
        )
        .execute()
    )
    pprint.pprint(res)


if __name__ == "__main__":
    main()